# run_deepeval_from_json.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from core.evaluation.eval_script import validate_list
from core.evaluation.io_utils import load_json  # пример LLM-judge метрики


def load_items(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list (or dict with key 'items'), got: {type(data)}")
    return data


def to_test_case(item: dict[str, Any]) -> LLMTestCase:
    raw = item.get("raw_state") or {}
    user_input = raw.get("user_input") or item.get("input")
    if not user_input:
        raise ValueError(f"Missing input: benchmark_id={item.get('benchmark_id')}")

    decision = item.get("model_decision")

    actual_output = item.get("response_message") if decision == "chat_response" else item.get("generated_code")
    if actual_output is None:
        actual_output = ""  # иногда удобно не падать, а оценить как пустой ответ

    # если у тебя реально есть expected_output — подхватим; если нет, оставим None
    expected_output = item.get("expected_output") or raw.get("expected_output")

    # всё остальное — в metadata, чтобы видеть в отчёте/в UI
    metadata = {
        "benchmark_id": item.get("benchmark_id"),
        "benchmark_file": item.get("benchmark_file"),
        "expected_decision": item.get("expected_decision"),
        "model_decision": item.get("model_decision"),
        "timing_info": item.get("timing_info") or raw.get("timing_info"),
        "data_schema": (raw.get("metadata") or {}),
    }

    return LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        expected_output=expected_output,
        metadata=metadata,
    )


def decision_accuracy(items: list[dict[str, Any]]) -> float:
    good = 0
    total = 0
    for it in items:
        exp = it.get("expected_decision")
        pred = it.get("model_decision")
        if exp is None or pred is None:
            continue
        total += 1
        good += int(exp == pred)
    return good / total if total else float("nan")


judge = GeminiModel(model="gemini-2.5-flash", temperature=0, api_key="my_api")


def main() -> None:
    inference_res_path = "core/evaluation/inference_results/infer_3_decision_enhancing_qwen3-instruct-30b.json"
    items = validate_list(load_json(Path(inference_res_path)))
    test_cases = [to_test_case(it) for it in items]

    # 1) детерминированная метрика по decision (без LLM-as-judge)
    acc = decision_accuracy(items)
    print(f"[Decision accuracy] {acc:.4f}  (computed from expected_decision/model_decision)")

    # 2) пример LLM-as-judge метрики по тексту (вызовет judge-провайдера)
    # если expected_output отсутствует — AnswerRelevancy всё равно работает (input vs actual_output)
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=judge),
        # при желании добавишь FaithfulnessMetric и т.п.
    ]

    # evaluate() делает локальный прогон и (если залогинена) загрузит test run в Confident AI
    # после завершения обычно появляется ссылка на sharable report :contentReference[oaicite:4]{index=4}
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    # 3) локально сохраняем "нормальный" артефакт рядом (своего формата)
    out = {
        "decision_accuracy": acc,
        "n_items": len(items),
        "n_test_cases": len(test_cases),
    }
    Path("local_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved local_summary.json")


if __name__ == "__main__":
    main()

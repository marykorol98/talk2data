# workflow.py

import json
import time
from vllm import SamplingParams
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import JsonOutputParser
import time
from core.config import settings
# Import prompt templates and schemas
from core.prompts import (
    DECIDE_ACTION_PROMPT,
    CHAT_RESPONSE_PROMPT,
    CODE_GENERATION_PROMPT,
)
from core.schemas import AgentState, Decision
import torch
import atexit
import copy

from core.models import get_llm, get_tokenizer
from string import Template
import logging

logger = logging.getLogger(__name__)


def init_tokenizer_only():
    global tokenizer
    tokenizer = get_tokenizer()


def llm_init(max_retries: int = 3, retry_delay: float = 5.0):
    """
    Прогрев моделей с логированием времени и повторными попытками.
    """
    global llm
    global tokenizer

    start_time = time.perf_counter()
    logger.info("Starting model warm-up...")

    for attempt in range(1, max_retries + 1):
        try:
            tokenizer = get_tokenizer()
            llm = get_llm()
            elapsed = round(time.perf_counter() - start_time, 2)
            logger.info(f"Models ready (loaded in {elapsed} sec on attempt {attempt})")
            return
        except Exception as e:
            logger.error(f"LLM init failed on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} sec...")
                time.sleep(retry_delay)
            else:
                elapsed = round(time.perf_counter() - start_time, 2)
                logger.critical(
                    f"Failed to initialize models after {max_retries} attempts (elapsed {elapsed}s)"
                )
                raise


DECIDE_ACTION_DEFAULT = "chat_response"


def format_prompt(
    messages_template: list, state: AgentState, metadata_fields: dict | None = None
) -> str:
    """Format chat template as flat text for code generation."""
    metadata_fields = metadata_fields or {}
    local_prompt = copy.deepcopy(messages_template)

    mapping = {
        "input": str(state.get("user_input", "")),
        "history": str(state.get("conversation_history", "")),
        "metadata": str(state.get("metadata", {})),
    }
    if metadata_fields:
        mapping.update({k: str(v) for k, v in metadata_fields.items()})

    formatted_messages = []
    for msg in local_prompt:
        try:
            tmpl = Template(msg["content"])
            content = tmpl.safe_substitute(mapping)
        except Exception as e:
            logger.info(f"[format_prompt] Template substitution error: {e}")
            # fallback — оставляем оригинал
            content = msg["content"]

        formatted_messages.append({"role": msg["role"], "content": content})

    return tokenizer.apply_chat_template(
        formatted_messages, tokenize=False, add_generation_prompt=True
    )


def decide_action(state: AgentState) -> AgentState:
    """Decision node with enhanced logging using print and timing."""
    start = time.perf_counter()
    logger.info(f"[decide_action] Starting with state: {state}")

    parser = JsonOutputParser(pydantic_object=Decision)

    try:
        # Формируем prompt
        prompt = format_prompt(DECIDE_ACTION_PROMPT, state)
        logger.info(f"[decide_action] Formatted prompt: {prompt}")

        # Настраиваем параметры сэмплирования
        sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.0,  # полная детерминированность
            top_p=1.0,  # отключает сэмплирование по вероятностям
            stop=["</s>", "\n\n", "\nUser:"],  # можно добавить безопасные стоп-токены
            repetition_penalty=1.0,  # не трогаем (нет смысла для коротких ответов)
        )

        logger.info(f"[decide_action] Sampling parameters: {sampling_params}")

        # Генерация ответа от LLM
        outputs = llm.generate([prompt], sampling_params)
        raw_response = outputs[0].outputs[0].text.strip()
        logger.info(f"[decide_action] Raw LLM response: {raw_response}")

        # Парсим результат
        decision = parser.parse(raw_response)
        logger.info(f"[decide_action] Parsed decision: {decision}")

    except Exception as e:
        logger.info(f"[decide_action] Decision error: {e}")
        decision = {"action": DECIDE_ACTION_DEFAULT}
        logger.info(f"[decide_action] Defaulting decision to: {decision}")

    # Считаем время выполнения
    elapsed = time.perf_counter() - start
    timing_info = state.get("timing_info", {})
    timing_info["decide_action_sec"] = round(elapsed, 4)
    state["timing_info"] = timing_info
    logger.info(f"[decide_action] Time elapsed: {elapsed:.4f} sec")

    # Сохраняем решение в состоянии
    state["decision"] = decision
    logger.info(f"[decide_action] Final state: {state}")

    return state


def route_action(state: AgentState) -> str:
    """Helper to decide next node based on 'decision.action'."""
    try:
        return state.get("decision", {}).get("action", DECIDE_ACTION_DEFAULT)
    except Exception:
        return "chat_response"


async def generate_code_node(state: AgentState):
    """Streaming code generation using vLLM, returning both stream tokens and final code block."""
    import httpx
    import time

    start = time.perf_counter()

    code_prompt = format_prompt(CODE_GENERATION_PROMPT, state)
    accumulated = ""  # сюда накапливаем текст для итоговой обработки

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "http://localhost:6001/v1/completions",
            json={
                "model": settings.LLM_MODEL_NAME,
                "prompt": code_prompt,
                "stream": True,
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "stop": ["<|", "</s>"],
            }
        ) as resp:

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break

                data = json.loads(payload)
                token = data["choices"][0]["text"]

                accumulated += token  # сохраняем токены в общий буфер

                # STREAM на фронт
                yield {"stream_token": token, "type": "code"}

    # === Когда генерация закончена, формируем финальный код ===

    # Вырезаем ```python ... ```
    if "```python" in accumulated:
        code_block = accumulated.split("```python")[-1].split("```")[0].strip()
    else:
        # fallback — всё содержимое
        code_block = accumulated.strip()

    elapsed = time.perf_counter() - start

    timing_info = state.get("timing_info", {})
    timing_info["generate_code_sec"] = round(elapsed, 4)
    state["timing_info"] = timing_info

    state.update(
        {
            "generated_code": code_block,
            "response_message": "Here is the generated code:",
        }
    )

    # После стрима финальный state
    yield state



async def generate_chat_response_node(state: AgentState):
    """Стриминг токенов от vLLM вместо синхронной генерации."""
    import httpx
    chat_prompt = format_prompt(CHAT_RESPONSE_PROMPT, state)

    state["response_message"] = ""   # чтобы накапливать полный ответ

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "http://localhost:6001/v1/completions",  # vLLM endpoint
            json={
                "model": settings.LLM_MODEL_NAME,
                "prompt": chat_prompt,
                "stream": True,
                "max_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["</s>"],
            }
        ) as resp:

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                payload = line[len("data:") :].strip()

                if payload == "[DONE]":
                    break

                data = json.loads(payload)
                token = data["choices"][0]["text"]

                # накапливаем финальный текст для истории
                state["response_message"] += token

                # STREAM наружу (в handler_converse → RabbitMQ)
                yield {"stream_token": token}

    # финальное состояние возвращаем при завершении узла
    yield state



def create_workflow():
    builder = StateGraph(AgentState)
    builder.add_node("decide_action", decide_action)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("generate_chat_response", generate_chat_response_node)
    builder.add_conditional_edges(
        "decide_action",
        route_action,
        {"code_generation": "generate_code", "chat_response": "generate_chat_response"},
    )
    builder.add_edge("generate_code", END)
    builder.add_edge("generate_chat_response", END)
    builder.set_entry_point("decide_action")
    return builder.compile()


def safe_destroy_process_group():
    """
    Безопасный shutdown моделей. Позволяет избежать утечек на GPU
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            logger.error(f"Error during destroy_process_group: {e}")


atexit.register(safe_destroy_process_group)
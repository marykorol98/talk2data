# workflow.py

import atexit
import copy
import json
import logging
import time
from string import Template

import torch
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
from vllm import SamplingParams

from core.models import get_llm, get_tokenizer

# Import prompt templates and schemas
from core.prompts import (
    CHAT_RESPONSE_PROMPT,
    CODE_BEGIN,
    CODE_END,
    CODE_GENERATION_PROMPT,
    DECIDE_ACTION_PROMPT,
    PLAN_END,
    build_code_prompt,
    build_plan_prompt,
)
from core.schemas import AgentState, Decision
from core.tools import process_output_code, process_plan

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

    logger.info(f"state: {state}")

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

def generate_plan(state: AgentState) -> AgentState:
    start = time.perf_counter()
    prompt = tokenizer.apply_chat_template(build_plan_prompt(state), tokenize=False, add_generation_prompt=True)

    params = SamplingParams(
        max_tokens=256,
        temperature=0.0,   # JSON-стабильность
        top_p=1.0,
        seed=42,
        # IMPORTANT for Qwen: do NOT stop on "<|"
        stop=[PLAN_END],   # безопасно: стопим по нашему маркеру окончания
    )

    out = llm.generate([prompt], params)
    raw = (out[0].outputs[0].text or "").strip()

    plan = process_plan(raw)
    
    if not plan:
        plan = {
            "intent": "stats",
            "columns_used": [],
            "ops": ["fallback: plan JSON parse failed"],
            "plot_spec": {},
            "notes": "plan generation failed; proceed conservatively",
        }

    elapsed = time.perf_counter() - start
    timing = state.get("timing_info", {})
    timing["generate_plan_sec"] = round(elapsed, 4)
    state["timing_info"] = timing

    state["plan"] = plan
    return state


def generate_code_node(state: AgentState) -> AgentState:
    start = time.perf_counter()
    state = generate_plan(state)
    
    prompt = tokenizer.apply_chat_template(build_code_prompt(state), tokenize=False, add_generation_prompt=True)

    params = SamplingParams(
        max_tokens=640,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.05,
        presence_penalty=0.0,  # стабильнее при temp=0
        seed=42,
        stop=[CODE_END],       # стопим по нашему маркеру
    )

    out = llm.generate([prompt], params)
    raw = (out[0].outputs[0].text or "").strip()

    code = process_output_code(raw)
    
    elapsed = time.perf_counter() - start
    timing = state.get("timing_info", {})
    timing["generate_code_sec"] = round(elapsed, 4)
    state["timing_info"] = timing

    state["generated_code"] = code.strip()
    state["response_message"] = "Here's the generated code:"
    return state


def generate_chat_response_node(state: AgentState) -> AgentState:
    """Chat response generation with TTS integration, measure time."""
    start = time.perf_counter()
    chat_prompt = format_prompt(CHAT_RESPONSE_PROMPT, state)
    logger.info(f"[chat_prompt] Prompt for text generation: {chat_prompt}")
    sampling_params = SamplingParams(
        max_tokens=200,
        temperature=0.0,  # 0.7
        top_p=0.9,
        stop=["</s>"],
        seed=42,
    )

    outputs = llm.generate([chat_prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    elapsed_llm = time.perf_counter() - start
    logger.info(f"[test_response] Generated text response: {response}")
    # Attempt TTS
    # tts_start = time.perf_counter()
    # audio_b64 = None
    # try:
    #     audio_bytes = text_to_speech(response)
    #     audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    # except Exception as e:
    #     logger.info(f"TTS Error: {str(e)}")
    # tts_elapsed = time.perf_counter() - tts_start

    timing_info = state.get("timing_info", {})
    timing_info["generate_chat_response_sec"] = round(elapsed_llm, 4)
    # timing_info["tts_sec"] = round(tts_elapsed, 4)
    state["timing_info"] = timing_info

    # state["response_audio"] = audio_b64
    state["response_message"] = response
    return state


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

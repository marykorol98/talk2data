# worker.py
import asyncio
import base64
import json
from pathlib import Path
import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError
# from models import whisper_model
from core.workflow import create_workflow, llm_init
from core.schemas import ConversationRequest
import tempfile

import logging
from core.config import settings
from voice2text.whisper_model import Voice2Text

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Инициализация соединения
try:
    credentials = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=settings.RABBITMQ_HOST,
            credentials=credentials,
            heartbeat=60
        )
    )

    if not connection.is_open:
        raise ConnectionError("RabbitMQ connection failed silently")

    channel = connection.channel()

    channel.queue_declare(queue=settings.TASK_QUEUE, durable=True)
    channel.queue_declare(queue=settings.RESPONSE_QUEUE, durable=True)

    print("Successfully connected to RabbitMQ")

except AMQPConnectionError as conn_err:
    print(f"RabbitMQ connection failed: {conn_err}")
except AMQPChannelError as channel_err:
    print(f"Channel error: {channel_err}")
except Exception as e:
    print(f"Unexpected error: {e}")

logger.info("Worker connected to RabbitMQ")

llm_init()

whisper_model = Voice2Text()


def save_result_locally(result: dict, filename: str = "result.json") -> None:
    """Сохраняет result в локальный JSON-файл рядом с текущим модулем."""
    try:
        file_path = Path(__file__).with_name(filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logger.info("Result saved to %s", file_path)
    except Exception as e:
        logger.exception("Failed to save result: %s", e)


def load_result_locally(filename: str = "result.json") -> dict:
    """Загружает сохранённый result из JSON-файла."""
    file_path = Path(__file__).with_name(filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load result: %s", e)
        return {}


async def handle_converse(data: dict, ch):
    """LLM pipeline с потоком токенов через RabbitMQ."""
    req = ConversationRequest(**data)
    workflow = create_workflow()

    initial_state = {
        "user_input": req.user_input,
        "metadata": req.metadata,
        "conversation_history": req.conversation_history,
        "generated_code": None,
        "response_message": "",
        "response_audio": None,
        "decision": None,
        "timing_info": {},
    }

    async for event in workflow.astream(initial_state):

        # 1) STREAM TOKENS (частичный ответ)
        if "stream_token" in event:
            ch.basic_publish(
                exchange=settings.EXCHANGE,
                routing_key=settings.ROUTING_KEY,
                body=json.dumps({
                    "status": "stream",
                    "token": event["stream_token"],
                    "project_id": data.get("project_id")
                })
            )
            continue

        # 2) FINISHED NODE (финальное состояние)
        if isinstance(event, dict) and "response_message" in event:
            # это итоговое состояние
            final = event

            updated_history = req.conversation_history + [
                {
                    "user": req.user_input,
                    "system": final.get("response_message") or final.get("generated_code"),
                }
            ]

            result = {
                "status": "done",
                "task": "llm_agent_response",
                "result": {
                    "message": final.get("response_message"),
                    "code": final.get("generated_code"),
                    "updated_history": updated_history,
                    "timing": final.get("timing_info", {}),
                },
                "project_id": data.get("project_id")
            }

            ch.basic_publish(
                exchange=settings.EXCHANGE,
                routing_key=settings.ROUTING_KEY,
                body=json.dumps(result)
            )

            return



def handle_transcribe(data: dict):
    """Обработка аудио (Whisper)."""
    try:
        audio_bytes = base64.b64decode(data["file_bytes"])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path)

        return {
            "status": "done",
            "task": "transcribe",
            "result": {"text": result["text"]},
        }
    except Exception as e:
        return {"status": "error", "task": "transcribe", "error": str(e)}


task_mapping = {"converse": handle_converse, "transcribe": handle_transcribe}


def callback(ch, method, properties, body):
    """Основная функция обработки входящих задач (с поддержкой async-хэндлеров)."""
    try:
        msg = json.loads(body)
        task_type = msg.get("task")
        data = msg.get("data", {})

        handler = task_mapping.get(task_type)
        logger.info(f"Received task: {task_type}")

        if handler is None:
            # неизвестная задача → отправляем ошибку
            response = {"status": "error", "error": f"Unknown task: {task_type}"}

        else:
            # --- ВАЖНО ---
            # handle_converse теперь async def → его нужно await-ить
            # но callback синхронный, поэтому используем asyncio.run()
            if asyncio.iscoroutinefunction(handler):
                response = asyncio.run(handler(data, ch))
            else:
                response = handler(data)

        # добавляем project_id
        if isinstance(response, dict):
            response["project_id"] = data.get("project_id")

        # финальный publish (важно: потоковые токены handle_converse уже отправил сам)
        if response:
            ch.basic_publish(
                exchange=settings.EXCHANGE,
                routing_key=settings.ROUTING_KEY,
                body=json.dumps(response),
                mandatory=True,
            )

        logger.info(
            f"Sent response for {task_type}: {response.get('status')} {response.get('error', '')}"
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.error(f"Callback error: {e}", exc_info=True)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# Подписываемся на очередь задач
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=settings.TASK_QUEUE, on_message_callback=callback)

logger.info("Worker started. Waiting for tasks...")
logger.info(f"Listening to queue: {settings.TASK_QUEUE}")
channel.start_consuming()

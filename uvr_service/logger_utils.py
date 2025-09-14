from loguru import logger
import sys, os, socket, uuid
from fastapi import Request
import time
from contextvars import ContextVar
import json

# Config
SERVICE_NAME = os.getenv("SERVICE_NAME", "uvr-service")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
HOSTNAME = socket.gethostname()

# Per-request correlation id
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")

# Formats
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "CID={extra[correlation_id]} | <level>{message}</level>"
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
    "{name}:{function}:{line} | "
    "CID={extra[correlation_id]} | {message}"
)


def setup_logger():
    logger.remove()

    logger.add(
        sys.stdout,
        format=CONSOLE_FORMAT,
        colorize=True,
        level="INFO",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    logger.add(
        f"{LOG_DIR}/uvr-service.log",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        format=FILE_FORMAT,
        level="DEBUG",
        enqueue=True,
    )

    logger.add(
        f"{LOG_DIR}/uvr-service.json",
        level="INFO",
        serialize=True,
        rotation="200 MB",
        retention="15 days",
        enqueue=True,
    )

    # Add dynamic correlation_id injection via filter
    def add_correlation_id(record):
        record["extra"]["correlation_id"] = correlation_id.get("-")
        record["extra"]["service"] = SERVICE_NAME
        record["extra"]["environment"] = ENVIRONMENT
        record["extra"]["hostname"] = HOSTNAME
        return True

    logger.configure(patcher=add_correlation_id)

    return logger

# ──────────────────────────────────────────────
# Middleware for logger and timing
# ──────────────────────────────────────────────
async def logger_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    req_id = None
    body = None

    if request.method == "POST":
        try:
            content_type = request.headers.get("content-type", "").lower()
            body = await request.body()  # read once

            if "application/json" in content_type and body:
                try:
                    data = json.loads(body)
                    req_id = data.get("req_id")
                except Exception as e:
                    logger.error(f"Failed parsing JSON body: {e}")

            elif (
                "application/x-www-form-urlencoded" in content_type
                or "multipart/form-data" in content_type
            ):
                try:
                    form_data = await request.form()
                    req_id = form_data.get("req_id")
                except Exception as e:
                    logger.error(f"Failed parsing form body: {e}")

            if body is not None:
                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                request._receive = receive

        except Exception as e:
            logger.error(f"Error processing request body: {e}")

    # If no req_id found, generate one
    if not req_id:
        req_id = str(uuid.uuid4())

    # Set correlation id before endpoint runs
    set_correlation_id(req_id)

    # Call actual route handler
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    logger.info(
        f"Request completed | path={request.url.path} | req_id={req_id} | time_taken={process_time:.4f}s"
    )

    return response


# Global logger (import everywhere)
logger = setup_logger()


# Utility: set correlation id per request
def set_correlation_id(c_id: str | None = None):
    if not c_id:
        c_id = str(uuid.uuid4())
    correlation_id.set(c_id)

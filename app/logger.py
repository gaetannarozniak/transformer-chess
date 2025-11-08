import structlog
import logging
import sys

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.DEBUG)
logger = structlog.get_logger()

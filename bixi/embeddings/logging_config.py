import logging.config
from typing import Literal

__all__ = ['configure_logging']

# 定义日志格式
LOG_FORMAT = "%(asctime)s [%(thread)d] [%(levelname)7s]: %(message)s [%(filename)s - %(funcName)s:%(lineno)d]"

# 定义日志颜色
LOG_COLORS = {
    logging.DEBUG: "\033[36m{}\033[0m",
    logging.WARNING: "\033[33m{}\033[0m",
    logging.ERROR: "\033[31m{}\033[0m",
    logging.CRITICAL: "\033[35m{}\033[0m",
}

# 定义日志格式器
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if record.levelno in LOG_COLORS:
            msg = LOG_COLORS[record.levelno].format(msg)
        return msg


logging_config_dict = {
    'version': 1,
    'formatters': {
        'colored': {
            '()': ColoredFormatter,
            'format': LOG_FORMAT
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'colored',
        }
    },
    "loggers": {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    }
}

def configure_logging(logger_name: str, level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
    loggers: dict = logging_config_dict.get("loggers")
    loggers.setdefault(logger_name, {
        'handlers': ['console'],
        'level': level,
        'propagate': False,
    })
    logging.config.dictConfig(logging_config_dict)
    return logging_config_dict

def get_logging_configuration() -> dict:
    return logging_config_dict
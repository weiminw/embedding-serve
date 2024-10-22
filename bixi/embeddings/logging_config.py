import logging

from logging.handlers import RotatingFileHandler

# 定义日志格式
LOG_FORMAT = "%(asctime)s [%(thread)d] [%(levelname)7s]: %(message)s [%(filename)s - %(funcName)s:%(lineno)d]"

# 配置日志颜色
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

# 创建日志处理器
file_handler = RotatingFileHandler("log.log", maxBytes=1024*1024*10, backupCount=5)
console_handler = logging.StreamHandler()

# 创建日志格式器
file_formatter = logging.Formatter(LOG_FORMAT)
console_formatter = ColoredFormatter(LOG_FORMAT)

# 设置格式器
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# 配置日志输出
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler]
)

# 创建一个全局的日志对象
logger = logging.getLogger()
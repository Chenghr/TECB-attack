import os
import json
import time
import platform
import sys
import logging
from loguru import logger


def logging_config(args, process_id):
    # customize the log format
    while logging.getLogger().handlers:
        logging.getLogger().handlers.clear()
    
    console = logging.StreamHandler()
    
    if args.level == "INFO":
        console.setLevel(logging.INFO)
    elif args.level == "DEBUG":
        console.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError
    
    formatter = logging.Formatter(
        str(process_id)
        + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
    )

    console.setFormatter(formatter)

    # Create an instance
    logging.getLogger().addHandler(console)
    # logging.getLogger().info("test")
    logging.basicConfig()
    logger = logging.getLogger()

    if args.level == "INFO":
        logger.setLevel(logging.INFO)
    elif args.level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError
    
    logging.info(args)


def formatter(record):
    format_string = get_log_format(record["level"].name)
    return format_string.format(**record)


def get_log_format(level_name):
    # 默认格式
    time_format = "<green>{time:MM-DD/HH:mm:ss}</>"
    lvl_format = "<lvl><i>{level:^5}</></>"
    rcd_format = "<cyan>{file}:{line:}</>"
    msg_format = "<lvl>{message}</>"

    # 针对特定级别的自定义格式
    if level_name in ["WARNING", "CRITICAL"]:
        lvl_format = "<l>" + lvl_format + "</>"

    return "|".join([time_format, lvl_format, rcd_format, msg_format]) + "\n"


def setup_logger(args):
    logger.remove()
    level = getattr(args, 'level', 'INFO')

    # 添加终端日志处理器
    logger.add(
        sys.stderr,
        format=formatter,
        colorize=True,
        enqueue=True,
        level=level
    )

    # 添加文件日志处理器
    log_dir =  os.path.join(args.save, args.log_file_name)
    logger.add(
        log_dir,
        format=formatter,
        colorize=False, 
        enqueue=True,
        level=level
    )

    return logger


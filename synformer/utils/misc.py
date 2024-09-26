import datetime
import os
import re
import time


def get_config_name(config_path):
    config_path = str(config_path)
    return os.path.basename(config_path)[: os.path.basename(config_path).rfind(".")]


def get_experiment_name(model_name: str, version: str, version_time: datetime.datetime):
    # Replace illegal filename characters with _
    illegal_chars = re.compile(r"[\\/:\"*?<>|]+")
    model_name = illegal_chars.sub("_", model_name)
    version = illegal_chars.sub("_", version)
    time_prefix = version_time.strftime("%y%m%d%H%M")
    return f"{model_name}/{time_prefix}-{version}"


def get_experiment_version():
    return time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

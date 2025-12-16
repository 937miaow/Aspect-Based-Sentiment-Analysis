import yaml
from typing import Any, Dict

class Config:
    """配置类，支持字典的点号访问"""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

def load_config(path: str = "config.yaml") -> Config:
    """加载 YAML 并返回 Config 对象"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return Config(config_dict)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 解析错误: {e}")
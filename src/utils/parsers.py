import re
import json
from typing import Dict, Any, Optional

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 的原始输出中提取 JSON 字符串。
    
    能够处理的情况：
    1. 标准 JSON
    2. 包裹在 Markdown 代码块中的 JSON (```json... ```)
    3. 混杂在解释性文字中的 JSON
    """
    try:
        # 尝试匹配 Markdown 代码块
        json_code_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_code_block:
            return json.loads(json_code_block.group(1))
        
        # 尝试匹配最外层的 {}
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
            
        return None
    except json.JSONDecodeError:
        return None

def extract_think_content(text: str) -> str:
    """
    提取 <think> 标签内的内容用于调试
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
import torch
import json
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from pydantic import ValidationError

from.schemas import ABSAOutput
from.prompts import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE
from.utils.parsers import extract_json_from_text, extract_think_content

class ABSA_Qwen_CoT_Pipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化管道：加载配置和模型
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        print(f"[Pipeline] 正在加载模型: {self.cfg['model']['model_id']}...")
        
        # 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg['model']['model_id'],
            trust_remote_code=True
        )
        
        # 动态计算数据类型
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        # 加载模型 (支持 4bit)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model']['model_id'],
            torch_dtype=torch_dtype,
            device_map=self.cfg['model']['device_map'],
            load_in_4bit=self.cfg['model']['use_4bit'],
            trust_remote_code=True
        )
        self.model.eval()
        print("[Pipeline] 模型加载完成。")

    def analyze(self, text: str) -> dict:
        """
        对单条文本进行分析
        """
        # 1. 构建 Prompt
        # 获取 Pydantic 的 Schema JSON
        schema_json = json.dumps(ABSAOutput.model_json_schema(), indent=2)
        
        system_msg = SYSTEM_PROMPT_TEMPLATE.format(schema=schema_json)
        user_msg = USER_PROMPT_TEMPLATE.format(text_input=text)
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # 2. 应用 Chat Template 并开启 Thinking
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # 关键：激活 CoT
        )
        
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        
        # 3. 推理生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg['model']['max_new_tokens'],
                temperature=self.cfg['generation']['temperature'],
                top_p=self.cfg['generation']['top_p'],
                top_k=self.cfg['generation']['top_k'],
                do_sample=self.cfg['generation']['do_sample']
            )
            
        # 4. 解码 (只取生成的 tokens)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 5. 解析结果
        think_content = extract_think_content(response_text)
        json_data = extract_json_from_text(response_text)
        
        result = {
            "original_text": text,
            "raw_response": response_text,
            "think_content": think_content,
            "parsed_output": None,
            "error": None
        }
        
        if json_data:
            try:
                # 使用 Pydantic 验证
                validated_obj = ABSAOutput.model_validate(json_data)
                result["parsed_output"] = validated_obj
            except ValidationError as e:
                result["error"] = f"Schema Validation Error: {e}"
        else:
            result["error"] = "JSON Extraction Failed"
            
        return result
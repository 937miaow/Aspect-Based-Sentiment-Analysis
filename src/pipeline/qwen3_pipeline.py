import torch
import re
from unsloth import FastLanguageModel
from transformers import TextStreamer
from..templates import SYSTEM_PROMPT, USER_PROMPT

class Qwen3CoTPipeline:
    def __init__(self, model_id: str, load_in_4bit: bool = True, max_seq_length: int = 2048):
        print(f"Loading Qwen3 Model: {model_id}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,  # 自动检测 (Float16/Bfloat16)
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        self.eos_token = self.tokenizer.eos_token

    def predict(self, text: str, aspect: str, stream: bool = False) -> dict:
        """
        执行推理并解析输出
        """
        # 1. 格式化 Prompt
        # Qwen3 推荐使用 Chat 模板
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(text=text, aspect=aspect)},
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # 2. 生成 (Qwen3 Thinking 模式推荐参数: Temp 0.6, TopP 0.95)
        # 这里的 enable_thinking=True 是 Qwen3 的特定行为，Unsloth 会保留
        streamer = TextStreamer(self.tokenizer) if stream else None
        
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=1024,
            temperature=0.6, 
            top_p=0.95,
            min_p=0.0,
            use_cache=True,
            streamer=streamer
        )
        
        decoded_output = self.tokenizer.decode(outputs, skip_special_tokens=False)
        
        # 3. 解析结果
        return self._parse_output(decoded_output)

    def _parse_output(self, raw_output: str) -> dict:
        """
        分离 <think> 内容和最终 JSON
        """
        # 提取 assistant 的回复部分
        if "<|im_start|>assistant" in raw_output:
            response = raw_output.split("<|im_start|>assistant")[-1].replace(self.eos_token, "").strip()
        else:
            response = raw_output

        # 提取思维链
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        rationale = think_match.group(1).strip() if think_match else ""
        
        # 提取最终结论 (通常在 </think> 之后)
        final_answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        return {
            "full_response": response,
            "rationale": rationale,  # CoT 部分
            "prediction": final_answer # 期望是 JSON 或 Label
        }
import torch
import re
from unsloth import FastLanguageModel
from transformers import TextStreamer
from..templates import SYSTEM_PROMPT, USER_PROMPT

class Qwen3CoTPipeline:         
    def __init__(self, model, tokenizer):
        """
        初始化 Pipeline
        :param model: 显存中已加载的 Unsloth 模型对象
        :param tokenizer: 配套的 Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # 显式开启推理模式
        FastLanguageModel.for_inference(self.model)

    def predict(self, text: str, aspect: str, stream: bool = False) -> dict:
        """
        执行推理并解析输出
        """
        # 1. 格式化 Prompt
        # Qwen3 推荐使用 Chat 模板
        formatted_user_content = USER_PROMPT.format(text=text, aspect=aspect)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_content},
        ]
        
        # 2. Tokenize & Template application
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # 3. 生成配置
        # stream=True 时，会在控制台看到思考过程逐字蹦出来
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream else None

        # 使用 no_grad 节省显存
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=512,  # 给 CoT 留足空间
                temperature=0.6,     # Qwen 推荐 Thinking 模式带一点温度
                top_p=0.95,
                use_cache=True,
                streamer=streamer,   
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 4. 解码 (只取新生成的部分)
        generated_ids = outputs[0][inputs.shape[-1]:]
        decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 5. 解析
        parsed_result = self._parse_output(decoded_output)

        # 6. 返回结果时，把输入信息合并进去
        # 这样结果字典就是完整的：{text, aspect, sentiment, rationale, full_response}
        return {
            "input_text": text,      # 原文
            "input_aspect": aspect,  # 目标方面
            **parsed_result          # 解包解析后的结果 (rationale, sentiment, full_response)
        }

    def _parse_output(self, response: str) -> dict:
        """
        将模型的文本输出解析为字典 (JSON-like)
        """
        rationale = ""
        prediction = "Unknown"

        # 1. 提取 <think> 内容
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            rationale = think_match.group(1).strip()

        # 2. 提取 Final Sentiment
        # 你的模型被训练为输出 "Final Sentiment: Positive"
        if "Final Sentiment:" in response:
            prediction = response.split("Final Sentiment:")[-1].strip()
        else:
            # 兜底：如果模型没按格式输出，尝试去除 <think> 后剩下的就是结论
            prediction = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        return {
            "full_response": response,   # 完整原始答案
            "rationale": rationale,      # 思考过程
            "sentiment": prediction      # 最终情感 (Positive/Negative/...)
        }
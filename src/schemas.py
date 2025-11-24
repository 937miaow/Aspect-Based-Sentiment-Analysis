from pydantic import BaseModel, Field
from typing import List, Literal

# 定义情感极性的字面量类型，限制模型只能输出这三种
SentimentType = Literal["positive", "negative", "neutral"]

class AspectAnalysis(BaseModel):
    """
    单个方面的分析结果结构
    """
    aspect_term: str = Field(
       ..., 
        description="从评论中提取的具体方面名词 (例如: '食物', '服务', '屏幕')"
    )
    sentiment: SentimentType = Field(
       ..., 
        description="该方面的情感极性"
    )
    quote: str = Field(
       ..., 
        description="原文中支持该情感判断的引用片段"
    )

class ABSAOutput(BaseModel):
    """
    整条评论的分析输出，包含多个方面
    """
    aspects: List[AspectAnalysis] = Field(
        default_factory=list,
        description="包含所有识别出的方面分析结果的列表"
    )

    def to_tuple_set(self):
        """
        辅助函数：将结果转换为 (term, sentiment) 的集合，方便评测计算 F1
        """
        return {(item.aspect_term.lower(), item.sentiment) for item in self.aspects}
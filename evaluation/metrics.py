from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np

def map_sentiment_to_int(sentiment_str: str):
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    # 简单的清洗逻辑，防止模型输出 extra spaces
    clean_str = str(sentiment_str).lower().strip().replace('"', '').replace("'", "")
    return mapping.get(clean_str, -1) # -1 表示解析失败

def calculate_absa_metrics(y_true, y_pred):
    """
    y_true: List[str] (e.g., ['positive', 'negative'])
    y_pred: List[str]
    """
    # 过滤掉无法解析的预测 (-1)
    y_true_int = []
    y_pred_int = []
    
    parse_errors = 0
    
    for t, p in zip(y_true, y_pred):
        t_val = map_sentiment_to_int(t)
        p_val = map_sentiment_to_int(p)
        
        if p_val == -1:
            parse_errors += 1
            # 策略：解析失败视为错误，填一个不存在的类别或默认类别
            # 这里为了严格惩罚，我们填入一个错误的 int
            p_val = 999 
            
        y_true_int.append(t_val)
        y_pred_int.append(p_val)

    # SemEval 官方指标: Macro-F1
    # labels= 确保只计算这三类的 F1，忽略解析错误类(999)
    # 过滤掉无法解析的样本（真实标签为 -1 或 预测解析为 999）
    valid_indices = [i for i, (t, p) in enumerate(zip(y_true_int, y_pred_int)) if t != -1 and p != 999]
    if valid_indices:
        y_true_filtered = [y_true_int[i] for i in valid_indices]
        y_pred_filtered = [y_pred_int[i] for i in valid_indices]
        macro_f1 = f1_score(y_true_filtered, y_pred_filtered, labels=[0, 1, 2], average='macro', zero_division=0)
        acc = accuracy_score(y_true_filtered, y_pred_filtered)
    else:
        macro_f1 = 0.0
        acc = 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "parse_error_rate": parse_errors / len(y_true)
    }
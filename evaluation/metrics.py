from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def map_sentiment_to_int(sentiment_str: str):
    # 修正点 1: 加入 conflict 类别映射
    mapping = {
        "negative": 0, 
        "neutral": 1, 
        "positive": 2, 
        "conflict": 3
    }
    
    # 简单的清洗逻辑，防止模型输出 extra spaces 或标点
    clean_str = str(sentiment_str).lower().strip().replace('"', '').replace("'", "").rstrip(".")
    return mapping.get(clean_str, -1) # -1 表示解析失败

def calculate_absa_metrics(y_true, y_pred):
    """
    y_true: List[str] (e.g., ['positive', 'conflict', ...])
    y_pred: List[str]
    """
    y_true_int = []
    y_pred_int = []
    
    parse_errors = 0
    
    for t, p in zip(y_true, y_pred):
        t_val = map_sentiment_to_int(t)
        p_val = map_sentiment_to_int(p)
        
        # 处理预测解析错误
        if p_val == -1:
            parse_errors += 1
            p_val = 999  # 标记为由格式错误导致的未知类别
            
        # 处理真实标签异常 (极为罕见，但也需要过滤)
        if t_val == -1:
            continue 
            
        y_true_int.append(t_val)
        y_pred_int.append(p_val)

    # ==================================================
    # 计算指标
    # ==================================================
    
    if len(y_true_int) > 0:
        # labels 加入 3 (Conflict)
        # 注意：我们不把 999 放进 labels，这样预测为 999 的样本会被视为错误（Mismatch），
        # 从而正确地拉低 Accuracy 和 Recall，符合严格评测标准。
        target_labels = [0, 1, 2, 3]
        
        macro_f1 = f1_score(y_true_int, y_pred_int, labels=target_labels, average='macro', zero_division=0)
        acc = accuracy_score(y_true_int, y_pred_int)
    else:
        macro_f1 = 0.0
        acc = 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "parse_error_rate": parse_errors / len(y_true) if len(y_true) > 0 else 0
    }

def calculate_absa_metrics_without_conflict(y_true, y_pred):
    """
    计算 3分类 (Positive, Neutral, Negative) 指标。
    策略：
    1. 如果真实标签(Truth)是 'conflict'，直接从评测中移除该样本（不参与分母计算）。
    2. 如果真实标签是 Pos/Neg/Neu，但预测成了 'conflict'，算作错误（Mismatch）。
    """
    y_true_int = []
    y_pred_int = []
    
    parse_errors = 0
    skipped_conflicts = 0 # 记录跳过了多少个 conflict
    
    for t, p in zip(y_true, y_pred):
        t_val = map_sentiment_to_int(t)
        p_val = map_sentiment_to_int(p)
        
        # 1. 核心过滤逻辑：如果真实标签是 Conflict (3)，直接跳过
        if t_val == 3:
            skipped_conflicts += 1
            continue
            
        # 2. 处理预测解析错误
        if p_val == -1:
            parse_errors += 1
            p_val = 999 
            
        # 3. 处理真实标签异常
        if t_val == -1:
            continue
            
        y_true_int.append(t_val)
        y_pred_int.append(p_val)

    # ==================================================
    # 计算指标 (只关注 0, 1, 2)
    # ==================================================
    if len(y_true_int) > 0:
        # labels=[0, 1, 2] 
        # 此时如果 p_val 是 3 (模型预测了 conflict)，会被视为错误
        target_labels = [0, 1, 2]
        
        macro_f1 = f1_score(y_true_int, y_pred_int, labels=target_labels, average='macro', zero_division=0)
        acc = accuracy_score(y_true_int, y_pred_int)
    else:
        macro_f1 = 0.0
        acc = 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "parse_error_rate": parse_errors / len(y_true_int) if len(y_true_int) > 0 else 0,
        "skipped_conflicts": skipped_conflicts
    }
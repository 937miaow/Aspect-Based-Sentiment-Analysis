from typing import List, Tuple, Set

def calculate_f1_scores(gold_data: List[List[dict]], pred_data: List[List[dict]]):
    """
    计算基于 (Aspect Term, Sentiment) 精确匹配的 F1 分数。
    
    Args:
        gold_data: 黄金标准列表，每个元素是字典列表 [{'aspect_term': 'foo', 'sentiment': 'pos'},...]
        pred_data: 预测结果列表，结构同上
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    for gold_list, pred_list in zip(gold_data, pred_data):
        # 转换为集合进行比较: Set of (term, sentiment)
        # 注意：转换为小写以进行更宽松的匹配
        gold_set = set()
        if gold_list:
            for item in gold_list:
                gold_set.add((item['aspect_term'].lower(), item['sentiment']))
        
        pred_set = set()
        if pred_list:
            for item in pred_list:
                pred_set.add((item['aspect_term'].lower(), item['sentiment']))
        
        # 计算集合交集
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "details": {"tp": tp, "fp": fp, "fn": fn}
    }
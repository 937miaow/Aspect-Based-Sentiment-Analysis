import sys
import os
import json
import pandas as pd
from tqdm import tqdm

# 将项目根目录添加到路径，以便导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import ABSA_Qwen_CoT_Pipeline
from src.utils.dataloader import load_semeval_xml
from evaluation.compute_metrics import calculate_f1_scores

def main():
    # 1. 加载数据
    test_file = "data/raw/semeval2014_task4/Restaurants_Test_Gold.xml"
    print(f"正在加载测试数据: {test_file}")
    try:
        df_test = load_semeval_xml(test_file)
        # 仅用于快速调试，正式跑分时请注释掉下面这行
        # df_test = df_test.head(10) 
    except Exception as e:
        print(f"错误: {e}")
        return

    # 2. 初始化 Pipeline
    pipeline = ABSA_Qwen_CoT_Pipeline(config_path="config/config.yaml")
    
    predictions = []
    raw_results = []
    
    # 3. 批量推理
    print("开始推理...")
    for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
        text = row['text']
        result = pipeline.analyze(text)
        
        # 提取预测的方面列表
        pred_aspects = []
        if result['parsed_output']:
            # 将 Pydantic 对象转回字典列表
            pred_aspects = [
                {'aspect_term': a.aspect_term, 'sentiment': a.sentiment} 
                for a in result['parsed_output'].aspects
            ]
        
        predictions.append(pred_aspects)
        
        # 保存原始结果用于调试
        raw_results.append({
            "text": text,
            "gold": row['gold_aspects'],
            "pred": pred_aspects,
            "think": result['think_content'],
            "error": result['error']
        })

    # 4. 保存中间结果
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/predictions_debug.json", "w", encoding='utf-8') as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
        
    # 5. 计算指标
    gold_list = df_test['gold_aspects'].tolist()
    metrics = calculate_f1_scores(gold_list, predictions)
    
    print("\n" + "="*30)
    print("Benchmark 结果")
    print("="*30)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
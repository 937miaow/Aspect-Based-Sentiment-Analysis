# Aspect-Based-Sentiment-Analysis

## REQUIREMENT

```shell
conda create -n ABSA python=3.10 -y

pip3 install torch torchvision torchaudio # pytorch.org

pip3 install transformers datasets numpy jupyter scikit-learn tqdm trl unsloth
```

## Stru

```
ABSA-Qwen3-CoT/ 
├── config.yaml # 配置文件 
├── data/ 
│ ├── raw/ 
│ │ └── SemEval2014Task4/ 
│ │   ├── Laptops_Train.xml 
│ │   ├── Restaurants_Train.xml 
│ │   └── SemEval2014Task.py # HF Dataset Builder Script 
│ └── processed/ # 存放清洗后及蒸馏生成的数据 
├── src/ 
│ ├── init.py 
│ ├── templates.py # Prompt 模板 
│ ├── pipelines/ 
│ │ ├── init.py 
│ │ └── qwen3_pipeline.py # Qwen3 推理流水线 (含CoT解析) 
│ └── utils/ 
│   ├── init.py 
│   └── config_loader.py # YAML 配置解析器 
├── evaluation/ 
│ ├── init.py 
│ └── metrics.py # SemEval 官方指标计算 (Macro-F1) 
├── notebooks/ 
│ ├── 01_Data_Preparation.ipynb # 数据加载、清洗与格式化
│ ├── 02_Teacher_Distillation_Gen.ipynb # 步骤A: 使用 gpt-oss 生成 CoT 数据
│ ├── 03_Student_FineTuning.ipynb # 步骤B: 使用生成数据微调 Qwen3
│ └── 04_Inference_and_Eval.ipynb # 模型对比评估
└── requirements.txt
```
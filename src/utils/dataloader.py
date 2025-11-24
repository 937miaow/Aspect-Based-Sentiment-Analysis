import xml.etree.ElementTree as ET
import pandas as pd
import os

def load_semeval_xml(file_path: str) -> pd.DataFrame:
    """
    解析 SemEval 2014 Task 4 的 XML 文件。
    返回 DataFrame，包含 text 和 gold_aspects (list of dicts)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")

    tree = ET.parse(file_path)
    root = tree.getroot()
    
    data = []
    
    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        aspect_terms = sentence.find('aspectTerms')
        
        aspects_list = []
        if aspect_terms is not None:
            for aspect in aspect_terms.findall('aspectTerm'):
                term = aspect.get('term')
                polarity = aspect.get('polarity')
                
                # 过滤掉 conflict 类别 (通常做法)
                if polarity!= 'conflict':
                    aspects_list.append({
                        'aspect_term': term,
                        'sentiment': polarity
                    })
        
        data.append({
            'text': text,
            'gold_aspects': aspects_list
        })
        
    return pd.DataFrame(data)
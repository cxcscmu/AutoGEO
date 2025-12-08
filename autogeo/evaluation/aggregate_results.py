import os
import json
from typing import List, Dict, Any

def mean_dict_list(dict_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate mean values for each key across a list of dictionaries.
    
    Args:
        dict_list: List of dictionaries with numeric values
        
    Returns:
        Dictionary with mean values for each key
    """
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    result = {}
    for k in keys:
        vals = [d[k] for d in dict_list if k in d and d[k] is not None and type(d[k]) in (int, float)]
        if vals:
            result[k] = sum(vals) / len(vals)
    return result

def aggregate_json_files(json_folder: str, dic_name: str) -> Dict[str, Dict[str, float]]:
    """Aggregate evaluation results from multiple JSON files.
    
    Args:
        json_folder: Directory containing JSON files
        dic_name: Dictionary key name to aggregate (e.g., "autogeo_researchy_geo_gemini_geo_score")
        
    Returns:
        Dictionary with aggregated mean scores
    """
    all_item = []

    for filename in os.listdir(json_folder):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(json_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key, item in data.items():
            if dic_name in item:
                text_name = dic_name.replace("_dict", "_text") 
                if text_name in item and item[text_name] == "":
                    if "ori_dict" in item:
                        all_item.append(item["ori_dict"])
                    if "ori_object_dict" in item:
                        all_item.append(item["ori_object_dict"])
                else:
                    all_item.append(item[dic_name])
     
    mean_item = mean_dict_list(all_item)

    return {dic_name: mean_item}

if __name__ == "__main__":
    import pprint

    json_folder = "data/Researchy-GEO/test"
    result1 = aggregate_json_files(json_folder, "autogeo_researchy_geo_gemini_geo_score")
    result2 = aggregate_json_files(json_folder, "autogeo_researchy_geo_gemini_geu_score")
    pprint.pprint(result1)
    pprint.pprint(result2)

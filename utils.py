import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_data_source_target(file_name, return_num=False, return_json=False):
    """
    file_name: a .json file containing a list of items, each has 'input_text', 'target_text', as keys
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if return_json:
        if return_num:
            return data, len(data)
        return data

    keys = list(data[0].keys())
    source_target_pair = []
    for item in data:
        source_target_pair.append([item[key] for key in keys])

    if return_num:
        return pd.DataFrame(source_target_pair, columns=keys), len(data)
    return pd.DataFrame(source_target_pair, columns=keys)

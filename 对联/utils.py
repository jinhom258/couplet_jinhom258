import os
import torch
import pandas as pd
from pypinyin import lazy_pinyin, Style

def get_pingze(char: str) -> int:
    try:
        pinyin_with_tone = lazy_pinyin(char, style=Style.TONE3)[0]
        tone = int(pinyin_with_tone[-1])
        return 0 if tone in (1, 2) else 1
    except (IndexError, ValueError):
        return 0

def generate_square_subsequent_mask(sz: int, device) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(device)

def load_couplet_data(dataset_root: str, split: str) -> pd.DataFrame:
    in_path = os.path.join(dataset_root, split, 'in.txt')
    out_path = os.path.join(dataset_root, split, 'out.txt')
    
    with open(in_path, 'r', encoding='utf-8') as f:
        uppers = [line.strip() for line in f if line.strip()]
    with open(out_path, 'r', encoding='utf-8') as f:
        lowers = [line.strip() for line in f if line.strip()]
    
    assert len(uppers) == len(lowers), f"{split}集上下联行数不匹配！"
    
    data = []
    for u, l in zip(uppers, lowers):
        if len(u) == len(l):
            data.append({'upper': u, 'lower': l})
    
    return pd.DataFrame(data)

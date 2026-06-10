import torch
from torch.utils.data import Dataset

class CoupletDataset(Dataset):
    def __init__(self, df, vocab, max_seq_len):
        self.df = df
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        upper = self.df.iloc[idx]['upper']
        lower = self.df.iloc[idx]['lower']

        def encode_seq(seq: str) -> list[int]:
            return [self.vocab.word2idx[self.vocab.SOS]] + \
                   [self.vocab.word2idx.get(c, self.vocab.word2idx[self.vocab.UNK]) for c in seq] + \
                   [self.vocab.word2idx[self.vocab.EOS]]

        upper_ids = encode_seq(upper)
        lower_ids = encode_seq(lower)

        def encode_pingze(seq: str) -> list[int]:
            from utils import get_pingze
            return [0] + [get_pingze(c) for c in seq] + [0]

        upper_pz = encode_pingze(upper)
        lower_pz = encode_pingze(lower)

        def pad_seq(seq: list[int], pad_val: int) -> list[int]:
            if len(seq) > self.max_seq_len:
                return seq[:self.max_seq_len]
            return seq + [pad_val] * (self.max_seq_len - len(seq))

        upper_ids = pad_seq(upper_ids, self.vocab.word2idx[self.vocab.PAD])
        lower_ids = pad_seq(lower_ids, self.vocab.word2idx[self.vocab.PAD])
        upper_pz = pad_seq(upper_pz, 0)
        lower_pz = pad_seq(lower_pz, 0)

        return {
            'upper_ids': torch.tensor(upper_ids, dtype=torch.long),
            'lower_ids': torch.tensor(lower_ids, dtype=torch.long),
            'upper_pz': torch.tensor(upper_pz, dtype=torch.long),
            'lower_pz': torch.tensor(lower_pz, dtype=torch.long)
        }

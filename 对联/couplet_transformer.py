import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style
import os

# ===================== 1. 全局配置（路径已适配你的目录） =====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_ROOT = './couplet'  # 你的数据集根目录
MAX_SEQ_LEN = 16
BATCH_SIZE = 64
EPOCHS = 25
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LEARNING_RATE = 1e-4
LAMBDA_PZ = 0.35
BEAM_SIZE = 3

# ===================== 2. 工具函数（数据加载已修改） =====================
def get_pingze(char: str) -> int:
    try:
        pinyin_with_tone = lazy_pinyin(char, style=Style.TONE3)[0]
        tone = int(pinyin_with_tone[-1])
        return 0 if tone in (1, 2) else 1
    except (IndexError, ValueError):
        return 0

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(DEVICE)

# ✅ 新：适配分离式文件的数据加载函数
def load_couplet_data(split: str) -> pd.DataFrame:
    """
    加载分离式对联数据集
    split: 'train' 或 'test'
    返回：DataFrame，包含'upper'和'lower'两列
    """
    in_path = os.path.join(DATASET_ROOT, split, 'in.txt')
    out_path = os.path.join(DATASET_ROOT, split, 'out.txt')
    
    # 读取上联和下联
    with open(in_path, 'r', encoding='utf-8') as f:
        uppers = [line.strip() for line in f if line.strip()]
    with open(out_path, 'r', encoding='utf-8') as f:
        lowers = [line.strip() for line in f if line.strip()]
    
    # 验证行数一致
    assert len(uppers) == len(lowers), f"{split}集上下联行数不匹配！"
    
    # 过滤上下联长度不一致的样本
    data = []
    for u, l in zip(uppers, lowers):
        if len(u) == len(l):
            data.append({'upper': u, 'lower': l})
    
    return pd.DataFrame(data)

# ===================== 3. 词汇表类（新增加载已有词汇表功能） =====================
class Vocab:
    def __init__(self):
        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'
        self.word2idx = {self.PAD:0, self.SOS:1, self.EOS:2, self.UNK:3}
        self.idx2word = {0:self.PAD, 1:self.SOS, 2:self.EOS, 3:self.UNK}
        self.word_count = {}

    def add_word(self, word: str):
        if word not in self.word_count:
            self.word_count[word] = 0
        self.word_count[word] += 1

    def build_vocab(self, sentences: list[str], min_freq: int = 2):
        for sent in sentences:
            for char in sent:
                self.add_word(char)
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words:
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word

    # ✅ 新：加载已有vocabs文件
    def load_from_file(self, vocab_path: str):
        """从vocabs文件加载词汇表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[len(self.idx2word)] = word
        print(f"从{vocab_path}加载词汇表，大小：{len(self)}")

    # ✅ 新：保存词汇表到文件
    def save_to_file(self, vocab_path: str):
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word in self.word2idx.keys():
                if word not in [self.PAD, self.SOS, self.EOS, self.UNK]:
                    f.write(word + '\n')

    def __len__(self):
        return len(self.word2idx)

# ===================== 4. 自定义数据集类（无需修改） =====================
class CoupletDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab):
        self.df = df
        self.vocab = vocab

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
            return [0] + [get_pingze(c) for c in seq] + [0]

        upper_pz = encode_pingze(upper)
        lower_pz = encode_pingze(lower)

        def pad_seq(seq: list[int], pad_val: int) -> list[int]:
            if len(seq) > MAX_SEQ_LEN:
                return seq[:MAX_SEQ_LEN]
            return seq + [pad_val] * (MAX_SEQ_LEN - len(seq))

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

# ===================== 5. Transformer核心模块（无需修改） =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerCouplet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.d_model = D_MODEL
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.pos_encoder = PositionalEncoding(D_MODEL, dropout=DROPOUT)
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True
        )
        self.fc_out = nn.Linear(D_MODEL, vocab_size)
        self.fc_pingze = nn.Linear(D_MODEL, 2)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
                src_padding_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None):
        src_emb = self.embedding(src) * np.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        transformer_out = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        word_logits = self.fc_out(transformer_out)
        pingze_logits = self.fc_pingze(transformer_out)

        return word_logits, pingze_logits

# ===================== 6. 训练与验证模块（无需修改） =====================
def calculate_loss(word_logits: torch.Tensor, pingze_logits: torch.Tensor,
                   tgt_word: torch.Tensor, tgt_pingze: torch.Tensor,
                   vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    criterion_word = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.PAD])
    loss_word = criterion_word(word_logits.reshape(-1, word_logits.size(-1)), tgt_word.reshape(-1))

    mask = (tgt_word != vocab.word2idx[vocab.PAD]).float()
    criterion_pingze = nn.CrossEntropyLoss(reduction='none')
    loss_pingze = criterion_pingze(pingze_logits.reshape(-1, 2), tgt_pingze.reshape(-1))
    loss_pingze = (loss_pingze * mask.reshape(-1)).sum() / mask.sum()

    total_loss = loss_word + LAMBDA_PZ * loss_pingze
    return total_loss, loss_word, loss_pingze

def train_epoch(model: TransformerCouplet, loader: DataLoader,
                optimizer: torch.optim.Optimizer, vocab: Vocab) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Training'):
        upper_ids = batch['upper_ids'].to(DEVICE)
        lower_ids = batch['lower_ids'].to(DEVICE)
        lower_pz = batch['lower_pz'].to(DEVICE)

        tgt_input = lower_ids[:, :-1]
        tgt_output_word = lower_ids[:, 1:]
        tgt_output_pz = lower_pz[:, 1:]

        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))
        src_padding_mask = (upper_ids == vocab.word2idx[vocab.PAD]).to(DEVICE)
        tgt_padding_mask = (tgt_input == vocab.word2idx[vocab.PAD]).to(DEVICE)

        optimizer.zero_grad()
        word_logits, pingze_logits = model(
            upper_ids, tgt_input,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )

        loss, _, _ = calculate_loss(word_logits, pingze_logits, tgt_output_word, tgt_output_pz, vocab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def val_epoch(model: TransformerCouplet, loader: DataLoader, vocab: Vocab) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            upper_ids = batch['upper_ids'].to(DEVICE)
            lower_ids = batch['lower_ids'].to(DEVICE)
            lower_pz = batch['lower_pz'].to(DEVICE)

            tgt_input = lower_ids[:, :-1]
            tgt_output_word = lower_ids[:, 1:]
            tgt_output_pz = lower_pz[:, 1:]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))
            src_padding_mask = (upper_ids == vocab.word2idx[vocab.PAD]).to(DEVICE)
            tgt_padding_mask = (tgt_input == vocab.word2idx[vocab.PAD]).to(DEVICE)

            word_logits, pingze_logits = model(
                upper_ids, tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )

            loss, _, _ = calculate_loss(word_logits, pingze_logits, tgt_output_word, tgt_output_pz, vocab)
            total_loss += loss.item()

    return total_loss / len(loader)

# ===================== 7. 生成推理模块（新增批量测试函数） =====================
def generate_couplet(model: TransformerCouplet, upper_sent: str, vocab: Vocab) -> str:
    model.eval()
    upper_ids = [vocab.word2idx[vocab.SOS]] + \
                [vocab.word2idx.get(c, vocab.word2idx[vocab.UNK]) for c in upper_sent] + \
                [vocab.word2idx[vocab.EOS]]
    upper_ids = torch.tensor(upper_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    upper_pz_seq = [0] + [get_pingze(c) for c in upper_sent] + [0]
    target_len = len(upper_sent)

    beams = [(torch.tensor([vocab.word2idx[vocab.SOS]], dtype=torch.long).to(DEVICE), 0.0)]

    for step in range(target_len):
        new_beams = []
        for seq, score in beams:
            tgt_mask = generate_square_subsequent_mask(seq.size(0))
            with torch.no_grad():
                word_logits, _ = model(upper_ids, seq.unsqueeze(0), tgt_mask=tgt_mask)
            last_token_logits = word_logits[0, -1, :]
            last_token_probs = F.log_softmax(last_token_logits, dim=-1)

            topk_probs, topk_indices = torch.topk(last_token_probs, k=BEAM_SIZE * 3)

            for prob, idx in zip(topk_probs, topk_indices):
                char = vocab.idx2word[idx.item()]
                if char in [vocab.SOS, vocab.EOS, vocab.PAD, vocab.UNK]:
                    continue

                expected_pz = 1 - upper_pz_seq[step + 1]
                if step == target_len - 1:
                    expected_pz = 0

                if get_pingze(char) == expected_pz:
                    new_seq = torch.cat([seq, idx.unsqueeze(0)])
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:BEAM_SIZE]

        if not beams:
            beams = [(torch.cat([seq, topk_indices[0].unsqueeze(0)]), score + topk_probs[0].item())]

    best_seq = beams[0][0][1:]
    return ''.join([vocab.idx2word[idx.item()] for idx in best_seq])

# ✅ 新：批量测试整个测试集并保存结果
def batch_test(model: TransformerCouplet, vocab: Vocab, output_path: str = 'test_result.txt'):
    """批量生成测试集下联并保存到文件"""
    test_in_path = os.path.join(DATASET_ROOT, 'test', 'in.txt')
    with open(test_in_path, 'r', encoding='utf-8') as f:
        test_uppers = [line.strip() for line in f if line.strip()]
    
    print(f"开始批量测试{len(test_uppers)}个上联...")
    results = []
    for upper in tqdm(test_uppers):
        lower = generate_couplet(model, upper, vocab)
        results.append(f"{upper} | {lower}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print(f"测试结果已保存到{output_path}")

# ===================== 8. 主函数入口（已适配你的目录） =====================
if __name__ == '__main__':
    # 步骤1：加载数据集
    print('='*50)
    print('加载数据集...')
    train_df = load_couplet_data('train')
    test_df = load_couplet_data('test')
    print(f'训练集样本数：{len(train_df)}')
    print(f'测试集样本数：{len(test_df)}')

    # 步骤2：构建/加载词汇表
    print('\n构建词汇表...')
    vocab = Vocab()
    vocab_path = os.path.join(DATASET_ROOT, 'vocabs')
    
    if os.path.exists(vocab_path):
        # 优先加载已有词汇表
        vocab.load_from_file(vocab_path)
    else:
        # 自动构建词汇表并保存
        print('未找到vocabs文件，自动构建中...')
        all_sentences = pd.concat([train_df['upper'], train_df['lower']]).tolist()
        vocab.build_vocab(all_sentences, min_freq=2)
        vocab.save_to_file(vocab_path)
        print(f'词汇表已保存到{vocab_path}')
    
    print(f'词汇表总大小：{len(vocab)}')

    # 步骤3：构建DataLoader
    print('\n创建数据加载器...')
    train_dataset = CoupletDataset(train_df, vocab)
    test_dataset = CoupletDataset(test_df, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 步骤4：初始化模型
    print('\n初始化Transformer模型...')
    model = TransformerCouplet(vocab_size=len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    print(f'模型参数总量：{sum(p.numel() for p in model.parameters()):,}')

    # 步骤5：开始训练
    print('\n开始训练...')
    print('='*50)
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        train_loss = train_epoch(model, train_loader, optimizer, vocab)
        val_loss = val_epoch(model, test_loader, vocab)
        scheduler.step()

        print(f'训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_couplet.pth')
            print('✅ 保存最优模型！')

    # 步骤6：加载最优模型并测试
    print('\n' + '='*50)
    print('加载最优模型进行测试...')
    model.load_state_dict(torch.load('best_transformer_couplet.pth'))

    # 单例测试
    print('\n=== 单例对联生成测试 ===')
    test_cases = [
        '春回大地千山秀',
        '春风得意马蹄疾',
        '书山有路勤为径',
        '明月松间照',
        '海阔凭鱼跃'
    ]
    for upper in test_cases:
        lower = generate_couplet(model, upper, vocab)
        print(f'上联：{upper}')
        print(f'下联：{lower}\n')

    # 批量测试整个测试集
    batch_test(model, vocab)
    print('\n实验完成！')
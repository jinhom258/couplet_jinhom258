import os
import torch
from torch.utils.data import DataLoader
import pandas as pd

from config import *
from utils import load_couplet_data
from vocab import Vocab
from dataset import CoupletDataset
from model import TransformerCouplet
from trainer import train_epoch, val_epoch
from inference import generate_couplet, batch_test

def main():
    print('='*50)
    print(f'使用设备: {DEVICE}')
    print('加载数据集...')
    train_df = load_couplet_data(DATASET_ROOT, 'train')
    test_df = load_couplet_data(DATASET_ROOT, 'test')
    print(f'训练集样本数：{len(train_df)}')
    print(f'测试集样本数：{len(test_df)}')

    print('\n构建词汇表...')
    vocab = Vocab()
    vocab_path = os.path.join(DATASET_ROOT, 'vocabs')
    
    if os.path.exists(vocab_path):
        vocab.load_from_file(vocab_path)
    else:
        print('未找到vocabs文件，自动构建中...')
        all_sentences = pd.concat([train_df['upper'], train_df['lower']]).tolist()
        vocab.build_vocab(all_sentences, min_freq=2)
        vocab.save_to_file(vocab_path)
        print(f'词汇表已保存到{vocab_path}')
    
    print(f'词汇表总大小：{len(vocab)}')

    print('\n创建数据加载器...')
    train_dataset = CoupletDataset(train_df, vocab, MAX_SEQ_LEN)
    test_dataset = CoupletDataset(test_df, vocab, MAX_SEQ_LEN)
    
    # 使用多进程数据加载（Windows下num_workers设为0或2）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    print('\n初始化Transformer模型...')
    model = TransformerCouplet(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 混合精度训练scaler（仅GPU可用时）
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None
    if scaler:
        print('✅ 已启用混合精度训练')
    if GRADIENT_ACCUMULATION_STEPS > 1:
        print(f'✅ 已启用梯度累积，等效batch_size = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}')
    
    print(f'模型参数总量：{sum(p.numel() for p in model.parameters()):,}')

    print('\n开始训练...')
    print('='*50)
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print(f'当前学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, vocab, 
            LAMBDA_PZ, DEVICE,
            grad_accum_steps=GRADIENT_ACCUMULATION_STEPS,
            print_freq=PRINT_FREQ
        )
        scheduler.step()

        print(f'训练损失：{train_loss:.4f}')

        # 控制验证频率
        should_validate = VALIDATE_EVERY_EPOCH or ((epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0)
        if should_validate or epoch == EPOCHS - 1:
            val_loss = val_epoch(model, test_loader, vocab, LAMBDA_PZ, DEVICE, scaler)
            print(f'验证损失：{val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_transformer_couplet.pth')
                print('✅ 保存最优模型！')

    print('\n' + '='*50)
    print('加载最优模型进行测试...')
    model.load_state_dict(torch.load('best_transformer_couplet.pth', map_location=DEVICE, weights_only=True))

    print('\n=== 单例对联生成测试 ===')
    test_cases = [
        '春回大地千山秀',
        '春风得意马蹄疾',
        '书山有路勤为径',
        '明月松间照',
        '海阔凭鱼跃'
    ]
    for upper in test_cases:
        lower = generate_couplet(model, upper, vocab, BEAM_SIZE, DEVICE)
        print(f'上联：{upper}')
        print(f'下联：{lower}\n')

    batch_test(model, vocab, DATASET_ROOT, BEAM_SIZE, DEVICE)
    print('\n实验完成！')

if __name__ == '__main__':
    main()

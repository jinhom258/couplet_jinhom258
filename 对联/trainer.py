import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import generate_square_subsequent_mask

def calculate_loss(word_logits: torch.Tensor, pingze_logits: torch.Tensor,
                   tgt_word: torch.Tensor, tgt_pingze: torch.Tensor,
                   vocab, lambda_pz: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    criterion_word = torch.nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.PAD])
    loss_word = criterion_word(word_logits.reshape(-1, word_logits.size(-1)), tgt_word.reshape(-1))

    mask = (tgt_word != vocab.word2idx[vocab.PAD]).float()
    criterion_pingze = torch.nn.CrossEntropyLoss(reduction='none')
    loss_pingze = criterion_pingze(pingze_logits.reshape(-1, 2), tgt_pingze.reshape(-1))
    loss_pingze = (loss_pingze * mask.reshape(-1)).sum() / mask.sum()

    total_loss = loss_word + lambda_pz * loss_pingze
    return total_loss, loss_word, loss_pingze

def train_epoch(model, loader, optimizer, scaler, vocab, lambda_pz: float, device, 
                grad_accum_steps=1, print_freq=100, verbose=True) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    if verbose:
        pbar = tqdm(loader, desc='Training')
    else:
        pbar = loader
    
    for batch_idx, batch in enumerate(pbar):
        upper_ids = batch['upper_ids'].to(device, non_blocking=True)
        lower_ids = batch['lower_ids'].to(device, non_blocking=True)
        lower_pz = batch['lower_pz'].to(device, non_blocking=True)

        tgt_input = lower_ids[:, :-1]
        tgt_output_word = lower_ids[:, 1:]
        tgt_output_pz = lower_pz[:, 1:]

        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)
        src_padding_mask = (upper_ids == vocab.word2idx[vocab.PAD]).to(device, non_blocking=True)
        tgt_padding_mask = (tgt_input == vocab.word2idx[vocab.PAD]).to(device, non_blocking=True)

        # 混合精度前向传播
        if scaler is not None:
            with torch.cuda.amp.autocast():
                word_logits, pingze_logits = model(
                    upper_ids, tgt_input,
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask
                )
                loss, _, _ = calculate_loss(word_logits, pingze_logits, tgt_output_word, tgt_output_pz, vocab, lambda_pz)
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            word_logits, pingze_logits = model(
                upper_ids, tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask
            )
            loss, _, _ = calculate_loss(word_logits, pingze_logits, tgt_output_word, tgt_output_pz, vocab, lambda_pz)
            loss = loss / grad_accum_steps
            loss.backward()

        total_loss += loss.item() * grad_accum_steps

        # 梯度累积后更新
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        if verbose and (batch_idx + 1) % print_freq == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    return total_loss / len(loader)

def val_epoch(model, loader, vocab, lambda_pz: float, device, scaler=None, verbose=True) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        if verbose:
            pbar = tqdm(loader, desc='Validating')
        else:
            pbar = loader
        
        for batch in pbar:
            upper_ids = batch['upper_ids'].to(device, non_blocking=True)
            lower_ids = batch['lower_ids'].to(device, non_blocking=True)
            lower_pz = batch['lower_pz'].to(device, non_blocking=True)

            tgt_input = lower_ids[:, :-1]
            tgt_output_word = lower_ids[:, 1:]
            tgt_output_pz = lower_pz[:, 1:]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)
            src_padding_mask = (upper_ids == vocab.word2idx[vocab.PAD]).to(device, non_blocking=True)
            tgt_padding_mask = (tgt_input == vocab.word2idx[vocab.PAD]).to(device, non_blocking=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    word_logits, pingze_logits = model(
                        upper_ids, tgt_input,
                        tgt_mask=tgt_mask,
                        src_padding_mask=src_padding_mask,
                        tgt_padding_mask=tgt_padding_mask
                    )
            else:
                word_logits, pingze_logits = model(
                    upper_ids, tgt_input,
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask
                )

            loss, _, _ = calculate_loss(word_logits, pingze_logits, tgt_output_word, tgt_output_pz, vocab, lambda_pz)
            total_loss += loss.item()

    return total_loss / len(loader)

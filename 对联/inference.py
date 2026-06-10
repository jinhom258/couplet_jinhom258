import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_pingze, generate_square_subsequent_mask

def generate_couplet(model, upper_sent: str, vocab, beam_size: int, device) -> str:
    model.eval()
    upper_ids = [vocab.word2idx[vocab.SOS]] + \
                [vocab.word2idx.get(c, vocab.word2idx[vocab.UNK]) for c in upper_sent] + \
                [vocab.word2idx[vocab.EOS]]
    upper_ids = torch.tensor(upper_ids, dtype=torch.long).unsqueeze(0).to(device)

    upper_pz_seq = [0] + [get_pingze(c) for c in upper_sent] + [0]
    target_len = len(upper_sent)

    beams = [(torch.tensor([vocab.word2idx[vocab.SOS]], dtype=torch.long).to(device), 0.0)]

    for step in range(target_len):
        new_beams = []
        for seq, score in beams:
            tgt_mask = generate_square_subsequent_mask(seq.size(0), device)
            with torch.no_grad():
                word_logits, _ = model(upper_ids, seq.unsqueeze(0), tgt_mask=tgt_mask)
            last_token_logits = word_logits[0, -1, :]
            last_token_probs = F.log_softmax(last_token_logits, dim=-1)

            topk_probs, topk_indices = torch.topk(last_token_probs, k=beam_size * 3)

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
        beams = new_beams[:beam_size]

        if not beams:
            beams = [(torch.cat([seq, topk_indices[0].unsqueeze(0)]), score + topk_probs[0].item())]

    best_seq = beams[0][0][1:]
    return ''.join([vocab.idx2word[idx.item()] for idx in best_seq])

def batch_test(model, vocab, dataset_root: str, beam_size: int, device, output_path: str = 'test_result.txt'):
    test_in_path = os.path.join(dataset_root, 'test', 'in.txt')
    with open(test_in_path, 'r', encoding='utf-8') as f:
        test_uppers = [line.strip() for line in f if line.strip()]
    
    print(f"开始批量测试{len(test_uppers)}个上联...")
    results = []
    for upper in tqdm(test_uppers):
        lower = generate_couplet(model, upper, vocab, beam_size, device)
        results.append(f"{upper} | {lower}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print(f"测试结果已保存到{output_path}")

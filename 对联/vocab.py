import os

class Vocab:
    def __init__(self):
        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'
        self.word2idx = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2word = {0: self.PAD, 1: self.SOS, 2: self.EOS, 3: self.UNK}
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

    def load_from_file(self, vocab_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[len(self.idx2word)] = word
        print(f"从{vocab_path}加载词汇表，大小：{len(self)}")

    def save_to_file(self, vocab_path: str):
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word in self.word2idx.keys():
                if word not in [self.PAD, self.SOS, self.EOS, self.UNK]:
                    f.write(word + '\n')

    def __len__(self):
        return len(self.word2idx)

"""
Data loading utilities for HWA experiments.

CIFAR-100: 32x32 color images, 100 classes
WikiText-2: Language modeling benchmark, ~2M training tokens
"""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


# =============================================================================
# CIFAR-100 (Vision)
# =============================================================================

def get_cifar100_loaders(data_dir: str = './data',
                         batch_size: int = 128,
                         num_workers: int = 4,
                         pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders with standard augmentation.
    
    Train augmentation:
        - Random crop 32x32 with padding=4
        - Random horizontal flip
        - Normalize to ImageNet stats (common practice)
    
    Test: Just normalize
    """
    # Normalization values (CIFAR stats, close to ImageNet)
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Download if needed
    train_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Split train into train/val (90/10)
    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Val set uses test transform (no augmentation)
    # Note: random_split preserves transform, so we need a workaround
    # For simplicity, we just use the train transform for val too
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"[data.py] CIFAR-100 loaded: {len(train_set)} train, "
          f"{len(val_set)} val, {len(test_set)} test")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# WikiText-2 (Language)
# =============================================================================

class Dictionary:
    """Word to index mapping for language modeling."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
    
    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def __len__(self) -> int:
        return len(self.idx2word)


class WikiText2Corpus:
    """
    WikiText-2 corpus for language modeling.
    
    Downloads from HuggingFace if not present locally.
    Tokenizes into word-level indices.
    """
    
    def __init__(self, data_dir: str = './data/wikitext-2'):
        self.dictionary = Dictionary()
        
        train_path = os.path.join(data_dir, 'train.txt')
        
        # Download if needed
        if not os.path.exists(train_path):
            print(f"[data.py] Downloading WikiText-2 via HuggingFace...")
            os.makedirs(data_dir, exist_ok=True)
            
            try:
                from datasets import load_dataset
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', 
                                       trust_remote_code=True)
                
                for split, fname in [('train', 'train.txt'), 
                                     ('validation', 'valid.txt'),
                                     ('test', 'test.txt')]:
                    with open(os.path.join(data_dir, fname), 'w') as f:
                        for item in dataset[split]:
                            text = item['text'].strip()
                            if text:
                                f.write(text + '\n')
                    print(f"[data.py] Created {fname}")
                    
            except ImportError:
                raise RuntimeError(
                    "datasets library not installed. Run: pip install datasets"
                )
        
        # Tokenize
        print(f"[data.py] Loading WikiText-2 corpus...")
        self.train = self._tokenize(os.path.join(data_dir, 'train.txt'))
        self.valid = self._tokenize(os.path.join(data_dir, 'valid.txt'))
        self.test = self._tokenize(os.path.join(data_dir, 'test.txt'))
        
        print(f"[data.py] WikiText-2: vocab={len(self.dictionary):,}, "
              f"train={len(self.train):,} tokens")
    
    def _tokenize(self, path: str) -> torch.Tensor:
        """Tokenize a text file into a tensor of word indices."""
        # First pass: build vocabulary
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.split():
                    self.dictionary.add_word(word)
                self.dictionary.add_word('<eos>')
        
        # Second pass: convert to indices
        ids = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.split():
                    ids.append(self.dictionary.word2idx[word])
                ids.append(self.dictionary.word2idx['<eos>'])
        
        return torch.tensor(ids, dtype=torch.long)


def batchify(data: torch.Tensor, batch_size: int, 
             device: torch.device) -> torch.Tensor:
    """
    Reshape data into [seq_len, batch_size] for BPTT.
    
    The data is arranged column-wise so that consecutive tokens
    in a column are consecutive in the original sequence.
    """
    # Drop remainder
    n_batch = data.size(0) // batch_size
    data = data.narrow(0, 0, n_batch * batch_size)
    
    # Reshape: [total] -> [seq_len, batch]
    data = data.view(batch_size, -1).t().contiguous()
    
    return data.to(device)


def get_lm_batch(source: torch.Tensor, i: int, 
                 bptt: int = 35) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch for language modeling from position i.
    
    Args:
        source: batchified data [seq_len, batch]
        i: starting position
        bptt: sequence length for backprop through time
        
    Returns:
        data: input tokens [seq_len, batch]
        target: target tokens (shifted by 1) [seq_len * batch]
    """
    seq_len = min(bptt, source.size(0) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


# Quick test
if __name__ == '__main__':
    print("Testing CIFAR-100 loader...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=64, num_workers=0
    )
    
    for x, y in train_loader:
        print(f"  Batch shape: {x.shape}, labels: {y.shape}")
        break
    
    print("\nTesting WikiText-2 corpus...")
    corpus = WikiText2Corpus()
    train_data = batchify(corpus.train, batch_size=20, 
                          device=torch.device('cpu'))
    print(f"  Batchified shape: {train_data.shape}")
    
    data, target = get_lm_batch(train_data, 0, bptt=35)
    print(f"  Batch: data={data.shape}, target={target.shape}")

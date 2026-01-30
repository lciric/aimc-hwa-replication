import torch.nn as nn
from src.layers import AnalogLSTM, AnalogLinear
class AnalogLSTMLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, physics_engine=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = AnalogLSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, physics_engine=physics_engine, dropout=0.5)
        self.decoder = AnalogLinear(in_features=hidden_dim, out_features=vocab_size, physics_engine=physics_engine)
        self.decoder.weight = self.embedding.weight
    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        output, hidden = self.lstm(emb, hidden)
        decoded = self.decoder(output)
        return decoded, hidden

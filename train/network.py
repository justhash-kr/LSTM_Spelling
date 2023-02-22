import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, hidden_size, num_layers, pad_idx):
        super().__init__()
        self.embed_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        self.lstm_layer = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.5
        )

        self.last_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size*2,out_features=hidden_size),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
            nn.Sigmoid()            
        )
    
    def forward(self, x):
        embed_x = self.embed_layer(x)
        output, (_, _) = self.lstm_layer(embed_x)
        output = output[:, -1, :]
        last_output = self.last_layer(output)
        return last_output
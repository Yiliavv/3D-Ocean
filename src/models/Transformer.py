from torch import manual_seed, zeros, sin, cos, exp, arange, log, tensor, meshgrid
from torch.nn import Module, Embedding, Transformer, Softmax, init


class SSTTransformer(Module):
    def __init__(self, shape):
        super(SSTTransformer, self).__init__()

        manual_seed(0)

        self.embedding = Embedding(num_embeddings=6400, embedding_dim=512)

        # Transformer编码器-解码器
        self.transformer = Transformer(d_model=512, nhead=8,
                                       num_encoder_layers=6,
                                       num_decoder_layers=6,
                                       dim_feedforward=2048,
                                       dropout=0.2, batch_first=True)

        self.position_encoding = PositionalEncoding(shape[0], shape[1], 512)

        self.fc = Softmax(dim=1)

        self.initialize()

    def forward(self, src, tgt):
        # Transformer
        embed_x = self.embedding(src)
        embed_y = self.embedding(tgt)

        print(embed_x.shape)
        print(embed_y.shape)

        embed_x = self.position_encoding(embed_x)
        embed_y = self.position_encoding(embed_y)

        print(embed_x.shape)
        print(embed_y.shape)

        out = self.transformer(embed_x, embed_y)

        return out[:, :, -1]

    def initialize(self):
        init.kaiming_normal_(self.embedding.weight)


class PositionalEncoding(Module):
    def __init__(self, width, height, d_model):
        super(PositionalEncoding, self).__init__()

        pe = zeros(width, height, d_model)

        position_x, position_y = meshgrid(arange(0, width), arange(0, height))
        position_x.unsqueeze(1)
        position_y.unsqueeze(1)

        div_term = exp(arange(0, d_model, 2).float() * (-log(tensor(10000.0) / d_model)))

        print(position_x.shape)
        print(position_y.shape)
        print(div_term.shape)

        pe[:, 0::2, :] = sin(position_x * div_term)
        pe[:, 1::2, :] = cos(position_x * div_term)
        pe[:, :, 0::2] = cos(position_y * div_term)
        pe[:, :, 1::2] = sin(position_y * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)


def forward(self, x):
    x = x + self.pe[:, :x.size(1)].requires_grad_(False)

    return x

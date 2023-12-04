import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            # [b, 784] => [b, 256]
            nn.Linear(784, 256),
            nn.ReLU(),
            # [b, 256] => [b, 64]
            nn.Linear(256, 64),
            nn.ReLU(),
            # [b, 64] => [b, 16]
            nn.Linear(64, 16),
            nn.ReLU()
        )

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.enc(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            # [b, 16] => [b, 64]
            nn.Linear(16, 64),
            nn.ReLU(),
            # [b, 64] => [b, 256]
            nn.Linear(64, 256),
            nn.ReLU(),
            # [b, 256] => [b, 784]
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dec(x)
        x = x.view(-1, 1, 28, 28)
        return x


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

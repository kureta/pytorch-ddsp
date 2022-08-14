import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from .constants import SAMPLE_RATE
import torchcrepe
from einops import rearrange


class CrepeLoss:
    def __init__(self):
        self.crepe = torchcrepe.Crepe('tiny').cuda()
        for param in self.crepe.parameters():
            param.requires_grad = False

    def get_embedding(self, example):
        example = rearrange(example, "b c t -> (b c) t")
        example = AF.resample(example, SAMPLE_RATE, 16000)
        example = torch.nn.functional.pad(example, (1024 // 2, 1024 // 2))
        example = example.unfold(1, 1024, 256)
        example = rearrange(example, "b c t -> (b c) t")

        example = example - example.mean(dim=1, keepdim=True)
        example = example / torch.max(torch.tensor(1e-10, device=example.device),
                                      example.std(dim=1, keepdim=True))

        embedded = self.crepe.embed(example)

        return embedded

    def loss(self, x, y):
        em_x = self.get_embedding(x)
        em_y = self.get_embedding(y)
        embedding_loss = F.mse_loss(em_x, em_y)

        amp_x = get_amp(x)
        amp_y = get_amp(y)
        loudness_loss = F.mse_loss(amp_x, amp_y)

        return embedding_loss + loudness_loss


def get_amp(example):
    example = rearrange(example, "b c t -> (b c) t")
    example = torch.nn.functional.pad(example, (19200 // 2, 19200 // 2))
    example = example.unfold(1, 19200, 3 * 256)
    example = rearrange(example, "b c t -> (b c) t").unsqueeze(1)

    amp = AF.loudness(example, 48000)

    return amp

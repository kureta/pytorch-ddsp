from pathlib import Path

import librosa
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from src import utils
from src.utils.constants import HOP_LENGTH, N_FFT, SAMPLE_RATE
from src.utils.features import Loudness, get_f0

log = utils.get_pylogger(__name__)


def generate_data(wav_dir: Path, path: Path):
    ld = Loudness().cuda()

    wav_files = list(wav_dir.glob("*.[wav]*"))

    wav_data = []
    ld_data = []
    f0_data = []
    for wf in tqdm(wav_files):
        audio, _ = librosa.load(wf, sr=44100, mono=False, dtype="float32")
        audio = torch.from_numpy(audio).cuda()
        audio = AF.resample(audio.unsqueeze(0), 44100, SAMPLE_RATE)
        if (diff := audio.shape[-1] % HOP_LENGTH) != 0:
            audio = F.pad(audio, (0, HOP_LENGTH - diff))

        wav_data.append(audio[0])

    wav_data = torch.cat(wav_data, dim=1)
    length = wav_data.shape[-1]
    size = 10 * SAMPLE_RATE
    for idx in trange(length // size):
        end = min(idx + size, length)
        audio = wav_data[..., idx:end].unsqueeze(0)
        loudness = ld.get_amp(F.pad(audio, (N_FFT // 2, N_FFT // 2)))
        f0 = get_f0(F.pad(audio, (N_FFT // 2, N_FFT // 2)))

        if loudness.shape != f0.shape or loudness.shape[-1] != (audio.shape[-1] // HOP_LENGTH + 1):
            log.error(
                "Incompatible feature dimensions." f"{loudness.shape, f0.shape, audio.shape}"
            )
            exit()
        ld_data.append(loudness[0].cpu())
        f0_data.append(f0[0].cpu())

    wav_data = wav_data.cpu()
    ld_data = torch.cat(ld_data, dim=1)
    f0_data = torch.cat(f0_data, dim=1)

    data = {"audio": wav_data, "loudness": ld_data, "f0": f0_data}

    torch.save(data, path)

    return data


class DDSPDataset(Dataset):
    def __init__(self, path, wav_dir=None, example_duration=4, example_hop_length=1):
        super().__init__()
        path = Path(path).expanduser()
        if wav_dir is not None:
            wav_dir = Path(wav_dir).expanduser()
        if path.is_file():
            if wav_dir is not None:
                log.warning(
                    "You have provided both saved features path and wave file directory."
                    f"If you want features to be regenerated, delete {path}"
                )
            features = torch.load(path)
        else:
            if wav_dir is None or not wav_dir.is_dir():
                raise FileNotFoundError(f"Wave files directory {wav_dir} does not exist.")

            features = generate_data(wav_dir, path)

        audio_len = SAMPLE_RATE * example_duration
        audio_hop = SAMPLE_RATE * example_hop_length
        feature_len = audio_len // HOP_LENGTH + 1
        feature_hop = audio_hop // HOP_LENGTH
        self.audio = features["audio"].unfold(1, audio_len, audio_hop).transpose(0, 1)
        self.loudness = features["loudness"].unfold(1, feature_len, feature_hop).transpose(0, 1)
        self.f0 = features["f0"].unfold(1, feature_len, feature_hop).transpose(0, 1)

    def __len__(self):
        return len(self.f0)

    def __getitem__(self, idx):
        return self.f0[idx], self.loudness[idx], self.audio[idx]

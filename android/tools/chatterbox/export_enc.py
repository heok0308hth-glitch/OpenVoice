"""Export the voice-registration front-ends of Chatterbox (exact re-implementations, verified):
  s3_mel.onnx     wav24k[1,S] -> prompt mel [1,T,80]            (S3Gen mel_extractor)
  campplus.onnx   wav16k[1,S] -> x-vector [1,192]               (Kaldi fbank + mean norm + CAMPPlus)
  ve_mel.onnx     wav16k[1,S] -> mel [1,T,40]                   (VoiceEncoder librosa mel, power 2)
  ve_lstm.onnx    partials[N,160,40] -> embeds[N,256] (L2-normalized)
"""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, librosa
torch.set_num_threads(4)
import perth
perth.PerthImplicitWatermarker = perth.DummyWatermarker
import torchaudio.compliance.kaldi as K
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

OUT = sys.argv[1]; REF = sys.argv[2]


def dft_basis(n_fft, window):
    n = torch.arange(n_fft).float(); k = torch.arange(n_fft // 2 + 1).float()
    ang = 2 * math.pi * k[:, None] * n[None, :] / n_fft
    return torch.cat([torch.cos(ang), -torch.sin(ang)], 0) * window[None, :]


class S3Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("basis", dft_basis(1920, torch.hann_window(1920))[:, None, :])
        self.register_buffer("mel", torch.from_numpy(librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)).float())
    def forward(self, wav):
        y = F.pad(wav[:, None, :], (720, 720), mode="reflect")
        o = F.conv1d(y, self.basis, stride=480)
        re, im = o[:, :961], o[:, 961:]
        mag = torch.sqrt(re * re + im * im + 1e-9)
        m = torch.matmul(self.mel, mag)
        return torch.log(torch.clamp(m, min=1e-5)).transpose(1, 2)


class KaldiFbank(nn.Module):
    """torchaudio.compliance.kaldi.fbank(num_mel_bins=80) defaults, 16 kHz."""
    def __init__(self):
        super().__init__()
        self.register_buffer("eye", torch.eye(400)[:, None, :])
        n = torch.arange(400).float()
        win = (0.5 - 0.5 * torch.cos(2 * math.pi * n / 399)).pow(0.85)
        basis = dft_basis(512, torch.ones(512))            # frames are zero-padded 400 -> 512
        self.register_buffer("win", win)
        self.register_buffer("basis", basis[:, :400])       # only the first 400 inputs are non-zero
        mel, _ = K.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
        mel = F.pad(mel, (0, 1))                            # 256 -> 257 bins (Nyquist bin weight 0)
        self.register_buffer("mel", mel)
        self.eps = float(torch.finfo(torch.float).eps)
    def forward(self, wav):
        fr = F.conv1d(wav[:, None, :], self.eye, stride=160)[0].transpose(0, 1)   # [T,400]
        fr = fr - fr.mean(1, keepdim=True)
        prev = torch.cat([fr[:, :1], fr[:, :-1]], 1)
        fr = (fr - 0.97 * prev) * self.win
        spec = torch.matmul(fr, self.basis.t())
        re, im = spec[:, :257], spec[:, 257:]
        pw = re * re + im * im
        fb = torch.log(torch.clamp(torch.matmul(pw, self.mel.t()), min=self.eps))
        return fb


class CampWrap(nn.Module):
    def __init__(self, camp):
        super().__init__(); self.fb = KaldiFbank(); self.camp = camp
    def forward(self, wav):
        f = self.fb(wav)
        f = f - f.mean(0, keepdim=True)
        return self.camp(f[None])


class VEMel(nn.Module):
    def __init__(self):
        super().__init__()
        win = torch.from_numpy(librosa.filters.get_window("hann", 400, fftbins=True)).float()
        self.register_buffer("basis", dft_basis(400, win)[:, None, :])
        self.register_buffer("mel", torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=40, fmin=0, fmax=8000)).float())
    def forward(self, wav):
        y = F.pad(wav[:, None, :], (200, 200), mode="reflect")
        o = F.conv1d(y, self.basis, stride=160)
        re, im = o[:, :201], o[:, 201:]
        return torch.matmul(self.mel, re * re + im * im).transpose(1, 2)


if __name__ == "__main__":
    m = ChatterboxMultilingualTTS.from_pretrained("cpu")
    wav24, _ = librosa.load(REF, sr=24000); wav24 = wav24[:240000]
    wav16 = librosa.resample(wav24, orig_sr=24000, target_sr=16000).astype(np.float32)
    w24 = torch.tensor(wav24)[None]; w16 = torch.tensor(wav16)[None]
    with torch.no_grad():
        # S3 mel
        sm = S3Mel().eval()
        ref_mel = m.s3gen.mel_extractor(w24).transpose(1, 2)
        print("s3 mel max diff", float((sm(w24) - ref_mel).abs().max()), tuple(ref_mel.shape))
        torch.onnx.export(sm, (w24,), f"{OUT}/s3_mel.onnx", input_names=["wav"], output_names=["mel"],
                          dynamic_axes={"wav": {1: "S"}, "mel": {1: "T"}}, opset_version=17, dynamo=False)
        # fbank + CAMPPlus
        fb_ref = K.fbank(w16, num_mel_bins=80)
        print("fbank max diff", float((KaldiFbank()(w16) - fb_ref).abs().max()), tuple(fb_ref.shape))
        cw = CampWrap(m.s3gen.speaker_encoder).eval()
        xv_ref = m.s3gen.speaker_encoder.inference([w16[0]])
        xv = cw(w16)
        print("xvector cos", float(F.cosine_similarity(xv, xv_ref).item()))
        torch.onnx.export(cw, (w16,), f"{OUT}/campplus.onnx", input_names=["wav"], output_names=["xvector"],
                          dynamic_axes={"wav": {1: "S"}}, opset_version=17, dynamo=False)
        # voice encoder
        vm = VEMel().eval()
        from chatterbox.models.voice_encoder.melspec import melspectrogram
        mel_ref = melspectrogram(wav16, m.ve.hp).T
        print("ve mel max rel diff", float(np.abs(vm(w16)[0].numpy() - mel_ref).max() / (np.abs(mel_ref).max() + 1e-9)))
        torch.onnx.export(vm, (w16,), f"{OUT}/ve_mel.onnx", input_names=["wav"], output_names=["mel"],
                          dynamic_axes={"wav": {1: "S"}, "mel": {1: "T"}}, opset_version=17, dynamo=False)
        part = torch.randn(3, 160, 40).abs()
        torch.onnx.export(m.ve, (part,), f"{OUT}/ve_lstm.onnx", input_names=["partials"], output_names=["embeds"],
                          dynamic_axes={"partials": {0: "N"}, "embeds": {0: "N"}}, opset_version=17, dynamo=False)
        print("ve ref embed (first 4)", m.ve.embeds_from_wavs([wav16], sample_rate=16000).mean(0)[:4])
    print("done")

"""Reference (PyTorch) OpenVoice V2 Korean pipeline: MeloTTS-KR base TTS + OpenVoice tone color converter."""
import os, sys, json
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "melo_src"))
sys.path.insert(0, "/home/user/OpenVoice")
import torch, numpy as np, librosa
from types import SimpleNamespace

M = os.path.join(W, "models")

def to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
    return d

def load_melo():
    from melo.models import SynthesizerTrn
    cfg = json.load(open(f"{M}/melo_kr/config.json"))
    hps = to_ns(cfg)
    model = SynthesizerTrn(len(cfg["symbols"]), cfg["data"]["filter_length"] // 2 + 1,
                           cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
                           n_speakers=cfg["data"]["n_speakers"], num_tones=cfg["num_tones"],
                           num_languages=cfg["num_languages"], **cfg["model"])
    sd = torch.load(f"{M}/melo_kr/checkpoint.pth", map_location="cpu")
    model.load_state_dict(sd["model"], strict=True)
    model.eval()
    return model, cfg

_bert = {}
def bert_model():
    if not _bert:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        _bert["tok"] = AutoTokenizer.from_pretrained(f"{M}/bert")
        _bert["model"] = AutoModelForMaskedLM.from_pretrained(f"{M}/bert").eval()
    return _bert["tok"], _bert["model"]

def frontend(text, cfg):
    """Exactly MeloTTS KR clean_text + cleaned_text_to_sequence + intersperse."""
    from melo.text import korean
    from melo import commons
    norm = korean.text_normalize(text)
    phones, tones, word2ph = korean.g2p(norm)
    sym2id = {s: i for i, s in enumerate(cfg["symbols"])}
    ids = [sym2id[p] for p in phones]
    tones = [t + 11 for t in tones]
    langs = [4] * len(ids)
    ids = commons.intersperse(ids, 0); tones = commons.intersperse(tones, 0); langs = commons.intersperse(langs, 0)
    w2p = [w * 2 for w in word2ph]; w2p[0] += 1
    return norm, phones, ids, tones, langs, w2p

def bert_feature(norm, w2p, zero=False, n_phone=None):
    if zero:
        return torch.zeros(768, n_phone)
    tok, model = bert_model()
    with torch.no_grad():
        inputs = tok(norm, return_tensors="pt")
        res = model(**inputs, output_hidden_states=True)
        res = res["hidden_states"][-3][0]
    assert inputs["input_ids"].shape[-1] == len(w2p), (inputs["input_ids"].shape, len(w2p))
    feats = torch.cat([res[i].repeat(w2p[i], 1) for i in range(len(w2p))], 0)
    return feats.T

def melo_tts(model, cfg, text, zero_bert=False, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, seed=0):
    norm, phones, ids, tones, langs, w2p = frontend(text, cfg)
    ja_bert = bert_feature(norm, w2p, zero_bert, len(ids))
    bert = torch.zeros(1024, len(ids))
    torch.manual_seed(seed)
    with torch.no_grad():
        audio = model.infer(torch.LongTensor(ids)[None], torch.LongTensor([len(ids)]), torch.LongTensor([0]),
                            torch.LongTensor(tones)[None], torch.LongTensor(langs)[None], bert[None], ja_bert[None],
                            sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                            length_scale=1.0 / speed)[0][0, 0].numpy()
    return audio  # 44100 Hz

def load_converter():
    from openvoice.api import OpenVoiceBaseClass
    # NOTE: upstream ToneColorConverter forwards enable_watermark to the base class (TypeError); build it directly.
    c = OpenVoiceBaseClass(f"{M}/ov2/converter/config.json", device="cpu")
    c.load_ckpt(f"{M}/ov2/converter/checkpoint.pth")
    return c

def src_se():
    return torch.load(f"{M}/ov2/ses/kr.pth", map_location="cpu")

def extract_se_wave(conv, wav22k):
    from openvoice.mel_processing import spectrogram_torch
    hps = conv.hps
    y = torch.FloatTensor(wav22k)[None]
    spec = spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
    with torch.no_grad():
        return conv.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)

def convert_wave(conv, wav22k, s_se, t_se, tau=0.3):
    from openvoice.mel_processing import spectrogram_torch
    hps = conv.hps
    y = torch.FloatTensor(wav22k)[None]
    spec = spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
    with torch.no_grad():
        return conv.model.voice_conversion(spec, torch.LongTensor([spec.size(-1)]), sid_src=s_se, sid_tgt=t_se, tau=tau)[0][0, 0].numpy()

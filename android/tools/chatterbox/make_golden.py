"""Goldens for the Kotlin Chatterbox engine: tokenizer ids, puncNorm, voice features, near-greedy tokens."""
import sys, json, numpy as np, librosa
sys.path.insert(0, sys.path[0])
from cb_onnx import CBOnnx
from chatterbox.models.tokenizers.tokenizer import MTLTokenizer
from chatterbox.mtl_tts import punc_norm
W = sys.argv[1]; M = f"{W}/cbmodel"; OUT = f"{W}/cbgold"
texts = [d["text"] for d in json.load(open(f"{W}/golden_mecab.json"))]
texts += ["Hello, World! 이건 영어 섞인 문장이에요.", "가격은 3,500원이고 10%할인돼요...", "그는 [laughter] 웃었다 [sigh] 그리고…",
          "“따옴표” ‘작은’ — 대시; 콜론: 끝", "ＡＢＣ 全角 ①②", "e-mail@test.com #해시 $100 & co_op", "  여러   공백  ", "ㅋㅋㅋ ㅠㅠ 헐!!", "쉼표 없이 끝나는 문장"]
tok = MTLTokenizer(f"{M}/tokenizer.json")
gold = []
for t in texts:
    pn = punc_norm(t)
    ids = tok.text_to_tokens(pn, language_id="ko")[0].tolist()
    gold.append({"text": t, "punc": pn, "ids": ids})
json.dump(gold, open(f"{OUT}/cb_tok_golden.json", "w"), ensure_ascii=False)
print("tok golden", len(gold))

class Ref(CBOnnx):
    def speech_tokens(self, wav24):
        return self.tokzr.run(None, {"audio_values": wav24[None].astype(np.float32)})[0]
import os
os.symlink(f"{M}/s3_tokenizer.onnx", f"{OUT}/enc/speech_encoder.onnx") if not os.path.exists(f"{OUT}/enc/speech_encoder.onnx") else None
cb = Ref(M, f"{OUT}/enc", lm="t3_lm.onnx")
wav, _ = librosa.load(f"{W}/kref/spk0.wav", sr=24000)
v = cb.make_voice(wav)
np.savez(f"{OUT}/voice_spk0.npz", **v)
print({k: x.shape for k, x in v.items()})
sents = ["안녕하세요, 오늘 날씨가 정말 좋네요.", "이 프로젝트의 목표는 누구나 쉽게 자신의 목소리로 책을 읽어 주는 것입니다."]
res = []
for s in sents:
    sp = cb.generate(punc_norm(s), v, temperature=0.05)
    res.append({"text": s, "tokens": sp.tolist()})
    print(s, len(sp), sp[:20])
json.dump(res, open(f"{OUT}/greedy_spk0.json", "w"), ensure_ascii=False)

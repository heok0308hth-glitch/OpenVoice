"""Golden frontend outputs from the real MeloTTS/g2pkk Python code.
mode=mecab: original g2pkk annotate (mecab-ko) ; mode=distilled: same table-based annotator as the Kotlin port."""
import sys, os, re, json, random
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pyarrow.parquet as pq
from jamo import h2j
import g2pkk.g2pkk as G

def load_table():
    lv = {"A": {}, "B": {}, "C": {}}
    for line in open("/home/user/OpenVoice/android/core/src/main/resources/koreanvoice/p_table.tsv", encoding="utf8"):
        if line.startswith("#") or not line.strip(): continue
        l, k, v = line.rstrip("\n").split("\t"); lv[l][k] = v == "1"
    return lv
LV = load_table()
def tail(ch):
    j = h2j(ch); return j[2] if len(j) == 3 else None
def is_cand(w, i):
    t = tail(w[i])
    if t is None or t not in "ᆫᆬᆷᆱᆰᆲᆴ": return False
    o = h2j(w[i+1])[0] if "가" <= w[i+1] <= "힣" else None
    if o is None: return False
    return o in "ᄀᄁ" if t == "ᆰ" else o in "ᄀᄃᄉᄌ"
def predict(w, i):
    for l, k in (("A", w[max(0, i-1):]), ("B", w[i:]), ("C", w[i:i+2])):
        if k in LV[l]: return LV[l][k]
    t = tail(w[i]); return t is not None and t in "ᆬᆱᆰᆲᆴ" and w[i] not in "닭흙칡삶앎"
def distilled_annotate(string, mecab=None):
    out = []; i = 0
    while i < len(string):
        if not ("가" <= string[i] <= "힣"): out.append(string[i]); i += 1; continue
        j = i
        while j < len(string) and "가" <= string[j] <= "힣": j += 1
        w = string[i:j]
        for k, ch in enumerate(w):
            out.append(ch)
            if k < len(w) - 1 and is_cand(w, k) and predict(w, k): out.append("/P")
        i = j
    return "".join(out)

if __name__ == "__main__":
    mode, n = sys.argv[1], int(sys.argv[2])
    if mode == "distilled": G.annotate = distilled_annotate
    random.seed(1234)
    t = pq.read_table(f"{W}/data/kowiki2.parquet", columns=["text"])
    sents = []
    idx = list(range(t.num_rows)); random.shuffle(idx)
    for i in idx:
        for s in re.split(r"(?<=[.!?])\s+", t.column(0)[i].as_py()):
            s = s.strip()
            if 8 <= len(s) <= 120 and re.fullmatch(r"[가-힣 .,!?]+", s) and re.search(r"[가-힣]{2}", s): sents.append(s)
        if len(sents) >= n: break
    sents = sents[:n]
    extra = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "corpus.txt")) if l.strip()]
    sents = extra + sents
    from melo.text import korean
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{M}/bert")
    cfg = json.load(open(f"{M}/melo_kr/config.json"))
    out = []
    for s in sents:
        norm, phones, ids, tones, langs, w2p = frontend(s, cfg)
        bert_ids = tok(norm)["input_ids"]
        out.append({"text": s, "norm": norm, "phones": phones, "ids": ids, "word2ph": w2p, "bert_ids": bert_ids})
    json.dump(out, open(f"{W}/golden_{mode}.json", "w"), ensure_ascii=False)
    print(mode, len(out))

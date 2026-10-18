"""Distill mecab-based '/P' (verb-stem coda) annotations used by g2pkk into a compact table."""
import re, collections, json, sys, os, pickle
import pyarrow.parquet as pq
from jamo import h2j
import mecab
from g2pkk.utils import annotate
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mc = mecab.MeCab()
FIN = {"ᆫ", "ᆬ", "ᆷ", "ᆱ", "ᆰ", "ᆲ", "ᆴ"}
def fin(ch): j = h2j(ch); return j[-1] if len(j) == 3 else ""
def ons(ch): return h2j(ch)[0]
def fires(f, o):
    if f == "ᆰ": return o in "ᄀᄁ"
    return o in "ᄀᄃᄉᄌ"
def candidates(w):
    return [i for i in range(len(w) - 1) if fin(w[i]) in FIN and fires(fin(w[i]), ons(w[i + 1]))]
def p_positions(w):
    a = annotate(w, mc); pos = set(); k = -1
    for ch in a:
        if "가" <= ch <= "힣": k += 1
        elif ch == "P": pos.add(k)
    return pos
if __name__ == "__main__":
    cnt = collections.Counter()
    t = pq.read_table(f"{W}/data/kowiki2.parquet", columns=["text"])
    for i in range(t.num_rows):
        for w in t.column(0)[i].as_py().split():
            w = re.sub(r"^[^가-힣]+|[^가-힣]+$", "", w)
            if w and re.fullmatch(r"[가-힣]+", w):
                cnt[w] += 1
    print("unique words", len(cnt))
    data = {}
    for w, c in cnt.items():
        cs = candidates(w)
        if cs:
            pp = p_positions(w)
            data[w] = (c, [(i, i in pp) for i in cs])
    print("candidate words", len(data), "positions", sum(len(v[1]) for v in data.values()),
          "P", sum(sum(1 for _, p in v[1] if p) for v in data.values()))
    pickle.dump(data, open(f"{W}/data/p_data.pkl", "wb"))

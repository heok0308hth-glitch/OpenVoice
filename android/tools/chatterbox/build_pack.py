"""Assemble the v2 (Chatterbox Multilingual) model pack: rename the verified exports, split into ≤48 MB parts
for raw GitHub hosting, write modelpack/manifest.json and the app's asset manifest.
usage: build_pack.py <export_dir> <android_dir>
  export_dir must contain: t3_lm_q4.onnx t3_cond_fp16.onnx t3_embed.bin t3_embed.json tokenizer.json
  s3_tokenizer_c.onnx campplus_fp16.onnx s3_mel.onnx ve_mel.onnx ve_lstm.onnx ecapa_fp16.onnx
  s3gen_turbo_det.onnx s3gen_hq_det.onnx (dec_turbo_nb8 / dec_std_nb8 after fix_decoder_noise.py)"""
import sys, os, json, hashlib, shutil
E, A = sys.argv[1], sys.argv[2]
NAMES = {"t3_lm.onnx": "t3_lm_q4.onnx", "t3_cond.onnx": "t3_cond_fp16.onnx", "t3_embed.bin": "t3_embed.bin",
         "t3_embed.json": "t3_embed.json", "tokenizer.json": "tokenizer.json", "s3_tokenizer.onnx": "s3_tokenizer_c.onnx",
         "campplus.onnx": "campplus_fp16.onnx", "s3_mel.onnx": "s3_mel.onnx", "ve_mel.onnx": "ve_mel.onnx",
         "ve_lstm.onnx": "ve_lstm.onnx", "ecapa.onnx": "ecapa_fp16.onnx",
         # decoders after fix_decoder_noise.py (fixed CFM noise like the original PyTorch code)
         "s3gen_turbo.onnx": "s3gen_turbo_det.onnx", "s3gen_hq.onnx": "s3gen_hq_det.onnx"}
PART = 48 * 1000 * 1000
D = f"{A}/modelpack"
if os.path.isdir(D): shutil.rmtree(D)
os.makedirs(D)
files = {}
for dst, src in NAMES.items():
    p = f"{E}/{src}"; data = open(p, "rb").read()
    parts = []
    for i in range(0, max(1, len(data)), PART):
        name = f"{dst}.part{i // PART}"; open(f"{D}/{name}", "wb").write(data[i:i + PART]); parts.append(name)
    files[dst] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts}
    print(dst, round(len(data) / 1e6, 1), "MB", len(parts), "parts")
m = {"version": 2, "engine": "chatterbox-multilingual", "files": files}
json.dump(m, open(f"{D}/manifest.json", "w"), indent=1)
json.dump(m, open(f"{A}/app/src/main/assets/model_manifest.json", "w"), indent=1)
print("total", round(sum(f["size"] for f in files.values()) / 1e6, 1), "MB")

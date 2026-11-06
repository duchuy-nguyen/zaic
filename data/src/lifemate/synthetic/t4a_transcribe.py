import json, torch, soundfile as sf
from pathlib import Path
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from .common import OFFICIAL, RUNS, audio_refs, load_jsonl
REV="a86b604c346caf7148c37512eafe783a16420adb"
OUT=RUNS/"asr_cache"/"transcripts.jsonl"; OUT.parent.mkdir(parents=True,exist_ok=True)
done={json.loads(l)["audio"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()
fit=set(json.loads(Path("manifests/train_fit_provisional.json").read_text()))
samples=[s for s in load_jsonl(OFFICIAL/"train"/"train.jsonl") if s["id"] in fit]
dev="cuda" if torch.cuda.is_available() else "cpu"
m=AutoModelForSpeechSeq2Seq.from_pretrained("vinai/PhoWhisper-small",revision=REV).to(dev).eval()
p=AutoProcessor.from_pretrained("vinai/PhoWhisper-small",revision=REV)
with OUT.open("a",encoding="utf-8") as f:
    for ref in audio_refs(samples):
        if ref in done: continue
        wav,sr=sf.read(OFFICIAL/"train"/ref,dtype="float32")
        inp=p(wav,sampling_rate=sr,return_tensors="pt").to(dev)
        with torch.no_grad():
            txt=p.batch_decode(m.generate(**inp,do_sample=False),skip_special_tokens=True)[0]
        f.write(json.dumps({"audio":ref,"text":txt,"asr_rev":REV},ensure_ascii=False)+"\n")
print("transcribe done ->",OUT)
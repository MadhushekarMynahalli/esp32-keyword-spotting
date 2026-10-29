"""Step 1: download Google Speech Commands v2, simulate the laptop-jack -> ESP32 ADC chain,
compute log-mel features and save them to artifacts/."""
import argparse, json, tarfile, urllib.request
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from config import *
from features import logmel, fix_len

URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
rng = np.random.default_rng(0)


def fetch(tgz):
    """Download with https first, then plain http (some networks break TLS to this host)."""
    part = tgz.with_suffix(".part")
    def hook(b, bs, total):
        print(f"\r  {min(100, b * bs * 100 // max(total, 1))}%", end="", flush=True)
    for url in (URL, URL.replace("https://", "http://")):
        try:
            print("Downloading (~2.3 GB):", url)
            urllib.request.urlretrieve(url, part, hook)
            print()
            part.replace(tgz)
            return
        except Exception as e:                      # SSL errors arrive wrapped in URLError
            print(f"\n  failed: {e}")
    raise SystemExit(
        "\nCould not download automatically (your network is probably blocking/intercepting it).\n"
        "Fix: download this file in your browser (try a phone hotspot if it still fails):\n"
        f"  {URL}\n  (or {URL.replace('https://', 'http://')})\n"
        f"and save it as:\n  {tgz}\nthen run this script again.")


def download():
    DATA_DIR.mkdir(exist_ok=True)
    root = DATA_DIR / "speech_commands_v0.02"
    if (root / "testing_list.txt").exists():
        return root
    tgz = DATA_DIR / "speech_commands_v0.02.tar.gz"
    if not tgz.exists() or tgz.stat().st_size < 2_000_000_000:     # missing or incomplete
        fetch(tgz)
    print("Extracting (takes a few minutes) ...")
    root.mkdir(exist_ok=True)
    with tarfile.open(tgz) as t:
        t.extractall(root, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(root)
    return root


def read_wav(p):
    sr, x = wavfile.read(p)
    assert sr == SR, f"{p}: expected 16 kHz"
    return fix_len(x.astype(np.float32) / 32768.0)


def list_files(root):
    val = set((root / "validation_list.txt").read_text().split())
    test = set((root / "testing_list.txt").read_text().split())
    items = {"train": [], "val": [], "test": []}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        lab = LABELS.index(d.name) if d.name in KEYWORDS else 1
        for f in d.glob("*.wav"):
            rel = f"{d.name}/{f.name}"
            items["val" if rel in val else "test" if rel in test else "train"].append((f, lab))
    return items


def balance(items):
    """keep all keyword clips, subsample _unknown_ to ~1.5x an average keyword class"""
    kw = [i for i in items if i[1] >= 2]
    unk = [i for i in items if i[1] == 1]
    per = len(kw) // len(KEYWORDS)
    keep = rng.choice(len(unk), min(len(unk), int(per * 1.5)), replace=False)
    return kw + [unk[i] for i in keep], per


def load_noise(root):
    return [read_wav_any(p) for p in sorted((root / "_background_noise_").glob("*.wav"))]


def read_wav_any(p):
    sr, x = wavfile.read(p)
    return (x.astype(np.float32) / 32768.0)


def adc_chain(x, noise):
    """Mimic: random volume, room/background noise, ESP32 ADC noise and 12-bit quantisation."""
    n = len(x)
    s = int(rng.integers(-1600, 1600))                       # +-100 ms time shift
    x = np.roll(x, s)
    if s > 0: x[:s] = 0
    elif s < 0: x[s:] = 0
    x = x * 10 ** (rng.uniform(-14, 6) / 20)                 # volume / pot setting
    if rng.random() < 0.8:
        bg = noise[rng.integers(len(noise))]
        o = rng.integers(0, len(bg) - n)
        x = x + bg[o:o + n] * rng.uniform(0, 0.15)
    x = x + rng.normal(0, rng.uniform(1, 12) / ADC_FS, n)    # ADC noise: 1..12 counts rms
    return np.clip(np.round(x * ADC_FS) / ADC_FS, -1, 1).astype(np.float32)


def silence(noise):
    """'silence' = nothing but room noise + ESP32 ADC noise (+ sometimes quiet background audio)"""
    x = np.zeros(CLIP, np.float32)
    if rng.random() < 0.8:
        bg = noise[rng.integers(len(noise))]
        o = rng.integers(0, len(bg) - CLIP)
        x = bg[o:o + CLIP] * rng.uniform(0, 0.3)
    x = x + rng.normal(0, rng.uniform(1, 12) / ADC_FS, CLIP)
    return np.clip(np.round(x * ADC_FS) / ADC_FS, -1, 1).astype(np.float32)


def featurize(gen, total):
    feats, labels, buf = [], [], []
    for x, y in tqdm(gen, total=total):
        buf.append(x); labels.append(y)
        if len(buf) == 512:
            feats.append(logmel(np.stack(buf)).astype(np.float16)); buf = []
    if buf:
        feats.append(logmel(np.stack(buf)).astype(np.float16))
    return np.concatenate(feats), np.array(labels, np.int64)


def make_split(name, items, n_sil, noise, mode, aug_copies):
    def gen():
        for f, y in items:
            x = read_wav(f)
            if mode == "train":
                yield x, y
                for _ in range(aug_copies): yield adc_chain(x, noise), y
            elif mode == "adc":
                yield adc_chain(x, noise), y
            else:
                yield x, y
        for _ in range(n_sil * (1 + aug_copies if mode == "train" else 1)):
            yield silence(noise), 0
    total = (len(items) + n_sil) * (1 + aug_copies if mode == "train" else 1)
    print(f"[{name}] {total} clips")
    return featurize(gen(), total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-copies", type=int, default=1, help="extra ADC-simulated copies per training clip")
    a = ap.parse_args()
    ART_DIR.mkdir(exist_ok=True)
    root = download()
    noise = load_noise(root)
    items = list_files(root)
    out = {}
    for sp in ("train", "val", "test"):
        items[sp], per = balance(items[sp])
        out[sp] = (items[sp], per)

    Xtr, ytr = make_split("train", out["train"][0], out["train"][1], noise, "train", a.aug_copies)
    Xva, yva = make_split("val", out["val"][0], out["val"][1], noise, "clean", 0)
    Xte, yte = make_split("test", out["test"][0], out["test"][1], noise, "clean", 0)
    Xta, yta = make_split("test_adc", out["test"][0], out["test"][1], noise, "adc", 0)

    sub = Xtr[rng.choice(len(Xtr), min(8000, len(Xtr)), replace=False)].astype(np.float64)
    mean, std = float(sub.mean()), float(sub.std())
    print(f"normalisation: mean={mean:.4f} std={std:.4f}")
    json.dump({"mean": mean, "std": std}, open(ART_DIR / "norm.json", "w"))
    for n, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte), ("test_adc", Xta, yta)]:
        np.save(ART_DIR / f"X_{n}.npy", ((X.astype(np.float32) - mean) / std).astype(np.float16))
        np.save(ART_DIR / f"y_{n}.npy", y)
        print(n, X.shape, np.bincount(y, minlength=len(LABELS)))
    print("done -> artifacts/")

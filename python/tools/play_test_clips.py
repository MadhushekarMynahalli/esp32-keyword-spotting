"""Play Speech-Commands TEST clips out of the headphone jack so you can check the whole chain
without speaking.   python play_test_clips.py --word yes --n 8
"""
import argparse, math, random, time
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent / "data" / "speech_commands_v0.02"
ap = argparse.ArgumentParser()
ap.add_argument("--word", default="yes"); ap.add_argument("--n", type=int, default=5)
ap.add_argument("--gain", type=float, default=1.0); ap.add_argument("--pause", type=float, default=1.5)
ap.add_argument("--file"); ap.add_argument("--out-dev", type=int)
a = ap.parse_args()

if a.file:
    files = [Path(a.file)]
else:
    test = [l for l in (ROOT / "testing_list.txt").read_text().split() if l.startswith(a.word + "/")]
    files = [ROOT / p for p in random.sample(test, min(a.n, len(test)))]
fs = int(sd.query_devices(a.out_dev, "output")["default_samplerate"])
g = math.gcd(fs, 16000)
for f in files:
    sr, x = wavfile.read(f)
    x = x.astype(np.float32) / 32768.0
    x = x / (np.abs(x).max() + 1e-9) * 0.6 * a.gain                 # normalise, then apply gain
    x = resample_poly(np.pad(x, (4800, 4800)), fs // g, 16000 // g)
    y = np.clip(np.stack([x, x], 1), -1, 1).astype(np.float32)
    print("playing", f.name if a.file else f"{f.parent.name}/{f.name}")
    sd.play(y, fs, device=a.out_dev); sd.wait(); time.sleep(a.pause)

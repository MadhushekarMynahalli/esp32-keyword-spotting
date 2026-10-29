"""Send audio from your laptop (headset mic, or WAV files) to the ESP32 over USB and show detections.
No analog hardware needed. Close the Arduino serial monitor first (only one program can use the port).

  python stream_to_esp32.py --list                              # show audio devices
  python stream_to_esp32.py --port COM5 --in-dev 1 --gain 3     # live from the headset mic
  python stream_to_esp32.py --port COM5 --wav yes.wav no.wav    # play files into the ESP32
"""
import argparse, queue, threading, time
import numpy as np
import serial
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

BLOCK, SR = 320, 16000


class To16k:
    """Stateful low-pass + resample of a live stream to 16 kHz."""
    def __init__(self, fs):
        self.fs = fs
        self.sos = butter(8, 6800 / (fs / 2), output="sos") if fs != SR else None
        self.zi = np.zeros((self.sos.shape[0], 2)) if self.sos is not None else None
        self.pos, self.prev = 0.0, 0.0

    def __call__(self, x):
        if self.fs == SR:
            return x
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        step = self.fs / SR
        n = int(np.floor((len(y) - 1 - self.pos) / step)) + 1
        if n <= 0:
            self.pos -= len(y); self.prev = y[-1]; return np.zeros(0)
        p = self.pos + step * np.arange(n)
        ext = np.concatenate(([self.prev], y))
        out = np.interp(p + 1, np.arange(len(ext)), ext)
        self.pos = self.pos + step * n - len(y); self.prev = y[-1]
        return out


def pack(block_f):
    q = np.clip(np.round(block_f * 2048.0), -2048, 2047).astype("<i2")
    return b"\xa5\x5a" + q.tobytes()


def reader(ser, debug):
    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except Exception:
            return
        if not line: continue
        p = line.split(",")
        if p[0] == "DET" and len(p) >= 3:
            print(f"\n\033[1;32m>>> {p[1].upper():<8}\033[0m p={p[2]}")
        elif p[0] == "DBG":
            if debug and len(p) >= 6:
                print(f"\n\033[90m  esp: level={p[1]} infer={p[2]}ms top={p[3]} p={p[4]} dropped={p[5]}\033[0m")
        else:
            print("\nESP32:", line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=460800, help="must match AUDIO_BAUD in the sketch")
    ap.add_argument("--in-dev", type=int); ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--wav", nargs="+"); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    if a.list:
        import sounddevice as sd; print(sd.query_devices()); raise SystemExit
    if not a.port: raise SystemExit("--port is required (e.g. COM5 or /dev/ttyUSB0)")

    ser = serial.Serial(a.port, a.baud, timeout=0.5)
    print("Opening port (the ESP32 resets, wait 3 s) ...")
    time.sleep(3)
    threading.Thread(target=reader, args=(ser, a.debug), daemon=True).start()

    if a.wav:                                         # ---- file mode ----
        for f in a.wav:
            sr, x = wavfile.read(f)
            assert sr == SR, f"{f}: must be 16 kHz mono"
            x = x.astype(np.float32) / 32768.0
            if x.ndim > 1: x = x[:, 0]
            x = x / (np.abs(x).max() + 1e-9) * 0.5 * a.gain
            x = np.concatenate([np.zeros(4800, np.float32), x, np.zeros(16000, np.float32)])
            x = np.pad(x, (0, -len(x) % BLOCK))
            print("sending", f); t = time.time()
            for i in range(0, len(x), BLOCK):
                ser.write(pack(x[i:i + BLOCK]))
                t += BLOCK / SR; time.sleep(max(0, t - time.time()))
        time.sleep(1); raise SystemExit

    import sounddevice as sd                           # ---- live mic mode ----
    q = queue.Queue(maxsize=300)
    state = {"acc": np.zeros(0), "peak": 0.0}

    def make_cb(rs):
        def cb(indata, frames, t, status):
            y = rs(indata[:, 0].astype(np.float64) * a.gain)
            state["acc"] = np.concatenate([state["acc"], y])
            while len(state["acc"]) >= BLOCK:
                blk, state["acc"] = state["acc"][:BLOCK], state["acc"][BLOCK:]
                state["peak"] = max(state["peak"], float(np.abs(blk).max()))
                try: q.put_nowait(pack(blk))
                except queue.Full: pass
        return cb

    try:
        fs = SR; stream = sd.InputStream(samplerate=fs, channels=1, dtype="float32", device=a.in_dev,
                                         blocksize=BLOCK, callback=make_cb(To16k(fs)))
    except Exception:
        fs = int(sd.query_devices(a.in_dev, "input")["default_samplerate"])
        stream = sd.InputStream(samplerate=fs, channels=1, dtype="float32", device=a.in_dev,
                                blocksize=fs // 50, callback=make_cb(To16k(fs)))
    print(f"Mic open at {fs} Hz. Speak a keyword!  Level should reach ~0.2-0.6 when you talk (adjust --gain). Ctrl+C stops.")
    last = time.time()
    with stream:
        try:
            while True:
                ser.write(q.get())
                if time.time() - last > 0.5:
                    pk = state["peak"]; state["peak"] = 0.0; last = time.time()
                    print(f"\rmic level {'#' * int(min(pk, 1) * 40):<40} {pk:4.2f}", end="", flush=True)
        except KeyboardInterrupt:
            print("\nbye")

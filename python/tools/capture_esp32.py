"""Record raw ADC samples streamed by firmware/adc_streamer and report signal quality.
Use it to (1) verify wiring, (2) set the volume pot, (3) measure the noise floor,
(4) record your own real-hardware keyword clips for fine-tuning.

  python capture_esp32.py --port COM5 --seconds 5 --plot
  python capture_esp32.py --port /dev/ttyUSB0 --seconds 1.5 --out yes_01.wav
"""
import argparse, struct, time
import numpy as np
import serial
from scipy.io import wavfile

BLOCK, SR = 320, 16000
FRAME = 2 + BLOCK * 2


def capture(port, seconds, baud=460800):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(0.3); ser.reset_input_buffer()
    need = int(seconds * SR / BLOCK)
    buf, blocks = bytearray(), []
    t0 = time.time()
    while len(blocks) < need and time.time() - t0 < seconds + 10:
        buf += ser.read(ser.in_waiting or 1)
        while len(buf) >= FRAME:
            if buf[0] == 0xA5 and buf[1] == 0x5A:
                blocks.append(np.frombuffer(bytes(buf[2:FRAME]), dtype="<u2").copy())
                del buf[:FRAME]
            else:
                del buf[0]
    ser.close()
    return np.concatenate(blocks) if blocks else np.array([], np.uint16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True); ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--out"); ap.add_argument("--plot", action="store_true")
    ap.add_argument("--baud", type=int, default=460800, help="must match Serial.begin() in adc_streamer.ino")
    a = ap.parse_args()
    raw = capture(a.port, a.seconds, a.baud).astype(np.float32)
    if raw.size == 0:
        raise SystemExit("No data. Is adc_streamer flashed? Is the serial monitor closed?")
    x = raw - raw.mean()
    clip = int(((raw <= 5) | (raw >= 4090)).sum())
    print(f"samples      : {raw.size} ({raw.size/SR:.2f} s)")
    print(f"DC level     : {raw.mean():.0f} counts (ideal ~2048, anything 1500-2600 is OK)")
    print(f"peak |signal|: {np.abs(x).max():.0f} counts  (aim for 500-1500 while speech plays)")
    print(f"rms          : {x.std():.1f} counts")
    q = np.sort(np.abs(x))
    print(f"quietest 10% : {q[:len(q)//10].mean():.1f} counts  (~noise floor; set RMS_GATE ~3x this)")
    print(f"clipped      : {clip} samples" + ("   <-- TOO LOUD, turn the pot / laptop volume down" if clip else ""))
    if a.out:
        wavfile.write(a.out, SR, np.clip(x * 16, -32768, 32767).astype(np.int16))   # counts*16 == 16-bit full scale
        print("saved", a.out)
    if a.plot:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(np.arange(len(x)) / SR, x); ax[0].set_ylabel("counts"); ax[0].set_ylim(-2048, 2048)
        ax[1].specgram(x, NFFT=512, Fs=SR, noverlap=352); ax[1].set_xlabel("s"); plt.tight_layout(); plt.show()

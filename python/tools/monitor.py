"""Pretty serial monitor for the KWS firmware.   python monitor.py --port COM5 [--log detections.csv]"""
import argparse, collections, time
import serial

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True); ap.add_argument("--baud", type=int, default=115200)
ap.add_argument("--log"); ap.add_argument("--quiet", action="store_true", help="hide DBG lines")
a = ap.parse_args()
ser = serial.Serial(a.port, a.baud, timeout=1)
log = open(a.log, "a") if a.log else None
counts = collections.Counter()
print("Listening ... (Ctrl+C to quit)")
try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line: continue
        p = line.split(",")
        if p[0] == "DET" and len(p) >= 3:
            counts[p[1]] += 1
            print(f"\033[1;32m>>> {p[1].upper():<8}\033[0m  p={float(p[2]):.2f}   total: {dict(counts)}")
            if log: log.write(f"{time.time():.2f},{p[1]},{p[2]}\n"); log.flush()
        elif p[0] == "DBG" and not a.quiet and len(p) >= 6:
            print(f"\033[90m  level={float(p[1]):6.1f}  infer={p[2]:>4} ms  top={p[3]:<9} p={p[4]}  dropped_blocks={p[5]}\033[0m")
        else:
            print(line)
except KeyboardInterrupt:
    print("\nbye", dict(counts))

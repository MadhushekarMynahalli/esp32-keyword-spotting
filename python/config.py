"""Shared constants. Everything here is exported to firmware/kws_esp32/kws_config.h
by 03_export_headers.py, so training and the ESP32 always use identical settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
FW_DIR = ROOT.parent / "firmware" / "kws_esp32"

# ---- audio / feature settings (do not change one side only!) ----
SR = 16000            # sample rate
CLIP = 16000          # 1 second window fed to the network
WIN = 480             # 30 ms analysis window
HOP = 320             # 20 ms hop
NFFT = 512
NMEL = 40
FMIN, FMAX = 125.0, 7500.0
NFRAMES = 1 + (CLIP - WIN) // HOP     # = 49
ADC_FS = 2048.0       # 12-bit ADC, centred: +-2048 counts == +-1.0 full scale

# ---- classes ----
# NOTE: the repo you referenced uses all/must/none/never/only. Those words are NOT in the
# public Google Speech Commands dataset, so we default to words that are.
KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = ["_silence_", "_unknown_"] + KEYWORDS

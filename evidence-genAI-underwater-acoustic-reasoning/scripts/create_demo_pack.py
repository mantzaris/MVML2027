"""One synthetic interface demonstration, excluded from the real annotation set."""
from pathlib import Path
import numpy as np
import soundfile as sf
from underwater_reasoning.annotation import create_pack
from underwater_reasoning.dsp import record_hash

root = Path(".cache/annotation_demo_source")
root.mkdir(parents=True, exist_ok=True)
fs = 32000
t = np.arange(fs * 5) / fs
x = (.2 * (1 + .5 * np.sin(2 * np.pi * 12 * t)) * np.sin(2 * np.pi * 450 * t)).astype("float32")
sf.write(root / "synthetic.wav", x, fs, subtype="FLOAT")
pack = create_pack([{"clip_id": "synthetic_interface_demo", "waveform_sha256": record_hash(x, fs),
                     "audio_path": "synthetic.wav"}], root, Path(".cache/annotation_demo"),
                    Path(".cache/annotation_demo_private_key.json"))
print("Synthetic interface demo only:", ".cache/annotation_demo/index.html")

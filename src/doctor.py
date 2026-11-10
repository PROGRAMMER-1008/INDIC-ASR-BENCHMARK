"""
Environment check:  python -m src.doctor
Tells you, before a long run, whether your machine is set up correctly.
"""

import importlib
import platform
import shutil
import sys

PACKAGES = [
    ("numpy", True), ("pandas", True), ("matplotlib", True), ("jiwer", True),
    ("psutil", True), ("soundfile", True), ("torch", True), ("transformers", True),
    ("datasets", True), ("librosa", True), ("torchaudio", False), ("faster_whisper", False),
]


def main() -> None:
    print(f"Python      : {sys.version.split()[0]}  ({platform.platform()})")
    ok = True
    for name, required in PACKAGES:
        try:
            m = importlib.import_module(name)
            print(f"  [ok]      {name:15s} {getattr(m, '__version__', '')}")
        except Exception as e:
            tag = "MISSING" if required else "optional"
            print(f"  [{tag:8s}] {name:15s} {type(e).__name__}")
            ok = ok and not required

    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU         : {torch.cuda.get_device_name(0)} (CUDA) -> fp16 inference")
        else:
            print("GPU         : none detected -> CPU inference (use --profile quick first)")
    except ImportError:
        pass

    try:
        import psutil
        print(f"CPU / RAM   : {psutil.cpu_count()} threads, {psutil.virtual_memory().total / 1e9:.1f} GB RAM")
    except ImportError:
        pass
    print(f"Free disk   : {shutil.disk_usage('.').free / 1e9:.1f} GB "
          "(models + data need ~5-10 GB for the standard profile)")

    try:
        import datasets
        major = int(datasets.__version__.split(".")[0])
        if major >= 4:
            print("WARNING     : datasets >= 4.0 cannot load google/fleurs. Run: pip install 'datasets<4'")
            ok = False
    except Exception:
        pass
    print("\nReady to run." if ok else "\nFix the items above, then re-run this check.")


if __name__ == "__main__":
    main()

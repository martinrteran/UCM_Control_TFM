import platform
import subprocess
import sys

is_mac = platform.system() == "Darwin"

if is_mac:
    print("Installing PyTorch for macOS (CPU/MPS)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
else:
    print("Installing PyTorch CUDA 13.2…")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu132"
    ])

print("Installing other dependencies…")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

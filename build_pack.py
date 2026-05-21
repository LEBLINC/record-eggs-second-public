import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    specs = sorted(root.glob("*.spec"))
    if not specs:
        raise SystemExit("No .spec file found in project root.")
    spec = specs[0]
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root))


if __name__ == "__main__":
    main()

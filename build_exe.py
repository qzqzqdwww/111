"""Build the standalone Windows .exe with PyInstaller."""

from __future__ import annotations

import sys
import subprocess
import shutil
from pathlib import Path


def main() -> int:
    if sys.platform != "win32":
        print("This build script is for Windows only.")
        return 1

    # Check PyInstaller is available
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    root = Path(__file__).resolve().parent
    dist_dir = root / "dist"
    build_dir = root / "build"

    # Clean previous builds
    for d in (dist_dir, build_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=ReviewAgent",
        "--windowed",        # No console window (for web UI mode)
        "--console",         # Also build a console version for CLI
        "--add-data", f"{root / 'fixtures'};fixtures",
        "--add-data", f"{root / 'app' / 'static'};app/static",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "app.web",
        "--hidden-import", "app.static",
        "--collect-all", "httpx",
        str(root / "app" / "launcher.py"),
    ]

    print("Building .exe ...")
    print(f"  {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0:
        print("Build failed.")
        return result.returncode

    # Check output
    exe_dir = dist_dir / "ReviewAgent"
    if not exe_dir.exists():
        print(f"Expected output directory not found: {exe_dir}")
        return 1

    exe_path = exe_dir / "ReviewAgent.exe"
    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        return 1

    size_mb = exe_path.stat().st_size / 1024 / 1024
    print(f"\nBuilt: {exe_path}")
    print(f"Size:  {size_mb:.1f} MB")
    print("\nDone. Copy the entire 'dist/ReviewAgent' folder to the target machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

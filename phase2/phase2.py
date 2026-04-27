"""Phase 2 OCR の後方互換エントリポイント。"""

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from phase2.cli import main


if __name__ == "__main__":
    main()
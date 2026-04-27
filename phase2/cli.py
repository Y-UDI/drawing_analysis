import argparse

from .io import print_output_files, print_report, save_outputs
from .pipeline import run_ocr


def main() -> None:
    parser = argparse.ArgumentParser(description="図面解析 Phase 2-A — OCRパイプライン")
    parser.add_argument("input")
    parser.add_argument("--engine", choices=["auto", "vector", "tesseract"], default="auto")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--conf", type=int, default=30)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print(f"[INPUT]  {args.input}")
    result = run_ocr(
        args.input,
        engine=args.engine,
        dpi=args.dpi,
        conf_threshold=args.conf,
    )
    print_report(result)
    saved = save_outputs(result, args.output_dir, write_json=args.json)
    print_output_files(saved)
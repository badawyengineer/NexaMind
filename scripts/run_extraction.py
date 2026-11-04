import argparse
from pathlib import Path

from genai_engineering.core.providers import create_provider
from genai_engineering.task1_structured_extraction.extractor import Extractor

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=["invoice", "resume"])
parser.add_argument("file", type=Path)
args = parser.parse_args()
value = Extractor(create_provider()).extract(
    args.file.read_text(encoding="utf-8"), args.kind
)
print(Extractor.to_json(value))

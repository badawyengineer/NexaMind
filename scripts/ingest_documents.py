import argparse

from genai_engineering.core.providers import create_provider
from genai_engineering.task2_rag.service import RAGService

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
args = parser.parse_args()
service = RAGService(create_provider())
for file in args.files:
    print(f"{file}: added {service.ingest(file)} chunks")

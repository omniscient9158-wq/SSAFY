"""Runtime settings for Solubility Lab."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_DIR
load_dotenv(BASE_DIR / ".env")

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
SOURCE_MAP_PATH = BASE_DIR / "data" / "AqSolDB_원자료_출처매핑.csv"
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(BASE_DIR / "indexes"))).expanduser()
if not INDEX_DIR.is_absolute():
    INDEX_DIR = (BASE_DIR / INDEX_DIR).resolve()
META_COLLECTION = "aqsoldb_compounds_linked"
DOC_COLLECTION = "aqsoldb_documents"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-nano")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"

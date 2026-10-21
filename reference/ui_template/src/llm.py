"""GMS(OpenAI 호환)를 경유하는 LLM / Embedding 클라이언트."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config  # noqa: F401  (.env 로드를 위해 먼저 import)

GMS_BASE_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-nano")
# 키워드 추출은 '실제로 존재하는 위키 문서 제목'을 골라야 해서 추론이 필요하다.
# 추론을 하지 않는 빠른 모델은 없는 제목을 지어낸다. 그래서 따로 지정한다.
KEYWORD_MODEL = os.getenv("KEYWORD_MODEL", "gpt-5.4-nano")
# 문서 선별은 후보 30여 건을 놓고 고르는 판단이라 추론이 필요하다.
SELECT_MODEL = os.getenv("SELECT_MODEL", "gpt-5.4-nano")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def _key() -> str:
    key = os.environ.get("GMS_KEY")
    if not key:
        raise RuntimeError("GMS_KEY가 없습니다. src/.env를 확인하세요.")
    return key


def get_llm(model: str = "", **options) -> ChatOpenAI:
    """모델을 지정하지 않으면 .env의 CHAT_MODEL을 쓴다.

    options는 그대로 넘어간다. (예: reasoning_effort="low")
    """
    return ChatOpenAI(
        model=model or CHAT_MODEL, api_key=_key(), base_url=GMS_BASE_URL, **options
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=_key(), base_url=GMS_BASE_URL)

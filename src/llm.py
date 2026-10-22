"""GMS-compatible LLM and embedding clients. Never print or persist the API key."""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config  # load local .env before reading values


def _api_key() -> str:
    key = os.getenv("GMS_KEY", "").strip()
    if not key:
        raise RuntimeError("GMS_KEY is not configured. Add it to the local .env file.")
    return key


def get_llm(*, timeout: float | None = None, max_retries: int = 2) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.CHAT_MODEL,
        api_key=_api_key(),
        base_url=config.GMS_BASE_URL,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=_api_key(),
        base_url=config.GMS_BASE_URL,
    )

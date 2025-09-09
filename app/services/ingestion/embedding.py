"""텍스트 임베딩 생성을 위한 모듈."""
from __future__ import annotations
from dotenv import load_dotenv
import os
from typing import List, Tuple

from openai import OpenAI

load_dotenv(override=True)

def embed_text(text: str, model: str = "text-embedding-3-small") -> Tuple[List[float], str, int]:
    """주어진 텍스트에 대한 임베딩 벡터를 생성한다.

    환경 변수 ``OPENAI_API_KEY`` 에 저장된 키를 사용해 OpenAI Embedding API를 호출한다.

    Args:
        text: 임베딩을 생성할 문자열.
        model: 사용할 임베딩 모델명.

    Returns:
        (생성된 벡터, 모델명, 벡터 차원)의 튜플.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=text)
    vector = response.data[0].embedding
    return vector, model, len(vector)
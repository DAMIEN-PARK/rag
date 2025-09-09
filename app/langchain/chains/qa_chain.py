from __future__ import annotations

from typing import Any

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

def build_qa_chain(vectorstore: Any):
    """벡터스토어 기반 QA 체인을 생성한다.

    Args:
        vectorstore: ``as_retriever`` 메서드를 제공하는 벡터스토어 인스턴스.

    Returns:
        LangChain Runnable. ``invoke({"question": "..."})`` 형식으로 호출한다.
    """
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template(
        """당신은 질문과 답변을 도와주는 어시스턴트입니다.
#Context:
{context}

#Question:
{question}

#Answer:"""
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

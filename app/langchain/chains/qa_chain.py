from __future__ import annotations

# from dotenv import load_dotenv, find_dotenv
from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from app.utils import format_docs
# load_dotenv(find_dotenv())

def build_qa_chain(vectorstore: VectorStore, llm: BaseLanguageModel, *, k: int = 4):
    """벡터스토어 기반 QA 체인을 생성한다.

    생성된 체인은 ``invoke({"question": "..."})``와 같이 호출한다.

    Args:
        vectorstore: ``as_retriever`` 메서드를 제공하는 벡터스토어 인스턴스.

    Returns:
        LangChain Runnable.

    예시:
        >>> chain = build_qa_chain(vectorstore, llm)
        >>> chain.invoke({"question": "LangChain은 무엇인가요?"})
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
    )

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

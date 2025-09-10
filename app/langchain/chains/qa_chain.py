from __future__ import annotations

# from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
# load_dotenv(find_dotenv())

def build_qa_chain(vectorstore: VectorStore, llm: BaseLanguageModel, *, k: int = 4):
    """벡터스토어 기반 QA 체인을 생성한다.
    Args:
        vectorstore: ``as_retriever`` 메서드를 제공하는 벡터스토어 인스턴스.

    Returns:
        LangChain Runnable. ``invoke({"question": "..."})`` 형식으로 호출한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)
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
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

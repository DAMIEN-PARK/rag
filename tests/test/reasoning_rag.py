
# -*- coding: utf-8 -*-
"""
RAG with LangGraph · Docling · Qdrant · Ollama
- 구성/의존성 빌더 분리
- 노드 팩토리(클로저)로 DI
- 그래프 조립 함수 제공
- 샘플 실행 블록 분리
"""

import warnings

from firebase_admin.firestore_async import client

warnings.filterwarnings("ignore")

from typing import Annotated, List, TypedDict, Literal, Tuple
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_docling import DoclingLoader
# from langchain_docling.loader import ExportType
#
# from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant import Qdrant

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_qdrant import Qdrant
from langchain_community.vectorstores import Qdrant



from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# =========================
# 1) 상태 타입
# =========================
class RAGState(TypedDict):
    query: str
    thinking: str
    documents: List[Document]
    answer: str
    messages: Annotated[List, add_messages]
    mode: str


# =========================
# 2) 구성요소 빌더
# =========================
def build_llms() -> Tuple[ChatOllama, ChatOllama]:
    reasoning_llm = ChatOllama(model="deepseek-r1:7b", stop=["</think>"])
    answer_llm = ChatOllama(model="exaone3.5", temperature=0)
    return reasoning_llm, answer_llm


def load_and_split(file_path: str) -> List[Document]:
    loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
    docs = loader.load()
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
    )
    splits: List[Document] = [s for d in docs for s in splitter.split_text(d.page_content)]
    return splits


def build_retriever(splits: List[Document]) -> ContextualCompressionRetriever:
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    vector_store = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        location=":memory:",
        collection_name="rag_collection_0228",
        retrieval_mode=RetrievalMode.DENSE,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=5)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


def build_chains(reasoning_llm: ChatOllama, answer_llm: ChatOllama):
    reasoning_prompt = ChatPromptTemplate.from_template(
        """주어진 문서를 활용하여 사용자의 질문에 가장 적절한 답변을 작성하세요.

질문: {query}

문서 내용:
{context}

상세 추론:"""
    )
    answer_prompt = ChatPromptTemplate.from_template(
        """사용자 질문에 한글로 답변하세요. 제공된 문서와 추론 과정을 최대한 활용하세요.

질문:
{query}

추론 과정:
{thinking}

문서 내용:
{context}

답변:"""
    )
    reasoning_chain = reasoning_prompt | reasoning_llm | StrOutputParser()
    answer_chain = answer_prompt | answer_llm | StrOutputParser()
    return reasoning_chain, answer_chain


# =========================
# 3) 노드 팩토리(클로저)
# =========================
def make_classify_node():
    def classify_node(state: RAGState):
        q = state["query"]
        mode = "retrieve" if "Docling" in q else "generate"
        print("=====검색 시작=====" if mode == "retrieve" else "=====생성 시작=====")
        return {"mode": mode}
    return classify_node


def make_route_by_mode():
    def route_by_mode(state: RAGState) -> Literal["retrieve", "generate"]:
        return state["mode"]
    return route_by_mode


def make_retrieve_node(compression_retriever: ContextualCompressionRetriever):
    def retrieve(state: RAGState):
        q = state["query"]
        print("=====검색 시작=====")
        documents = compression_retriever.invoke(q)
        for doc in documents:
            print(doc.page_content)
            print("-" * 100)
        print("=====검색 완료=====")
        return {"documents": documents}
    return retrieve


def make_reasoning_node(reasoning_chain):
    def reasoning(state: RAGState):
        q = state["query"]
        docs = state["documents"]
        context = "\n\n".join(d.page_content for d in docs)
        print("=====추론 시작=====")
        thinking = reasoning_chain.invoke({"query": q, "context": context})
        return {"thinking": thinking}
    return reasoning


def make_generate_node(answer_chain):
    def generate(state: RAGState):
        q = state["query"]
        thinking = state.get("thinking", "")
        docs = state.get("documents", [])
        context = "\n\n".join(d.page_content for d in docs)
        print("=====답변 생성 시작=====")
        answer = answer_chain.invoke({"query": q, "thinking": thinking, "context": context})
        print("=====답변 생성 완료=====")
        return {"answer": answer, "messages": [HumanMessage(content=answer)]}
    return generate


# =========================
# 4) 그래프 조립기
# =========================
def build_app(file_path: str):
    # 의존성
    splits = load_and_split(file_path)
    compression_retriever = build_retriever(splits)
    reasoning_llm, answer_llm = build_llms()
    reasoning_chain, answer_chain = build_chains(reasoning_llm, answer_llm)

    # 노드
    classify_node = make_classify_node()
    route_by_mode = make_route_by_mode()
    retrieve_node = make_retrieve_node(compression_retriever)
    reasoning_node = make_reasoning_node(reasoning_chain)
    generate_node = make_generate_node(answer_chain)

    # 그래프
    workflow = StateGraph(RAGState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges("classify", route_by_mode, {
        "retrieve": "retrieve",
        "generate": "generate",
    })
    workflow.add_edge("retrieve", "reasoning")
    workflow.add_edge("reasoning", "generate")
    workflow.add_edge("generate", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app, {"splits": splits}


# =========================
# 5) 샘플 실행
# =========================
if __name__ == "__main__":
    from IPython.display import Image, display

    FILE_PATH = "https://arxiv.org/pdf/2408.09869"
    app, deps = build_app(FILE_PATH)

    # 그래프 이미지
    try:
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:
        pass

    input_ = {"query": "Docling이 다른 라이브러리와 다른 점이 무엇인지 설명해줘"}
    config = {"configurable": {"thread_id": 0}}

    # 이벤트 스트림(필요 시)
    try:
        import asyncio

        async def run():
            async for event in app.astream_events(input=input_, stream_mode="events", version="v2", config=config):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    print(chunk, end="", flush=True)
                elif kind == "on_retriever_end":
                    # 참고: 원 코드에 공백 포함되어 있었음. 수정.
                    print("\n[retriever completed]\n")
                elif kind == "on_chat_model_end":
                    print("\n")

        asyncio.run(run())
    except Exception:
        pass

    snap = app.get_state(config)
    _docs = snap.values.get("documents", [])
    print(f"\nretrieved_docs={len(_docs)}")
    print("state_keys:", list(snap.values.keys()))


# 변경점 요약: 설정·의존성 빌더 → 노드 팩토리 → 그래프 조립 → 실행 순으로 분리. 이벤트 키 `on_retriever_end`

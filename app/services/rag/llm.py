# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
#
# load_dotenv(overrides=True)
#
#
# # 객체 생성
# llm = ChatOpenAI(
#     temperature=0.1,  # 창의성 (0.0 ~ 2.0)
#     model="gpt-4.1-nano",  # 모델명
#     api_key="OPENAI_API_KEY"
# )
#
# model = ChatOpenAI()
# prompt = PromptTemplate.from_template("{topic} 에 대하여 3문장으로 설명해줘.")
# chain = prompt | model | StrOutputParser()
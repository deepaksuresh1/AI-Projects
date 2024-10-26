
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import yaml
from pprint import pprint

RAG_DATABASE = "D:\\Gen AI applications\\AI Projects\\RAG_Application\\Data\\chroma_1.db"
EMBEDDING_MODEL="text-embedding-ada-002"
LLM = "gpt-4-1106-preview"

OpenAI_API_KEY = yaml.safe_load(open("credentials.yml"))['openai']

embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL,api_key=OpenAI_API_KEY)

vectorstore = Chroma(
    persist_directory= RAG_DATABASE,
    embedding_function=embedding_function
    #,similarity_top_k=5,
)

retreiever = vectorstore.as_retriever()

retreiever


# Build a Rag_chain

model = ChatOpenAI(model = LLM, temperature=0.7, api_key= OpenAI_API_KEY)

templete = """Answer the questions based only on the following context: {context}

Question : {question}

"""

prompt = ChatPromptTemplate.from_template(templete)

rag_chain = (
    {"context": retreiever, "question": RunnablePassthrough()}
    |prompt
    |model
    |StrOutputParser()
)

result = rag_chain.invoke("""What are top 3 things needed in a good 
                          social media marketing strategy for Facebook:Meta)? Site any source used.""")

pprint(result)

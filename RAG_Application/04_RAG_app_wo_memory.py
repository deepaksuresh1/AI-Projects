from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import yaml
import os
import streamlit as st

# Key Parameters
RAG_DATABASE = "D:\\Gen AI applications\\AI Projects\\RAG_Application\\Data\\chroma_1.db"
EMBEDDING_MODEL="text-embedding-ada-002"
LLM = "gpt-4-1106-preview"

# Initialize the Streamlit app
st.set_page_config(page_title="Your Marketing AI Assistant", layout="wide")

# Load the API Key securely
OpenAI_API_KEY = yaml.safe_load(open("D:\\Gen AI applications\\AI Projects\\RAG_Application\\credentials.yml"))['openai']

# Function to create the processing chain
def create_chain():
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, api_key= OpenAI_API_KEY,
    )

    vectorstore = Chroma(
        persist_directory=RAG_DATABASE,
        embedding_function=embedding_function
    )

    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(
        model=LLM,
        temperature=0.7,api_key= OpenAI_API_KEY,
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain



# Main app interface
st.title("Your Marketing AI Copilot")

# Layout for input and output
col1, col2 = st.columns(2)

with col1:
    st.header("How can I help you?")
    question = st.text_area("Enter your Marketing question here:", height=300)
    submit_button = st.button("Submit")

with col2:
    st.header("AI Generated Answer")
    if submit_button and question:
        # Process the question through the chain
        with st.spinner("Processing your question..."):
            try:
                # Process the question through the chain
                chain = create_chain()
                answer = chain.invoke(question)
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.write("Your answer will appear here.")

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

import pandas as pd
import pytimetk as tk
import yaml
import os
from pprint import pprint


OpenAI_API_KEY = yaml.safe_load(open("credentials.yml"))['openai']
#load the data

youtube_df = pd.read_csv("D:\\Gen AI applications\\AI Projects\\RAG_Application\\Data\\youtube_videos.csv")

youtube_df.glimpse()

#Documenatation Loader

loader = DataFrameLoader(youtube_df,page_content_column='page_content')

documents = loader.load()

documents[0]

documents[0].metadata
documents[0].page_content

pprint(documents[0].page_content)

#Text Splitter for Chcck if the model can handle tokens properly
for doc in documents:
    print(len(doc.page_content))

#Text Splitting setups

CHUNK_SIZE = 15000

#Character Splitter:  Splits on simple default offsets
# Note:- For transcripts use " " as separator whereas for paragraph use "\n\n".
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                      chunk_overlap=1000,
                                      separator=" ")

docs =  text_splitter.split_documents(documents)

pprint(docs[0].page_content)

len(docs)

#Recursive Character Splitter :  User "smart" splitting and recursiverly tries to split until text is small enough
text_splitter_recursive = RecursiveCharacterTextSplitter(chunk_size= CHUNK_SIZE,
                                        chunk_overlap=1000)

docs_recursive = text_splitter_recursive.split_documents(documents)

len(docs_recursive)

## Post processing the text

for doc in docs_recursive:
    # Retrieve the title and author from the document's metadata
    title = doc.metadata.get('title', 'Unknown Title')
    author = doc.metadata.get('author', 'Unknown Author')
    video_url = doc.metadata.get('video_url', 'Unknown URL')
    
    # Prepend the title and author to the page content
    updated_content = f"Title: {title}\nAuthor: {author}\nVideo URL: {video_url}\n\n{doc.page_content}"
    
    # Update the document's page content
    doc.page_content = updated_content

docs_recursive

pprint(docs_recursive[0].page_content)

#Text Embeddings
embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002',api_key=OpenAI_API_KEY)

#ONLY ONE TIME RUN TO LOAD THE DOCUMENTATION TO VECTORDB
vectorstore = Chroma.from_documents(docs_recursive,
                                    embedding= embedding_function,
                                    persist_directory="D:\\Gen AI applications\\AI Projects\\RAG_Application\\Data\\chroma_1.db")

vectorstore

#THIS WILL LOAD THE DATABASE INSTRED OF CREATING THE DATABASE
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="D:\\Gen AI applications\\AI Projects\\RAG_Application\\Data\\chroma_1.db"
)

vectorstore

#test the case with Similarity Serach

result = vectorstore.similarity_search("How to create a social media strategy", k=4)

result

pprint(result[0].page_content)




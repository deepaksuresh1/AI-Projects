

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from pprint import pprint
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

import yaml

import streamlit as streamlit
import os
from tempfile import NamedTemporaryFile

OpenAI_API_KEY = yaml.safe_load(open("credentials.yml"))['openai']

model = ChatOpenAI(model="gpt-3.5-turbo",
           temperature=0,
           api_key=OpenAI_API_KEY)


#Load and then Summarization PDF files
def load_and_summarize(file, use_template= False):
    
    with NamedTemporaryFile(delete=False, suffix= ".pdf") as tmp:
        tmp.write(file.getvalue())
        file_path = tmp.name
        
        
    try:
        
        loader= PyPDFLoader(file_path)
        documents = loader.load()
        
        if use_template:
            #Bullets
            prompt_templete = """
            write a concise summary of the following:
            {text}

            use 3 to 7 numbered bullet points to describe key points.

            """

            prompt = PromptTemplate(input_variables=["text"],
                                    template=prompt_templete)


            llm_chain = LLMChain(prompt=prompt, llm= model)

            stuff_cahin = StuffDocumentsChain(llm_chain= llm_chain, document_variable_name = "text")

            response= stuff_cahin.invoke(documents)
            
        else:
            #No Bullets
            summarizer_chain = load_summarize_chain(llm = model, chain_type="stuff")
            response = summarizer_chain.invoke(documents)
        
        
    finally:
        
        os.remove(file_path)
        
    return response ["output_text"]



#Streamlit app setup

streamlit.title = "PDF Summarization"

streamlit.subheader("Upload a PDF File")
upload_file = streamlit.file_uploader("Choose PDF File", type="pdf")

if upload_file is not None:
    
    use_template = streamlit.checkbox("Use numbered bullet points? (if not paragraph will be returned)")
    
    if streamlit.button("Summarize PDF"):
        with streamlit.spinner("Summarizing your PDF......"):
            
            summary = load_and_summarize(upload_file, use_template)
            
            streamlit.subheader("Summarization Results")
            streamlit.markdown(summary)
else: 
    
    streamlit.write(" No file is uploaded, PLease upload the file ")


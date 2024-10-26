

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
        
       
        prompt_templete = """
            write a bussiness report from the following earnings call transcript:
            {text}

            Use the following Markdown format:
            # Insert Description Report Title
            
            ## Earning Call Summary
            Use 3 to 7 numbered bullet points
            
            ## Important Financial Information:
            Description the most important financial discussed during the meeting. Use 3 to 5 numbered bullet points.
            
            ## Key Bussiness Risk
            Description the risks discussed during the meeting. Use 3 to 5 numbered bullet points.
            
            ## Key Strategic Actions
            Description the strategic actions discussed during the meeting. Use 3 to 5 numbered bullet points.
            
            ## Conclusion
            conclude with overaching business actions that the company is pursuing thtah may have a positive or negative implications and what those implications are.
            
    
            """

        prompt = PromptTemplate(input_variables=["text"],
                                    template=prompt_templete)


        llm_chain = LLMChain(prompt=prompt, llm= model)

        stuff_cahin = StuffDocumentsChain(llm_chain= llm_chain, document_variable_name = "text")

        response= stuff_cahin.invoke(documents)
            
        
        
    finally:
        
        os.remove(file_path)
        
    return response ["output_text"]



#Streamlit app setup

streamlit.title = "Report Generator"

streamlit.subheader("Report Generator")
upload_file = streamlit.file_uploader("Choose PDF File", type="pdf")

if upload_file is not None:
    
    #use_template = streamlit.checkbox("Use numbered bullet points? (if not paragraph will be returned)")
    
    if streamlit.button("Generate Report"):
        with streamlit.spinner("Generating the Report......"):
            
            summary = load_and_summarize(upload_file)
            
            streamlit.subheader("Your Report")
            streamlit.markdown(summary)
else: 
    
    streamlit.write(" No file is uploaded, PLease upload the file ")


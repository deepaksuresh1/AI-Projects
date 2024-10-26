#import libraries

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from pprint import pprint
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

import yaml

OpenAI_API_KEY = yaml.safe_load(open("credentials.yml"))['openai']

loader = PyPDFLoader(r"D:\\Gen AI applications\\AI Projects\\PDF summarization\\pdf\NIKE-Inc-Q3FY24-OFFICIAL-Transcript-FINAL.pdf")

docs = loader.load()

model = ChatOpenAI(model="gpt-3.5-turbo",
           temperature=0,
           api_key=OpenAI_API_KEY)


summarizer_chain = load_summarize_chain(llm = model, chain_type="stuff")

response = summarizer_chain.invoke(docs)

pprint(response['output_text'])


#Expamding with prompt Engineering 
prompt_templete = """
write a concise summary of the following:
{text}

use 3 to 7 numbered bullet points to describe key points.

"""

prompt = PromptTemplate(input_variables=["text"],
                        template=prompt_templete)


llm_chain = LLMChain(prompt=prompt, llm= model)

stuff_cahin = StuffDocumentsChain(llm_chain= llm_chain, document_variable_name = "text")

response= stuff_cahin.invoke(docs)

pprint(response['output_text'])
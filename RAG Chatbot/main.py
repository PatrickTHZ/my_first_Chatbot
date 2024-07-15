from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class ChatBot():
    load_dotenv()
    loader = TextLoader('./Horoscope.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    force_download = True

    embeddings = HuggingFaceEmbeddings()
    os.environ['PINECONE_API_KEY'] = '604787e3-5ad4-45f5-a275-f1d66780733f'

    # Initialize Pinecone client
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    # Define Index Name
    index_name = "langchain-demo"

    # Define the repo ID and connect to Mixtral model on Huggingface
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.8, "top_k": 50},
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    template = """
    You are a fortune teller. These Human will ask you a questions about their life. 
    Use following piece of context to answer the question. 
    If you don't know the answer, just say you don't know. 
    Keep the answer within 2 sentences and concise.

    Context: {context}
    Question: {question}
    Answer: 

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    rag_chain = (
            {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )


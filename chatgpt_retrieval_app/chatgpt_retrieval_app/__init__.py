import os
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

from django.conf import settings

def init():
    print('initializing the application')

    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    persist_path = settings.PERSIST_PATH
    
    print(f"Opening vectorstore from path {persist_path}")

    vectorstore = Chroma(persist_directory=persist_path, embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    print(f"Setting up retrieval chain")

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        verbose=True
    )
    return chain

print(sys.argv)


import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o-mini")

    query = "tell me LG's history"

    #sample without rag
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    # initialize a vector store object with index name and our embeddings module
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # download retrieval qa chat prompt from lc hub . this is the prompt we gonna send to the llm
    # the prompt sent to llm after we retrieve the info
    ''' system
        Answer any use questions based solely on the context below:
        <context>
        {context}
        </context>
        messages list
        chat_history
        human
        {input}
    '''
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # This creates an LLM Chain where the model receives:
    # the prompt template, a “stuffed” => {list of retrieved documents, the user question}
    # “Stuff” means all docs (top-k retrieved chunks) are stuffed directly into the prompt.

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    '''
    What is this?
    Connects retriever -> document processor → LLM
    vectorstore.as_retriever() wraps your vector database so it can return top-k similar chunks.
    combine_docs_chain tells the LLM how to use the retrieved docs.

    What the chain does internally
    Takes user query. (when chain invoked with input )
    Sends query -> vectorstore to retrieve top-k chunks.
    Feeds chunks -> prompt template → LLM.
    LLM generates final answer.
    '''
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # .invoke() executes the chain one time synchronously.
    # You must send the input argument as a dictionary:
    # { "input": query } matches the prompt field name.
    result = retrival_chain.invoke(input={"input": query})

    print(result)

    '''
 `hub.pull(...)`  Loads an official RAG prompt template       
| `create_stuff_documents_chain`  LLM chain that formats docs + user question 
| `vectorstore.as_retriever()`   Finds top-k relevant chunks                 
| `create_retrieval_chain`    Links retriever → LLM chain                 
| `.invoke()`      Executes the whole pipeline                 
    '''
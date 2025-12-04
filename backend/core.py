from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate

load_dotenv()

from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name="document-helper", embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0, model ="gpt-4o-mini")
    #chat = Ollama(model="llama3")

    retrieval_qa_chat_prompt: PromptTemplate = hub.pull(
        "langchain-ai/retrieval-qa-chat",
    )

    template = """
    Answer any use questions based solely on the context below:

    <context>
    {context}
    </context>

    if the answer is not provided in the context say "Answer not in context"
    Question:
    {input}
    """
    retrieval_qa_chat_prompt2 = PromptTemplate.from_template(template=template)

    stuff_documents_chain = create_stuff_documents_chain(
        chat, retrieval_qa_chat_prompt2
    )

    #normal old retriever
    # qa = create_retrieval_chain(retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain)

    #follow up question is reformatted into a new question that contains or holds all info
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    #retieve context not just based on current prompt but based on our history too
    # basically retrieve based on rephrased prompt
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )


    #basically in our retrieval chain we replaced the normal pinecone retriever with history aware retriever
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    #result = qa.invoke(input={"input": query})
    # new: invoke with chat history
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    #reformatting our result and returning the new version!
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    # res = run_llm(query="What is a LangChain Chain?")
    res = run_llm(query="does LG sell air conditioners?")

    print(res["result"])
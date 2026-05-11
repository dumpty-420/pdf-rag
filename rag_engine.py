import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_ollama import ChatOllama
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LCDocument
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import List, Any

from prompt import QA_PROMPT


def load_and_index_pdfs(directory, api_key):
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    documents = SimpleDirectoryReader(directory).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index


class LlamaIndexRetrieverWrapper(BaseRetriever):
    vector_retriever: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[LCDocument]:
        nodes = self.vector_retriever.retrieve(query)
        return [
            LCDocument(page_content=node.get_content(), metadata=node.metadata)
            for node in nodes
        ]


def get_qa_chain(index, api_key):
    vector_retriever = index.as_retriever(similarity_top_k=5)
    retriever = LlamaIndexRetrieverWrapper(vector_retriever=vector_retriever)

    llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    top_p=0.9,
    top_k=40,
    num_predict=512,   # max output tokens
    num_ctx=4096,      # context window (input token limit)
    repeat_penalty=1.1,
)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    print("LangChain retriever is getting created...")

    qa_chain = (
        {
            "context": RunnableLambda(lambda x: x["input"]) | retriever | format_docs,
            "input": RunnableLambda(lambda x: x["input"])
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain


if __name__ == "__main__":
    print("Loading and indexing PDFs...")
    index = load_and_index_pdfs("rag_files", api_key=None)
    print("Index created!")

    print("Creating QA chain...")
    qa_chain = get_qa_chain(index, api_key=None)
    print("QA chain created!")

    print("Testing query...")
    response = qa_chain.invoke({"input": "What happens to firm investment when monetary policy tightens?"})
    print("Answer:", response)





# import os
# from llama_index.core import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     Settings,
#     StorageContext,
# )
# from llama_index.llms.gemini import Gemini
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from prompt import QA_PROMPT


# def load_and_index_pdfs(directory, api_key):
#     # Set up LlamaIndex Settings (No LLM for indexing, only Embeddings)
#     Settings.llm = None
#     Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     # Load documents
#     documents = SimpleDirectoryReader(directory).load_data()

#     # Create Index
#     index = VectorStoreIndex.from_documents(documents)
#     return index


# from langchain_core.retrievers import BaseRetriever
# from typing import List, Any
# from langchain_core.documents import Document as LCDocument
# from langchain_huggingface import HuggingFacePipeline


# class LlamaIndexRetrieverWrapper(BaseRetriever):
#     vector_retriever: Any

#     def _get_relevant_documents(
#         self, query: str, *, run_manager=None
#     ) -> List[LCDocument]:
#         nodes = self.vector_retriever.retrieve(query)
#         docs = []
#         for node in nodes:
#             docs.append(
#                 LCDocument(page_content=node.get_content(), metadata=node.metadata)
#             )
#         return docs


# def get_qa_chain(index, api_key):
#     # Create LangChain Retriever from LlamaIndex
#     vector_retriever = index.as_retriever(similarity_top_k=5)

#     retriever = LlamaIndexRetrieverWrapper(vector_retriever=vector_retriever)

#     # Create LangChain LLM (Using local Qwen model)
#     llm = HuggingFacePipeline.from_model_id(
#         model_id="Qwen/Qwen2.5-0.5B-Instruct",
#         task="text-generation",
#         pipeline_kwargs={
#             "max_new_tokens": 512,
#             "do_sample": True,
#             "temperature": 0.3,
#             "top_k": 50,
#             "top_p": 0.95,
#         },
#         model_kwargs={"torch_dtype": "auto", "low_cpu_mem_usage": True},
#     )

#     # Create QA Chain
#     print("langchain retriever is getting created")
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={"prompt": QA_PROMPT},
#         return_source_documents=True,
#     )

#     return qa_chain

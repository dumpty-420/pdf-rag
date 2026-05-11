# from langchain_core.prompts import ChatPromptTemplate

# QA_PROMPT = ChatPromptTemplate.from_template("""
# Answer the question based only on the following context.
# If you don't know the answer, say "I don't know."

# Context: {context}

# Question: {input}

# Answer:
# """)


# from langchain_core.prompts import PromptTemplate

# template = """Answer the question based only on the following context:

# {context}

# Question: {question}

# Answer:"""

# QA_PROMPT = PromptTemplate(
#     template=template, input_variables=["context", "question"]
# )


# from langchain_core.prompts import PromptTemplate

# # Enhanced prompt template for RAG
# template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {input}
# Helpful Answer:"""

# QA_PROMPT = PromptTemplate(
#     template=template, input_variables=["context", "input"]
# )

# # Query Transformation Prompt for Multi-Query Retriever
# query_gen_template = """You are an AI assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. 
# By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
# Provide these alternative questions separated by newlines. Do not add any numbering or labels.

# Original question: {input}

# Alternative questions:"""

# QUERY_GEN_PROMPT = PromptTemplate(
#     template=query_gen_template,
#     input_variables=["input"]
# )


from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Main QA Prompt
QA_PROMPT = ChatPromptTemplate.from_template(
    "Use the following pieces of context to answer the question at the end.\n"
    "If you don't know the answer, just say that you don't know.\n\n"
    "Instructions:\n"
    "1. Answer should be based on the PDFs information only.\n"
    "2. Do not hallucinate or give repetitive answers.\n"
    "3. Do not add any additional information not present in the PDFs.\n"
    "4. Be concise and specific.\n\n"
    "Example of a good answer:\n"
    "Question: What happens to firm borrowing when interest rates rise?\n"
    "Answer: When interest rates rise, firms facing multiple tight financial constraints "
    "reduce their external borrowing significantly more than unconstrained firms. "
    "The most rate-sensitive constraint becomes binding, curtailing the firm's ability to borrow.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Helpful Answer:"
)

# Query Transformation Prompt
QUERY_GEN_PROMPT = PromptTemplate(
    template=(
        "You are an AI assistant. Your task is to generate 3 different versions "
        "of the given user question to retrieve relevant documents from a vector database.\n"
        "Provide these alternative questions separated by newlines. "
        "Do not add any numbering or labels.\n\n"
        "Original question: {question}\n\n"
        "Alternative questions:"
    ),
    input_variables=["question"]
)
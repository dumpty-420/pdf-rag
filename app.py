# import streamlit as st
# import os
# import subprocess
# import time
# from dotenv import load_dotenv
# from query_engine import get_query_engine, get_qa_chain_from_index

# load_dotenv()

# st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚")
# st.title("📚 World Econ Paper Q&A (FAISS Edition)")

# # Sidebar for API Key
# with st.sidebar:
#     st.header("Settings")

#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         api_key = st.text_input("Enter Gemini API Key", type="password")
#     else:
#         st.success("Gemini API Key loaded")

#     st.divider()

#     if st.button("Process PDFs (Ingest)"):
#         with st.spinner("Running ingestion script (Local FAISS + HuggingFace)..."):
#             try:
#                 import sys

#                 result = subprocess.run(
#                     [sys.executable, "ingestion.py"], capture_output=True, text=True
#                 )
#                 if result.returncode == 0:
#                     st.success("Ingestion complete!")
#                     st.text(result.stdout)
#                 else:
#                     st.error("Ingestion failed.")
#                     st.text(result.stderr)
#             except Exception as e:
#                 st.error(f"Error running ingestion: {e}")

# # Main Chat Interface
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Ask a question about the papers"):
#     if not os.path.exists("./storage_index"):
#         st.error(
#             "Index not found. Please click 'Process PDFs' to create the vector database."
#         )
#     else:
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     index = get_query_engine(api_key)
#                     qa_chain = get_qa_chain_from_index(index, api_key)

#                     # Retry logic for Quota Limits
#                     max_retries = 3
#                     response = None
#                     for attempt in range(max_retries):
#                         try:
#                             # Chain now takes a plain string and returns a plain string
#                             response = qa_chain.invoke(prompt)
#                             break
#                         except Exception as e:
#                             error_str = str(e)
#                             if "429" in error_str or "ResourceExhausted" in error_str:
#                                 if attempt < max_retries - 1:
#                                     wait_time = 10 * (attempt + 1)
#                                     st.warning(
#                                         f"Rate limit hit. Retrying in {wait_time} seconds..."
#                                     )
#                                     time.sleep(wait_time)
#                                 else:
#                                     raise e
#                             else:
#                                 raise e

#                     if response:
#                         # Response is now a plain string
#                         st.markdown(response)
#                         st.session_state.messages.append(
#                             {"role": "assistant", "content": response}
#                         )

#                 except Exception as e:
#                     st.error(f"Error generating answer: {e}")


import streamlit as st
import os
import subprocess
import traceback
import time
from dotenv import load_dotenv
from query_engine import get_query_engine, get_qa_chain_from_index

load_dotenv()

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚")
st.title("📚 World Econ Paper Q&A (FAISS Edition)")

# Sidebar
with st.sidebar:
    st.header("Settings")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter API Key", type="password")
    else:
        st.success("API Key loaded")

    st.divider()

    if st.button("Process PDFs (Ingest)"):
        with st.spinner("Running ingestion script..."):
            try:
                import sys

                result = subprocess.run(
                    [sys.executable, "ingestion.py"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    st.success("Ingestion complete!")
                    st.text(result.stdout)
                else:
                    st.error("Ingestion failed.")
                    st.text(result.stderr)
            except Exception as e:
                st.error(f"Error running ingestion: {e}")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the papers"):
    if not os.path.exists("./storage_index"):
        st.error(
            "Index not found. Please click 'Process PDFs' to create the vector database."
        )
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    index = get_query_engine(api_key)
                    qa_chain = get_qa_chain_from_index(index, api_key)

                    max_retries = 3
                    response = None
                    for attempt in range(max_retries):
                        try:
                            response = qa_chain.invoke({"question": prompt})
                            break
                        except Exception as e:
                            error_str = str(e)
                            if "429" in error_str or "ResourceExhausted" in error_str:
                                if attempt < max_retries - 1:
                                    wait_time = 10 * (attempt + 1)
                                    st.warning(
                                        f"Rate limit hit. Retrying in {wait_time} seconds..."
                                    )
                                    time.sleep(wait_time)
                                else:
                                    raise e
                            else:
                                raise e

                    if response:
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.code(traceback.format_exc())

# import streamlit as st
# import os
# import subprocess
# from dotenv import load_dotenv
# from query_engine import get_query_engine, get_qa_chain_from_index

# load_dotenv()

# st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚")

# st.title("📚 World Econ Paper Q&A (FAISS Edition)")

# # Sidebar for API Key
# with st.sidebar:
#     st.header("Settings")

#     # Check for Gemini Key
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         api_key = st.text_input("Enter Gemini API Key", type="password")
#     else:
#         st.success("Gemini API Key loaded")

#     st.divider()

#     if st.button("Process PDFs (Ingest)"):
#         with st.spinner("Running ingestion script (Local FAISS + HuggingFace)..."):
#             try:
#                 # Run the ingestion script as a subprocess
#                 # No API key needed for ingestion now (using local embeddings)
#                 import sys

#                 result = subprocess.run(
#                     [sys.executable, "ingestion.py"], capture_output=True, text=True
#                 )

#                 if result.returncode == 0:
#                     st.success("Ingestion complete!")
#                     st.text(result.stdout)
#                 else:
#                     st.error("Ingestion failed.")
#                     st.text(result.stderr)
#             except Exception as e:
#                 st.error(f"Error running ingestion: {e}")

# import time

# # Main Chat Interface
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Ask a question about the papers"):
#     if not os.path.exists("./storage_index"):
#         st.error(
#             "Index not found. Please click 'Process PDFs' to create the vector database."
#         )
#     else:
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     # Get index and chain
#                     index = get_query_engine(api_key)
#                     qa_chain = get_qa_chain_from_index(index, api_key)

#                     # Retry logic for Quota Limits
#                     max_retries = 3
#                     response = None
#                     for attempt in range(max_retries):
#                         try:
#                             response = qa_chain.invoke({"query": prompt})
#                             break
#                         except Exception as e:
#                             # Check for ResourceExhausted or similar 429
#                             error_str = str(e)
#                             if "429" in error_str or "ResourceExhausted" in error_str:
#                                 if attempt < max_retries - 1:
#                                     wait_time = 10 * (
#                                         attempt + 1
#                                     )  # simple backoff: 10s, 20s
#                                     st.warning(
#                                         f"Rate limit hit. Retrying in {wait_time} seconds..."
#                                     )
#                                     time.sleep(wait_time)
#                                 else:
#                                     raise e
#                             else:
#                                 raise e

#                     if response:
#                         answer = response["result"]
#                         st.markdown(answer)
#                         st.session_state.messages.append(
#                             {"role": "assistant", "content": answer}
#                         )

#                         with st.expander("Source Documents"):
#                             for doc in response["source_documents"]:
#                                 source = doc.metadata.get("file_name", "Unknown")
#                                 page = doc.metadata.get("page_label", "N/A")
#                                 st.write(f"**{source} - Page {page}**")
#                                 st.text(doc.page_content[:200] + "...")
#                 except Exception as e:
#                     st.error(f"Error generating answer: {e}")

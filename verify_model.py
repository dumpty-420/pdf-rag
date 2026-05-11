print("Importing...")
from langchain_ollama import ChatOllama

print("Loading model...")
try:
    llm = ChatOllama(
        model="llama3.2",
        temperature=0.3,
    )
    print("Model loaded successfully!")
    print("Testing generation...")
    response = llm.invoke("Hello, who are you?")
    print(response.content)
except Exception as e:
    print(f"Error: {e}")


# import os

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


# print("Importing...")
# from langchain_huggingface import HuggingFacePipeline
# import torch

# print("Loading model...")
# try:
#     llm = HuggingFacePipeline.from_model_id(
#         model_id="Qwen/Qwen2.5-0.5B-Instruct",
#         task="text-generation",
#         model_kwargs={
#             "torch_dtype": "auto",
#             "low_cpu_mem_usage": True,
#             "device_map": "cpu",  # force CPU
#         },
#         pipeline_kwargs={"max_new_tokens": 100},
#     )
#     print("Model loaded successfully!")
#     print("Testing generation...")
#     print(llm.invoke("Hello, who are you?"))
# except Exception as e:
#     print(f"Error: {e}")

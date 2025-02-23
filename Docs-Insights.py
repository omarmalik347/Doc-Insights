import os
import pandas as pd
import fitz
from transformers import AutoTokenizer, AutoModelForCausalLM
from pandasai import PandasAI
import streamlit as st


api_key = os.getenv("HF_API_KEY")

if api_key is None:
    st.warning("API key is not set. Please ensure it's configured correctly.")
else:
    st.info("API key loaded successfully")


model_name = "meta-llama/llama-2-7b-chat"

# Load the model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key)

# wrap
class HuggingFaceLLM:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# file uploads
def read_csv(file_path):
    return pd.read_csv(file_path)


def read_excel(file_path):
    return pd.read_excel(file_path)


def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# file handler
def read_file(file):
    if file.type == "application/pdf":
        text = read_pdf(file)
        return text
    elif file.type == "text/csv":
        return read_csv(file)
    elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        return read_excel(file)
    else:
        st.warning("Unsupported file format.")
        return None

# Streamlit
st.title("Data Query App using Hugging Face")

# File upload
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "pdf"])


if uploaded_file is not None:

    data = read_file(uploaded_file)

    if isinstance(data, pd.DataFrame):
        st.write("### Data Preview")
        st.write(data.head())
        

        llm = HuggingFaceLLM(tokenizer, model)
        pandas_ai = PandasAI(llm)
        
        # User input
        query = st.text_input("Ask a question about the data:")
        if query:
            response = pandas_ai.run(data, prompt=query)
            st.write(f"Answer: {response}")
    
    elif isinstance(data, str):
        st.write("### Extracted Text from PDF:")
        st.write(data[:1000]) 
        

        llm = HuggingFaceLLM(tokenizer, model)
        
        # User input for query
        query = st.text_input("Ask a question about the text:")
        if query:
            response = llm.query(f"Please summarize the following text: {data[:2000]}")  # Limit to first 2000 characters
            st.write(f"Answer: {response}")

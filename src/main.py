# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a simple standalone implementation showing a rag pipeline using a small local model (TinyLlama).
# It uses a simple Streamlit UI and one file implementation of a minimalistic RAG pipeline.

############################################
# Component #1 - Document Loader
############################################

import streamlit as st
import os
import shutil

# Keep everything local: never contact Hugging Face (models are loaded from models/), and
# turn off unstructured's usage analytics. These must be set before
# transformers/huggingface_hub are imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DO_NOT_TRACK", "true")

st.set_page_config(layout = "wide")

# Data folders live in the project root (one level up from src/), whatever folder you run from
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_store_path = os.path.join(PROJECT_DIR, "vectorstore")

def clear_vector_store():
    # Deleting the saved store forces it to be rebuilt from the current documents
    shutil.rmtree(vector_store_path, ignore_errors=True)

with st.sidebar:
    DOCS_DIR = os.path.join(PROJECT_DIR, "uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files = True)
        submitted = st.form_submit_button("Upload!")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
                f.write(uploaded_file.read())
        clear_vector_store()

    st.subheader("Documents in the Knowledge Base")
    doc_names = sorted(name for name in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, name)))
    if not doc_names:
        st.caption("No documents yet.")
    for name in doc_names:
        name_col, remove_col = st.columns([4, 1])
        name_col.markdown(f"📄 {name}")
        if remove_col.button("✕", key=f"remove-{name}", help=f"Remove {name}"):
            os.remove(os.path.join(DOCS_DIR, name))
            clear_vector_store()
            st.rerun()

############################################
# Component #2 - Embedding Model and LLM
############################################

import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_community.embeddings import HuggingFaceEmbeddings

# Both models run locally from the project's models/ folder - no API key needed.
# Download them once with the commands in readme-setup.md.
LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_DIR = os.path.join(PROJECT_DIR, "models", "TinyLlama")
EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIR = os.path.join(PROJECT_DIR, "models", "all-MiniLM-L6-v2")

for name, path in [(LLM_NAME, LLM_DIR), (EMBEDDING_NAME, EMBEDDING_DIR)]:
    if not os.path.isdir(path):
        st.error(f"Model not found in `{path}`. Download it from the project folder with:\n\n"
                 f"`hf download {name} --local-dir {os.path.relpath(path, PROJECT_DIR)}`")
        st.stop()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@st.cache_resource(show_spinner="Loading TinyLlama...")
def load_llm():
    device = get_device()
    # float16 is only reliable on GPU; use float32 on CPU
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(LLM_DIR)
    model = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype=dtype).to(device)
    return tokenizer, model

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_DIR)

tokenizer, model = load_llm()
embedder = load_embedder()

with st.sidebar:
    st.caption(f"LLM: {LLM_NAME} on {model.device}")

############################################
# Component #3 - Vector Database Store
############################################

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Load raw documents from the directory. PDFs use pypdf, since unstructured's
# PDF support needs heavy extra dependencies (poppler, tesseract, etc).
pdf_documents = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
other_documents = DirectoryLoader(DOCS_DIR, exclude=["**/*.pdf"]).load()
raw_documents = pdf_documents + other_documents


# Check for existing vector store
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    vectorstore = FAISS.load_local(vector_store_path, embedder, allow_dangerous_deserialization=True)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                # TinyLlama only has a 2048 token context, so keep chunks small
                text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, embedder)

            with st.spinner("Saving vector store"):
                vectorstore.save_local(vector_store_path)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

############################################
# Component #4 - LLM Response Generation and Chat
############################################

st.subheader("Chat with your AI Assistant, Envie!")
# Inference runs inside this process, so its network activity can be checked in Resource Monitor
st.caption(f"Process ID: {os.getpid()} · Model: {LLM_NAME} on {model.device} · Location: {LLM_DIR}")

if "messages" not in st.session_state:
    st.session_state.messages = []

def source_label(doc):
    name = os.path.basename(doc.metadata.get("source", ""))
    page = doc.metadata.get("page")
    # PyPDFLoader pages are 0-indexed
    return f"{name}, page {page + 1}" if page is not None else name

def show_sources(sources):
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"**{source['label']}**")
            st.caption(source["text"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

SYSTEM_PROMPT = "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."

def stream_response(user_content):
    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        streamer=streamer,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    for text in streamer:
        yield text

user_input = st.chat_input("Ask a question about your documents")

if user_input and vectorstore is None:
    st.warning("Upload a document first so Envie has something to answer from.")

if user_input and vectorstore is not None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in stream_response(augmented_user_input):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        # Show exactly which chunks of the documents were given to the model
        sources = [{"label": source_label(doc), "text": doc.page_content} for doc in docs]
        show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})

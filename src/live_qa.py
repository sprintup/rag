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

# Live Q&A with Envie, for a presentation: the same RAG steps as main.py, answering questions
# about this project's own files with Qwen 3.5, served by Ollama. Everyone who opens the app sees
# one shared chat, and questions wait in a queue and are answered one at a time. There are no
# uploads.
#
# Run it with:  streamlit run src/live_qa.py   (Ollama must be installed, with qwen3.5:4b pulled)
# The terminal then shows a presenter key; open the app with ?key=<that key> to control the queue.
# See readme-live-qa.md, for running it on your computer or in GitHub Codespaces.

import hmac
import html
import itertools
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from threading import Thread

import requests
import streamlit as st

# Keep everything local: never contact Hugging Face (the embedding model is loaded from models/).
# This must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(layout = "wide")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Envie answers from every .md and .py file in the project, found by walking all its folders.
# Only these two file types are read, so secrets such as a .env file are never included.
DOC_EXTENSIONS = (".md", ".py")
# Folders that are skipped: installed libraries, downloaded models, generated files and caches.
# Hidden folders (starting with ".") are skipped too, except .github with the skills and agents.
SKIP_FOLDERS = {"models", "vectorstore", "uploaded_docs", "node_modules", "__pycache__"}

def project_files():
    for folder, subfolders, files in os.walk(PROJECT_DIR):
        # Changing subfolders in place tells os.walk not to go into the skipped ones
        subfolders[:] = sorted(name for name in subfolders
                               if name not in SKIP_FOLDERS and "venv" not in name
                               and (not name.startswith(".") or name == ".github"))
        for name in sorted(files):
            if name.endswith(DOC_EXTENSIONS):
                yield os.path.join(folder, name)

MAX_QUESTION_CHARS = 300   # longer questions are refused
MAX_QUEUE = 30             # questions waiting at once, from everyone
MAX_WAITING_PER_PERSON = 2 # questions waiting at once, from one browser

############################################
# The models and the search index
############################################

# The model is served by Ollama on this computer, through its OpenAI-compatible API
OLLAMA_URL = "http://127.0.0.1:11434"
LLM_NAME = "qwen3.5:4b"
EMBEDDING_DIR = os.path.join(PROJECT_DIR, "models", "all-MiniLM-L6-v2")

if not os.path.isdir(EMBEDDING_DIR):
    st.error(f"Embedding model not found in `{EMBEDDING_DIR}`. Download it first: see 01_readme-setup.md.")
    st.stop()

def ollama_models():
    # The names of the models Ollama has downloaded, or None if Ollama isn't answering
    try:
        return [model["name"] for model in requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()["models"]]
    except requests.RequestException:
        return None

@st.cache_resource(show_spinner="Starting Ollama...")
def start_ollama():
    # In a codespace, Ollama doesn't start by itself after a restart, so start it if it's installed
    if ollama_models() is None and shutil.which("ollama"):
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        for _ in range(20):
            time.sleep(0.5)
            if ollama_models() is not None:
                break

start_ollama()
models = ollama_models()
if models is None:
    st.error(f"Can't reach Ollama at {OLLAMA_URL}. Start Ollama, then reload this page (see readme-live-qa.md).")
    st.stop()
if LLM_NAME not in models:
    st.error(f"Ollama doesn't have {LLM_NAME} yet. Download it with `ollama pull {LLM_NAME}`, then reload this page.")
    st.stop()

@st.cache_resource(show_spinner="Indexing this project's files (once, when the app starts)...")
def build_vectorstore():
    # Short chunks keep the prompt short, and on a CPU a shorter prompt means a faster answer. This
    # splitter cuts at blank lines first, then at line ends, so code is split into short chunks too.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
    chunks = []
    for path in project_files():
        document = TextLoader(path, encoding="utf-8").load()[0]
        for chunk in splitter.split_documents([document]):
            # Turn the chunk's position in the file into line numbers, so answers can cite them
            first = document.page_content.count("\n", 0, chunk.metadata["start_index"]) + 1
            chunk.metadata["lines"] = (first, first + chunk.page_content.count("\n"))
            chunks.append(chunk)
    # Kept in memory and built once for everyone, so no pickle file is written or loaded
    return FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name=EMBEDDING_DIR))

vectorstore = build_vectorstore()

############################################
# Answering a question (the same steps as main.py)
############################################

SYSTEM_PROMPT = ("You are a helpful AI assistant named Envie. You will reply to questions only based on the context "
                 "that you are provided. If something is out of context, you will refrain from replying and politely "
                 "decline to respond to the user. Each part of the context starts with its source in square brackets, "
                 "e.g. [src/main.py, lines 100-107]. Whenever you use a part, cite its source in your answer, written "
                 "exactly the same way.")

def source_label(doc):
    # e.g. "src/main.py, lines 100-107"
    first, last = doc.metadata["lines"]
    lines = f"line {first}" if first == last else f"lines {first}-{last}"
    return f"{os.path.relpath(doc.metadata['source'], PROJECT_DIR).replace(os.sep, '/')}, {lines}"

def build_prompt(question):
    # Retrieve the 4 most similar chunks, and put them above the question, each with its source
    docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
    context = "".join(f"[{source_label(doc)}]\n{doc.page_content}\n\n" for doc in docs)
    prompt = "Context:\n\n" + context + "Question: " + question + "\n"
    # Keep exactly which chunks were given to the model, to show as sources
    sources = [{"label": source_label(doc), "text": doc.page_content} for doc in docs]
    return prompt, sources

def stream_response(user_content):
    # The same OpenAI-style request the VS Code plugin sends; "stream" asks for the reply piece by
    # piece, as server-sent events ("data: {...}" lines)
    response = requests.post(f"{OLLAMA_URL}/v1/chat/completions", stream=True, timeout=600, json={
        "model": LLM_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
        "max_tokens": 400,
        "stream": True,
        # Qwen thinks before answering by default, which takes minutes on a CPU: answer straight away
        "reasoning_effort": "none",
    })
    response.raise_for_status()
    response.encoding = "utf-8"
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        text = json.loads(data)["choices"][0]["delta"].get("content")
        if text:
            yield text

############################################
# One shared chat and a queue of questions
############################################

class LiveChat:
    """One conversation shared by every browser: a queue of questions, answered one at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.queue = []       # questions waiting: {"id", "text", "owner"}
        self.messages = []    # answered questions: {"question", "answer", "sources"}
        self.current = None   # the question being answered now, in the same form
        self.paused = False
        self.ids = itertools.count(1)
        # Qwen answers one question at a time, in this background thread
        Thread(target=self.answer_questions, daemon=True).start()

    def add(self, text, owner):
        text = text.strip()
        with self.lock:
            if len(text) > MAX_QUESTION_CHARS:
                return f"Please keep questions under {MAX_QUESTION_CHARS} characters."
            if len(self.queue) >= MAX_QUEUE:
                return "The queue is full. Try again in a moment."
            if sum(item["owner"] == owner for item in self.queue) >= MAX_WAITING_PER_PERSON:
                return "You already have questions waiting. Wait for Envie to answer them first."
            self.queue.append({"id": next(self.ids), "text": text, "owner": owner})
            return f"Added to the queue, number {len(self.queue)}."

    def remove(self, item_id):
        with self.lock:
            self.queue = [item for item in self.queue if item["id"] != item_id]

    def move(self, item_id, step):
        # step -1 moves a question up the queue, +1 down
        with self.lock:
            index = next((i for i, item in enumerate(self.queue) if item["id"] == item_id), None)
            if index is not None and 0 <= index + step < len(self.queue):
                self.queue[index], self.queue[index + step] = self.queue[index + step], self.queue[index]

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused

    def clear_answers(self):
        with self.lock:
            self.messages = []

    def snapshot(self):
        # A copy for drawing the page, so the background thread can keep writing
        with self.lock:
            current = dict(self.current) if self.current else None
            return list(self.messages), current, list(self.queue), self.paused

    def answer_questions(self):
        while True:
            with self.lock:
                item = None if self.paused or not self.queue else self.queue.pop(0)
                if item:
                    self.current = {"question": item["text"], "answer": "", "sources": []}
            if item is None:
                time.sleep(0.5)
                continue
            try:
                prompt, sources = build_prompt(item["text"])
                with self.lock:
                    self.current["sources"] = sources
                for piece in stream_response(prompt):
                    with self.lock:
                        self.current["answer"] += piece
            except Exception as error:
                with self.lock:
                    self.current["answer"] += f"\n\n(Envie couldn't answer this one: {error})"
            with self.lock:
                self.messages.append(self.current)
                self.current = None

@st.cache_resource
def get_live_chat():
    # Created once per server, and shared by every browser
    return LiveChat()

@st.cache_resource
def get_presenter_key():
    # A new random key each time the app starts, shown only in the terminal
    key = secrets.token_urlsafe(8)
    print(f"\n  Presenter key: {key}\n  Add ?key={key} to the app's address to get the queue controls.\n",
          flush=True)
    return key

############################################
# The page
############################################

# Questions and answers are shown exactly as typed: characters that Markdown or HTML would treat
# as formatting are replaced by HTML character codes, so nobody can put links, images or HTML on
# everyone's screen
MARKDOWN_SPECIAL = set("\\`*_{}[]()#+-.!|~<>&\"'=:")

def plain(text):
    escaped = "".join(f"&#{ord(c)};" if c in MARKDOWN_SPECIAL else c for c in text)
    return escaped.replace("\n", "  \n")

def shorten(text, length=60):
    return text if len(text) <= length else text[:length - 1] + "…"

def show_sources(sources):
    # The files and lines Envie was given, listed by the app itself, so they're right even when the
    # model forgets to cite them. The expander shows the exact text of each.
    st.caption("From: " + " · ".join(plain(source["label"]) for source in sources))
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"**{plain(source['label'])}**")
            st.caption(source["text"])

@st.experimental_fragment(run_every=1)
def show_chat(live):
    messages, current, queue, paused = live.snapshot()
    for message in messages + ([current] if current else []):
        with st.chat_message("user"):
            st.markdown(plain(message["question"]))
        with st.chat_message("assistant"):
            if message is current:
                st.markdown(plain(message["answer"]) + "▌" if message["answer"] else "*Thinking…*")
            else:
                st.markdown(plain(message["answer"]))
            if message["sources"]:
                show_sources(message["sources"])
    if not messages and not current:
        st.caption("No questions answered yet. Ask one below!")
    elif not current and paused and queue:
        st.caption("Paused by the presenter.")

@st.experimental_fragment(run_every=1)
def show_queue(live, is_presenter):
    _, _, queue, paused = live.snapshot()
    st.subheader(f"Question queue ({len(queue)})")
    if is_presenter:
        pause_col, clear_col = st.columns(2)
        pause_col.button("▶ Resume" if paused else "⏸ Pause", on_click=live.toggle_pause,
                         help="Stop or restart answering. A question already being answered finishes first.")
        clear_col.button("🗑 Clear chat", on_click=live.clear_answers, help="Remove the answered questions from the chat")
    if not queue:
        st.caption("No questions waiting.")
    for position, item in enumerate(queue):
        # Hovering over a question shows all of it
        label = (f'<span title="{html.escape(item["text"], quote=True)}">'
                 f'{position + 1}. {plain(shorten(item["text"]))}</span>')
        if is_presenter:
            text_col, up_col, down_col, remove_col = st.columns([6, 1, 1, 1])
            text_col.markdown(label, unsafe_allow_html=True)
            up_col.button("↑", key=f"up-{item['id']}", on_click=live.move, args=(item["id"], -1),
                          disabled=position == 0, help="Move up")
            down_col.button("↓", key=f"down-{item['id']}", on_click=live.move, args=(item["id"], 1),
                            disabled=position == len(queue) - 1, help="Move down")
            remove_col.button("✕", key=f"remove-{item['id']}", on_click=live.remove, args=(item["id"],),
                              help="Remove from the queue")
        else:
            st.markdown(label, unsafe_allow_html=True)

live = get_live_chat()
presenter_key = get_presenter_key()
is_presenter = hmac.compare_digest(st.query_params.get("key", "").encode(), presenter_key.encode())

# Each browser gets its own ID, to limit how many questions one person can have waiting
if "client_id" not in st.session_state:
    st.session_state.client_id = uuid.uuid4().hex

with st.sidebar:
    st.subheader("Live Q&A")
    st.caption("Envie answers questions about this project's files, one at a time. Everyone sees the same chat.")
    st.caption(f"Model: {LLM_NAME}, served by Ollama")
    show_queue(live, is_presenter)

st.subheader("Ask Envie about this project!")
st.caption(f"Model: {LLM_NAME}, served by Ollama at {OLLAMA_URL} · Answers come only from this project's files")
show_chat(live)

question = st.chat_input(f"Ask Envie a question about this project (up to {MAX_QUESTION_CHARS} characters)")
if question:
    st.toast(live.add(question, st.session_state.client_id))

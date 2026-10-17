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
import re
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

@st.cache_resource
def ollama_start_lock():
    # Shared by every browser, so two page loads at once don't both start Ollama
    return threading.Lock()

def start_ollama():
    # In a codespace, Ollama doesn't start by itself after a restart, so start it if it's installed.
    # This runs on every page load, so it also works if Ollama was installed after the app started.
    if ollama_models() is not None or not shutil.which("ollama"):
        return
    with ollama_start_lock():
        if ollama_models() is None:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            for _ in range(20):
                time.sleep(0.5)
                if ollama_models() is not None:
                    break

start_ollama()
models = ollama_models()
if models is None and not shutil.which("ollama"):
    st.error("Ollama isn't installed here. In a codespace, run `bash .devcontainer/post-create.sh` in the "
             "terminal, which installs it and downloads Qwen; on your own computer, install it from ollama.com. "
             "Then reload this page.")
    st.stop()
if models is None:
    st.error(f"Ollama is installed but isn't answering at {OLLAMA_URL}. Start it (in a codespace: "
             "`nohup ollama serve >/tmp/ollama.log 2>&1 &`), then reload this page.")
    st.stop()
if LLM_NAME not in models:
    st.error(f"Ollama doesn't have {LLM_NAME} yet. Download it with `ollama pull {LLM_NAME}`, then reload this page.")
    st.stop()

@st.cache_resource(show_spinner="Indexing this project's files (once, when the app starts)...")
def build_vectorstore():
    # 4 chunks of up to 800 characters give Qwen enough to explain, while keeping the prompt short:
    # on a CPU, every extra word in the prompt adds waiting time. This splitter cuts at blank lines
    # first, then at line ends, so code is split into short chunks too.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, add_start_index=True)
    chunks = []
    file_chunks = {}   # each file's chunks, in order
    for path in project_files():
        document = TextLoader(path, encoding="utf-8").load()[0]
        file_chunks[path] = splitter.split_documents([document])
        for chunk in file_chunks[path]:
            # Turn the chunk's position in the file into line numbers, so answers can cite them
            first = document.page_content.count("\n", 0, chunk.metadata["start_index"]) + 1
            chunk.metadata["lines"] = (first, first + chunk.page_content.count("\n"))
            chunks.append(chunk)
    # Normalized embeddings make FAISS's relevance score a similarity from 0 to 1 (see build_prompt).
    # Kept in memory and built once for everyone, so no pickle file is written or loaded.
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_DIR, encode_kwargs={"normalize_embeddings": True})
    return FAISS.from_documents(chunks, embedder), file_chunks

vectorstore, file_chunks = build_vectorstore()
indexed_files = sorted(file_chunks)

############################################
# Answering a question (the same steps as main.py)
############################################

# Envie answers like a mentor: the style is adapted from the "Mentor mode" agent in
# github/awesome-copilot (agents/mentor.agent.md, MIT License), shortened for a small model, and for
# a live Q&A where people want an answer rather than only questions back.
SYSTEM_PROMPT = """You are Envie, a friendly mentor answering a class's questions about one software project, live during a presentation. The project shows how to run AI models locally and use them in VS Code: a RAG app, a local inference server, a VS Code extension, AI harnesses such as GitHub Copilot, Claude Code and Codex, MCP servers, agents and skills, and the safety rules for using them.

How to answer:
- Use only the context given with the question. Each part of it starts with its source in square brackets, e.g. [src/main.py, lines 100-107]. Cite the source of every fact you use, copying the whole bracket exactly, always including the file name, e.g. [src/main.py, lines 100-107], never just [100-107]. Never invent files, lines or facts.
- Be clear and concise: at most about 120 words. Explain why, not just what.
- Like a mentor, help people think: point out an assumption or a common mistake when there is one, and end with one short question that makes them think further.
- If something in the question is unsafe, say so plainly and explain why.
- If the context doesn't answer the question, say so in one sentence, and suggest a related question about the project.
- If the message is only a greeting, greet back in one sentence and invite a question about the project."""

def source_label(doc):
    # e.g. "src/main.py, lines 100-107"
    first, last = doc.metadata["lines"]
    lines = f"line {first}" if first == last else f"lines {first}-{last}"
    return f"{os.path.relpath(doc.metadata['source'], PROJECT_DIR).replace(os.sep, '/')}, {lines}"

# Chunks less similar to the question than this are left out. Measured on this project: greetings
# and off-topic messages ("hello?", "thanks!") score below 0.1, while real questions' best matches
# score 0.3 and up. Without a cut-off, "hello?" would get 4 unrelated chunks and an odd answer.
RELEVANCE_CUTOFF = 0.2

def build_prompt(question):
    # Retrieve the 4 most similar chunks, keeping only the ones that really match
    docs = [doc for doc, score in vectorstore.similarity_search_with_relevance_scores(question, k=4)
            if score >= RELEVANCE_CUTOFF]
    # A question that names a file, e.g. "What does inference_server.py do?", also gets that file's
    # opening chunk, where files describe themselves, and its best match: otherwise readmes that
    # mention the file can crowd out the file itself
    for path in indexed_files:
        if os.path.basename(path).lower() in question.lower():
            opening = [chunk for chunk in file_chunks[path] if chunk.metadata["lines"][0] == 1][:1]
            for doc in opening + vectorstore.similarity_search(question, k=1, filter={"source": path}):
                if doc not in docs:
                    docs.insert(0, doc)
    docs = docs[:5]
    if docs:
        context = "".join(f"[{source_label(doc)}]\n{doc.page_content}\n\n" for doc in docs)
    else:
        context = "(Nothing in the project matches this message.)\n\n"
    # The citation rule is repeated after the question: small models follow the last instruction best
    prompt = ("Context:\n\n" + context + "Question: " + question + "\n\n"
              "Answer as Envie. Cite each fact with its source in square brackets, copied exactly from the "
              "context, e.g. [src/main.py, lines 100-107]. Cite only sources given above.\n")
    # Keep exactly which chunks were given to the model, to show as sources and check citations
    sources = [{"label": source_label(doc), "text": doc.page_content,
                "file": source_label(doc).rsplit(", line", 1)[0], "lines": doc.metadata["lines"]}
               for doc in docs]
    return prompt, sources

def unverified_citations(answer, sources):
    # Citations in the answer, e.g. [src/main.py, lines 100-107], that don't fall inside a chunk the
    # model was actually given, and project files it mentions without having been given them. A small
    # model sometimes invents these; they're shown with a warning.
    unverified = []
    for match in re.finditer(r"\[([^\[\],]+),\s*lines?\s+(\d+)(?:\s*[-–]\s*(\d+))?\]", answer):
        file, first = match.group(1).strip(), int(match.group(2))
        last = int(match.group(3) or first)
        if not any(source["file"] == file and source["lines"][0] <= first and last <= source["lines"][1]
                   for source in sources):
            unverified.append(match.group(0))
    # A file name is fine if it's one of the sources, or mentioned in their text (e.g. "SKILL.md")
    given = {os.path.basename(source["file"]) for source in sources}
    given_text = "\n".join(source["text"] for source in sources)
    for path in indexed_files:
        name = os.path.basename(path)
        if (re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", answer)
                and name not in given and name not in given_text
                and not any(name in citation for citation in unverified)):
            unverified.append(name)
    return list(dict.fromkeys(unverified))

def stream_response(user_content):
    # The same OpenAI-style request the VS Code plugin sends; "stream" asks for the reply piece by
    # piece, as server-sent events ("data: {...}" lines)
    response = requests.post(f"{OLLAMA_URL}/v1/chat/completions", stream=True, timeout=600, json={
        "model": LLM_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
        "max_tokens": 300,   # about 120 words are asked for; this is a safety limit
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
    # In a codespace, the forwarded address is built from variables Codespaces sets
    port = st.get_option("server.port")
    if os.environ.get("CODESPACE_NAME") and os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN"):
        address = (f"https://{os.environ['CODESPACE_NAME']}-{port}."
                   f"{os.environ['GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN']}")
    else:
        address = f"http://localhost:{port}"
    print(f"\n  Presenter link (queue controls; don't share it): {address}/?key={key}"
          f"\n  Participant link (share this one):                {address}\n", flush=True)
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

@st.cache_resource
def github_files_url():
    # The address of this project's files on GitHub, at the current commit, e.g.
    # https://github.com/sprintup/rag/blob/<commit>. A commit rather than a branch keeps the line
    # numbers right after later pushes. None if this isn't a GitHub clone, or git isn't available.
    try:
        git = lambda *args: subprocess.run(["git", "-C", PROJECT_DIR, *args], capture_output=True,
                                           text=True, timeout=5).stdout.strip()
        remote, commit = git("config", "--get", "remote.origin.url"), git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.match(r"(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?/?$", remote)
    return f"https://github.com/{match.group(1)}/blob/{commit}" if match and commit else None

def github_link(text, file, first, last):
    # A link to those lines of the file on GitHub, opening in a new tab, or just the text without a
    # GitHub clone. The address is built here, never taken from the model's answer, and the text is
    # plain(), so this is the only HTML on the page (shown with unsafe_allow_html=True).
    base = github_files_url()
    if not base:
        return plain(text)
    lines = f"L{first}" if first == last else f"L{first}-L{last}"
    # GitHub shows Markdown files rendered, where line links don't work; ?plain=1 shows the text with
    # line numbers, so #L... scrolls to the lines and highlights them, as it does for code
    plain_view = "?plain=1" if file.lower().endswith(".md") else ""
    url = html.escape(f"{base}/{file}{plain_view}#{lines}", quote=True)
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{plain(text)}</a>'

def linked_answer(answer, sources):
    # The answer as plain text, with each citation that matches a source turned into a link to
    # those lines on GitHub. Citations that don't match stay plain, and are warned about below.
    shown = plain(answer)
    for match in re.finditer(r"\[([^\[\],]+),\s*lines?\s+(\d+)(?:\s*[-–]\s*(\d+))?\]", answer):
        file, first = match.group(1).strip(), int(match.group(2))
        last = int(match.group(3) or first)
        if any(source["file"] == file and source["lines"][0] <= first and last <= source["lines"][1]
               for source in sources):
            shown = shown.replace(plain(match.group(0)), github_link(match.group(0), file, first, last))
    return shown

PREVIEW_LINES = 3   # lines of each source shown in the Sources dropdown; the link opens the rest

def preview(text):
    # The first few non-empty lines of a source, so opening Sources doesn't fill the screen
    lines = [line for line in text.splitlines() if line.strip()]
    shown = "\n".join(lines[:PREVIEW_LINES])
    return shown + " …" if len(lines) > PREVIEW_LINES else shown

def show_sources(sources):
    # The files and lines Envie was given, listed by the app itself, so they're right even when the
    # model forgets to cite them, and linked to GitHub. The expander shows the exact text of each.
    st.caption("From: " + " · ".join(github_link(source["label"], source["file"], *source["lines"])
                                     for source in sources), unsafe_allow_html=True)
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"**{github_link(source['label'], source['file'], *source['lines'])}**",
                        unsafe_allow_html=True)
            st.caption(plain(preview(source["text"])))

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
                st.markdown(linked_answer(message["answer"], message["sources"]), unsafe_allow_html=True)
                unverified = unverified_citations(message["answer"], message["sources"])
                if unverified:
                    st.caption("⚠ Not in the sources Envie was given, so possibly made up: "
                               + " · ".join(plain(citation) for citation in unverified))
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
st.caption(f"Model: {LLM_NAME}, served by Ollama on the computer running this app · Answers come only from "
           "this project's files, and their citations link to them on GitHub")
show_chat(live)

question = st.chat_input(f"Ask Envie a question about this project (up to {MAX_QUESTION_CHARS} characters)")
if question:
    st.toast(live.add(question, st.session_state.client_id))

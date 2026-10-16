# Envie – Local RAG Chatbot User Guide

Envie is a chatbot that answers questions about documents you give it, such as a PDF of
lecture slides. It uses a technique called **RAG (Retrieval-Augmented Generation)**: before
answering, it *retrieves* the most relevant parts of your documents, then *generates* an answer
using only those parts.

Everything runs **on your own computer**. There is no API key, no cloud AI service, and once
the models are downloaded, it works without the internet.

| Part | What it is | Where it lives |
|---|---|---|
| App / web page | [Streamlit](https://streamlit.io), a Python library that turns a script into a web page | [src/main.py](src/main.py) |
| Chat model (LLM) | TinyLlama 1.1B Chat, a small open-source language model | [models/TinyLlama/](models/TinyLlama/), the weights are in `model.safetensors` |
| Embedding model | `all-MiniLM-L6-v2`, which turns text into numbers so it can be searched by meaning | [models/all-MiniLM-L6-v2/](models/all-MiniLM-L6-v2/) |
| Vector store | FAISS, a fast index for searching those numbers | [vectorstore/](vectorstore/) |
| Your documents | Files you upload | [uploaded_docs/](uploaded_docs/) |

> First time on this computer? Follow [01_readme-setup.md](01_readme-setup.md) to install the app and download the models.

---

## 1. Running the app

```powershell
.\venv_rag\Scripts\Activate.ps1
streamlit run src/main.py
```

Your browser opens <http://localhost:8501>. The first load takes 10–30 seconds while the
models load into memory.

**In GitHub Codespaces** (see [readme-dev-container-startup.md](readme-dev-container-startup.md)), skip the activate line and just
run `streamlit run src/main.py` in the codespace terminal. The app opens in a new browser tab; if
it doesn't, copy the codespace's URL into a new tab and replace `.github.dev` with
`-8501.app.github.dev` (details in [readme-dev-container-startup.md](readme-dev-container-startup.md)).

- **Upload:** in the sidebar, choose a file (PDF, TXT, DOCX, etc.) and click **Upload!**
- **Documents in the Knowledge Base:** lists your files. Click **✕** to remove one.
- **Use existing vector store if available:** leave this on **Yes**. Uploading or removing a
  file rebuilds the store automatically.
- **Chat:** type a question in the box at the bottom.

---

## 2. How RAG works: the big picture

A language model on its own only knows what it learned during training. It has never seen
*your* lecture slides. RAG gets around this without retraining the model: it finds the relevant
parts of your documents, then pastes them into the question it sends to the model.

### The outline

RAG works in two phases.

**Phase A – Indexing (happens when you upload a document)**

1. **Load:** read the text out of the document.
2. **Chunk:** cut the text into small pieces.
3. **Embed:** turn each piece into a list of numbers that captures its meaning.
4. **Store:** save those numbers in a vector store so they can be searched quickly.

**Phase B – Answering (happens every time you ask a question)**

5. **Embed the question** the same way.
6. **Retrieve:** find the stored pieces whose numbers are closest to the question's.
7. **Augment:** build a prompt that contains those pieces plus the question.
8. **Generate:** the language model writes an answer using that prompt.
9. **Cite:** show which pieces were used.

### Key terms

| Term | Meaning | In this app |
|---|---|---|
| **Chunk** | A small piece of a document, a few sentences long. Small pieces make search more precise and fit within what the model can read at once. | ~500 characters each |
| **Token** | The unit a language model reads and writes: a word or part of a word. "Stacks are LIFO" is about 6 tokens. | TinyLlama can read ~2,048 tokens at once |
| **Vector** | Simply an ordered list of numbers, such as `[0.12, -0.48, 0.03]`. You can think of it as a point in space: 2 numbers is a point on a map, 3 is a point in a room, 384 is a point in a 384-dimensional "meaning space". | 384 numbers per chunk |
| **Embedding** | A vector produced by an *embedding model* to represent the **meaning** of some text. Texts with similar meanings get vectors that are close together, even if they use different words. "stack" and "LIFO" land near each other; "stack" and "pizza" don't. | Made by `all-MiniLM-L6-v2` |
| **Vector store** | A database built for vectors. Given a new vector, it quickly finds the closest stored ones. This is called *similarity search*. | FAISS, saved in `vectorstore/` |
| **LLM** | Large Language Model: a model that generates text by predicting the next token, over and over. | TinyLlama 1.1B, on your CPU |
| **Prompt** | Everything sent to the LLM in one go: instructions, context and question. | System prompt + chunks + your question |
| **System prompt** | Hidden instructions that set the model's behaviour. | "Answer only from the context, otherwise decline" |

### A worked example

Suppose you upload the *Abstract & Concrete Data Types* lecture PDF and ask about stacks.

**When you upload the PDF (Phase A)**

1. You click **Upload!** The Streamlit app saves the file to `uploaded_docs/`.
2. The app reads all 102 pages. Page 22 contains text like:
   *"Stack: push onto the top, pop also from the top. 'Last in, first out' (LIFO). The inverse of a
   Queue (if there is such a thing) is a Stack…"*
3. That text becomes one or more chunks of about 500 characters each. Each chunk remembers it
   came from page 22.
4. The embedding model turns each chunk into a vector, for example
   `[0.021, -0.113, 0.087, … 384 numbers in total]`. These values are illustrative.
5. FAISS stores every chunk's vector, and the whole index is saved to `vectorstore/`.

**When you ask a question (Phase B)**

1. You type: *"What is the inverse of a queue?"*
2. The same embedding model turns your question into its own 384-number vector.
3. FAISS compares it with every stored vector and returns the **3 closest chunks**. The page 22
   chunk is close because it talks about queues, stacks and inverses.
4. The app builds the prompt:
   ```
   [System] You are a helpful AI assistant named Envie. You will reply to questions only
            based on the context that you are provided...
   [User]   Context: Stack push onto the top pop also from the top "Last in, first out"...
            (plus the other 2 chunks)
            Question: What is the inverse of a queue?
   ```
5. TinyLlama reads the prompt and generates an answer one token at a time, entirely on your CPU,
   e.g. *"The inverse of a queue is a stack, which works on a last in, first out (LIFO) basis."*
   The answer streams onto the page as it is generated.
6. Under the answer, **Sources** lists the 3 chunks, e.g.
   *"Abstract and Concrete Data Types (with notes).pdf, page 22"*, so you can check the answer
   against the slide.

If you ask something not in the slides, like *"What is the capital of France?"*, the retrieved
chunks won't contain the answer, and the system prompt tells TinyLlama to decline.

---

## 3. How it works, step by step in the code

This section follows the same steps through the code. Open [src/main.py](src/main.py) alongside it; the
file is divided into four labelled "Components".

### Step 1 – Saving your documents (Component #1)
When you click **Upload!**, each file is written into the `uploaded_docs/` folder
([src/main.py:52-57](src/main.py#L52-L57)). The old vector store is then deleted with
`clear_vector_store()` ([src/main.py:39-41](src/main.py#L39-L41)), so it gets rebuilt with the new
file. The ✕ buttons work the same way, but delete the file instead
([src/main.py:63-69](src/main.py#L63-L69)).

### Step 2 – Loading the models (Component #2)
- `load_llm()` ([src/main.py:100-107](src/main.py#L100-L107)) loads TinyLlama from the `models/TinyLlama/` folder. The
  **tokenizer** converts text into the number IDs ("tokens") the model works with, and back again.
- `load_embedder()` ([src/main.py:109-111](src/main.py#L109-L111)) loads the embedding model.
- `@st.cache_resource` means the models load **once** and are reused for every question.
  Without it, Streamlit would reload them every time you click something.

### Step 3 – Reading and chunking the documents (Component #3)
- PDFs are read page by page with `PyPDFLoader`. Other file types use `unstructured`
  ([src/main.py:133-135](src/main.py#L133-L135)). Each page keeps a note of its file name and page
  number, which is what makes citations possible later.
- The text is cut into **chunks** of about 500 characters
  ([src/main.py:150-151](src/main.py#L150-L151)). Small chunks matter because TinyLlama can only read
  about 2,048 tokens (roughly 1,500 words) at once.

### Step 4 – Building the vector store (Component #3)
- Every chunk is passed through the embedding model, which turns it into a list of 384
  numbers called an **embedding** (or vector). Chunks with similar meanings get similar numbers.
- FAISS stores all the embeddings in an index ([src/main.py:154](src/main.py#L154)), which is saved to
  the `vectorstore/` folder ([src/main.py:157](src/main.py#L157)). Next time the app starts, it loads
  that folder instead of redoing the work ([src/main.py:141-142](src/main.py#L141-L142)).

**What's in `vectorstore/`, and what's a pickle file?** The folder holds two files:
- **`index.faiss`**: the vectors themselves, in FAISS's own format.
- **`index.pkl`**: everything else needed to use them, such as each chunk's text, its source file
  and page number (used by **Sources**), and how the chunks map to the vectors. It's saved as a
  **pickle** file.

**Pickle** is Python's built-in way to save Python objects to a file and load them back later,
like saving a game. It's convenient, but loading a pickle can **run any code** that was put into
the file. A pickle file from someone else is as dangerous as running their program. That's why
`src/main.py` has to opt in explicitly with `allow_dangerous_deserialization=True` when it loads the
folder ([src/main.py:142](src/main.py#L142)). Here, it's safe because the app only loads the index
it built itself. ⚠ **Never load a `vectorstore/` folder you didn't create.** If you're unsure where
one came from, delete it: the app rebuilds it from `uploaded_docs/`. See
[readme-safety.md](readme-safety.md), section 2.

### Step 5 – Retrieving relevant chunks (Component #4)
When you ask a question, the question is also turned into an embedding. FAISS finds the
**3 chunks** whose embeddings are closest to it ([src/main.py:217-218](src/main.py#L217-L218)). This is
the "retrieval" in RAG.

### Step 6 – Building the prompt (Component #4)
The 3 chunks are pasted together with your question into one message:
`Context: <chunks>  Question: <your question>` ([src/main.py:222-226](src/main.py#L222-L226)).
A **system prompt** ([src/main.py:191](src/main.py#L191)) tells the model to answer only from that
context and to decline otherwise.

### Step 7 – Generating the answer (Component #4)
`stream_response()` ([src/main.py:193-208](src/main.py#L193-L208)) formats the messages the way
TinyLlama expects, then runs `model.generate` on your CPU. The model predicts the answer
one token at a time, up to 256 tokens. A `TextIteratorStreamer` hands each new piece of text
to the page as it is produced, which is why the answer appears word by word
([src/main.py:232-235](src/main.py#L232-L235)).

**Worth pointing out in a demo:** small models often don't know when to stop. After answering,
TinyLlama may carry on and invent the next turns of the conversation ("User: … Envie: …")
until it reaches the 256-token limit. This shows that an LLM is only predicting likely next
text; it has no real sense of a conversation ending.

### Step 8 – Showing the sources (Component #4)
After the answer, the same 3 chunks from Step 5 are listed under **Sources**
([src/main.py:237-239](src/main.py#L237-L239)), labelled with file name and page number by
`source_label()` ([src/main.py:173-177](src/main.py#L173-L177)). These are exactly the text the model
was given, so you can check whether the answer really came from the document.

---

## 4. Demo: proving the model runs locally

You can do this live, even while presenting over Zoom.

**How the app stays offline.** At the top of [src/main.py](src/main.py#L27-L31), two settings are
applied before any AI library loads:
- `HF_HUB_OFFLINE=1` makes the libraries use the downloaded models without checking
  huggingface.co for updates.
- `DO_NOT_TRACK=true` turns off the usage analytics that `unstructured` would otherwise send.

**Steps**
1. Start the app as in section 1. If it was already running, stop it (`Ctrl+C`) and start it
   again so the settings take effect.
2. Note the **Process ID** under the heading, e.g.
   `Process ID: 33212 · Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 on cpu · Location: C:\...
ag\models\TinyLlama`
   ([src/main.py:168](src/main.py#L168)). All model work happens inside this one process, using the
   model files in that folder on your disk.
3. Press `Win+R`, type `resmon` and press Enter. Open the **Network** tab.
4. In **Processes with Network Activity**, click the **PID** column header to sort, then tick
   the `python3.10.exe` row with your Process ID. If it isn't listed at all, it has no network
   activity, which proves the point.
5. Expand **TCP Connections** and ask Envie a question. Every row should say
   **IPv4 loopback** or **IPv6 loopback**. Loopback means "this same computer": it's your browser
   talking to the app on port 8501, and it never leaves your machine. Point out that
   apps like Zoom or Outlook in the list above *do* show outside addresses.
6. Optional: open **Task Manager → Performance** to show the CPU working hard while the
   answer is generated. The computation is happening on your machine.

**Double-check from PowerShell** (replace `<PID>`). If this prints nothing, the app has no
connections outside your computer:

```powershell
Get-NetTCPConnection -OwningProcess <PID> |
  Where-Object { $_.RemoteAddress -notin '127.0.0.1','::1','0.0.0.0','::' }
```

If you see a `13.226.x.x` address, that's huggingface.co. It means the app was started
before the offline setting was in place; restart it.

**In GitHub Codespaces** there is no Resource Monitor, but the same check works from a second
terminal (click **+** in the terminal panel). Replace `<PID>` with the Process ID from the app:

```bash
ss -tnp | grep "pid=<PID>,"     # every connection should be to 127.0.0.1 (this machine)
top -p <PID>                    # CPU usage jumps while an answer is generated
```

Here, "local" means the model runs inside the codespace's own machine, not on a hosted AI
service. The **Location** under the heading shows the model folder,
`/workspaces/rag/models/TinyLlama`.

## 5. Demo: showing citations from the PDF

1. Upload a PDF, for example your lecture slides, and wait for **"Vector store created and saved."**
2. Ask a question that uses the same wording as the slides. For the *Abstract & Concrete
   Data Types* slides, for example: *"What is the inverse of a queue?"*
3. When the answer finishes, click **Sources** under it. Each entry shows the document,
   **page number**, and the exact text given to the model, e.g.
   *"Abstract and Concrete Data Types (with notes).pdf, page 22"*.
4. Open the PDF at that page to show the class the text matches.
5. Ask something that isn't in the document (e.g. *"What is the capital of France?"*).
   Envie should decline, which shows it only answers from your documents.

**Rehearse your questions first.** TinyLlama is small, so answers can be basic or
inconsistent. Retrieval matches by *meaning*, not exact words, so short lookups like a
person's name often find the wrong pages. Questions phrased like the document work best.
**Sources** always shows what the model was actually given, even when the answer is weak.

---

## 6. Other files

| File / folder | Purpose |
|---|---|
| [src/](src/) | All the Python code: [main.py](src/main.py) is the app; [small_model.py](src/small_model.py) is a terminal-only TinyLlama test (`python src/small_model.py`). |
| [requirements.txt](requirements.txt) | Libraries to install, with exact versions. |
| `models/` | The downloaded models: the files the app runs. Not committed to git (too large). |
| `vectorstore/` | Saved search index: `index.faiss` (vectors) and `index.pkl` (a pickle file with the chunks' text and sources). Safe to delete; it is rebuilt from `uploaded_docs/`. ⚠ Never load one you didn't create (see Step 4). |
| `venv_rag/` | The Python environment (see [01_readme-setup.md](01_readme-setup.md)). |
| [.devcontainer/](.devcontainer/) | Dev container config for running in GitHub Codespaces; `post-create.sh` installs everything and downloads the models. |
| [03_readme-plugin-startup.md](03_readme-plugin-startup.md) | The TinyLlama VS Code plugin's own chat panel (`Ctrl+Shift+P` → TinyLlama: Open Chat) and tab: its files, how it works, and how it could be published. |
| [04_readme-harness.md](04_readme-harness.md) | Opening the Extension Development Host, and using `@tinyllama` and skills (e.g. `/pirate`) in Copilot's Chat view. |
| [05_readme-models.md](05_readme-models.md) | Replacing TinyLlama with Qwen 2.5, using Ollama, Foundry Local or this project's own server, including Copilot Agent mode. |
| [06_readme-mcp-agents.md](06_readme-mcp-agents.md) | MCP and custom agents: the Fetch server ([.vscode/mcp.json](.vscode/mcp.json)) and the Idea Generator agent ([.github/agents/](.github/agents/)), used together to design an activity for kids' word of the day. |
| [readme-more-info.md](readme-more-info.md) | Inspecting the hidden context VS Code sends to the model (Chat Debug View, agent logs, Ollama's debug log). |
| [readme-model-training.md](readme-model-training.md) | How to fine-tune your own model (LoRA/QLoRA on a small Qwen) and publish it on Hugging Face, then use it here or in Ollama. |
| [readme-dev-container-startup.md](readme-dev-container-startup.md) | Running and demoing the app inside the dev container / Codespaces, with `bash` commands. |
| [01_readme-setup.md](01_readme-setup.md) | Installation, downloading models from Hugging Face, and what changed from the original. |
| [readme.md](readme.md) | Project background: the original NVIDIA source code and research links. |

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `cannot import name 'AutoTokenizer' from 'transformers'` just after starting | The browser reconnected while the app was still loading. Refresh the page (`F5`). |
| "Model not found in …models\TinyLlama" | The models aren't downloaded yet. Run the `hf download` command shown on the page, or see [01_readme-setup.md](01_readme-setup.md). |
| `Activate.ps1` cannot be loaded / is blocked | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| "No documents available to process!" | Upload a file first. |
| Answers ignore a new or removed file | Make sure the radio is on **Yes**, or delete the `vectorstore/` folder to force a rebuild. |
| Very slow answers | Normal on a CPU. Close other heavy apps. |

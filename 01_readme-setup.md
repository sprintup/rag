# Envie – Setup

This file covers installing the app (on Windows or in GitHub Codespaces), downloading models from Hugging Face, and how the project
changed from the original NVIDIA example. For how to run and demo the app, see
[02_readme-rag-user-guide.md](02_readme-rag-user-guide.md). For the original source and background links, see
[readme.md](readme.md).

## Setup (one time, Windows PowerShell)

Requires Python 3.10–3.12. About 5 GB of disk space is needed for the libraries and models.

**1. Create the Python environment and install the libraries** from the project folder:

```powershell
python -m venv venv_rag                 # create an isolated Python environment in venv_rag/
.\venv_rag\Scripts\Activate.ps1         # turn it on (your prompt will start with (venv_rag))
pip install -r requirements.txt         # install the libraries listed in requirements.txt
```

If PowerShell says `Activate.ps1` cannot be loaded, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**2. Download the two models into the `models/` folder** (about 2.3 GB, once per computer).
With the environment still activated, from the project folder:

```powershell
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/TinyLlama
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
```

`hf` is the Hugging Face command-line tool. It was installed by step 1 (it comes with the
`huggingface-hub` library in `requirements.txt`), and it only works while the environment is
activated. If you get "hf is not recognized", run `.\venv_rag\Scripts\Activate.ps1` first.

This creates `models/TinyLlama/` and `models/all-MiniLM-L6-v2/`. The next section explains
these commands, how to download in the browser instead, and what the files are.

**3. Run the app** by following [02_readme-rag-user-guide.md](02_readme-rag-user-guide.md).

## Alternative setup: GitHub Codespaces (or any dev container)

Instead of installing on Windows, you can run the app in a **dev container**: a ready-made Linux
environment described by [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json).
**GitHub Codespaces** builds that container on a cloud machine and opens it in VS Code in your
browser, so nothing needs installing on your computer.

**What the dev container does for you**
- Uses a Python 3.12 Linux image.
- Runs [.devcontainer/post-create.sh](.devcontainer/post-create.sh) once, which:
  - installs `requirements.txt`, using the CPU-only build of PyTorch (Codespaces has no GPU);
  - downloads both models into `models/` with `hf download`.
- Forwards port 8501, so the app opens in a browser tab when it starts.
- Turns off Streamlit's XSRF check. Otherwise, file uploads fail with "403" when they go
  through the Codespaces proxy.

**Starting a codespace**
1. Commit and push your changes to GitHub. Codespaces builds from what's on GitHub, not your
   local folder.
2. On the repository page (<https://github.com/sprintup/rag>), click **Code → Codespaces →
   Create codespace on `<your branch>`**. For faster answers, use **… → New with options** and
   pick a 4-core machine.
3. Wait for setup to finish. The first build takes about 5–10 minutes, mostly downloading
   PyTorch and the models. You can watch progress with **View creation log**. It ends with
   "Setup complete".
4. In the codespace terminal, run:
   ```bash
   streamlit run src/main.py
   ```
   Don't use the URLs Streamlit prints. Copy the codespace's URL from the address bar into a new
   tab, and replace `.github.dev` with `-8501.app.github.dev`, e.g.
   `https://crispy-tribble-ww94rrr56gh996-8501.app.github.dev/`.
5. From here on, follow [readme-dev-container-startup.md](readme-dev-container-startup.md): the
   same instructions as the user guide, with `bash` commands instead of PowerShell.

**Things to know**
- **No venv needed:** the container is already isolated, so skip the `venv_rag` steps.
- **Uploads don't carry over:** `uploaded_docs/`, `vectorstore/` and `models/` are not in git, so
  a new codespace starts empty. Upload your PDF again.
- **"Local" means inside the codespace:** the model runs on the codespace's cloud machine,
  not on a hosted AI service. The Resource Monitor demo in the user guide is Windows-only; see
  its Codespaces note instead.
- **Stop it when you're done:** codespaces use your free monthly hours while running. Stop or
  delete them at <https://github.com/codespaces>.
- **Running locally:** the same container works on your own computer with Docker Desktop and the
  VS Code **Dev Containers** extension. Open the folder and choose **Reopen in Container**.

## Downloading models from Hugging Face

[Hugging Face](https://huggingface.co/models) is a website that hosts open-source AI models,
a bit like GitHub for models. Each model has an ID made of `owner/name`, e.g.
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, and is really a folder of files. This app uses two models,
and loads both from the project's `models/` folder:

| Model ID | Role | Size | Project folder | Set in |
|---|---|---|---|---|
| [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Chat model (LLM) | ~2.2 GB | `models/TinyLlama/` | [src/main.py:83](src/main.py#L83) (`LLM_DIR`) |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Embedding model | ~90 MB | `models/all-MiniLM-L6-v2/` | [src/main.py:85](src/main.py#L85) (`EMBEDDING_DIR`) |

`models/` is listed in `.gitignore`, so the large files are never committed. If a folder is
missing when the app starts, the app shows the command to download it.

### Downloading the model yourself vs. using Ollama

This project uses **both** ways of getting a model. TinyLlama is downloaded as `model.safetensors`
and run by the project's own Python code. Qwen runs in **Ollama**
([05_readme-models.md](05_readme-models.md)). The difference:

| | Downloading `model.safetensors` yourself | Using Ollama |
|---|---|---|
| **Command** | `hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/TinyLlama` | `ollama pull qwen3.5:4b` |
| **Where from** | Hugging Face: almost any open model, by its `owner/name` ID | Ollama's model library (`ollama.com/library`), **or Hugging Face**: any repository with GGUF files, e.g. `ollama pull hf.co/<owner>/<repository>` (optionally with a quantization tag such as `:Q4_K_M`). It can also import model folders (`ollama create`). |
| **What you get** | The **original weights**, as published (TinyLlama: 16-bit numbers, 2.1 GB) | A **compressed (quantized)** copy, e.g. 4-bit `Q4_K_M` (`qwen3.5:4b`: 3.3 GB), in Ollama's own format (GGUF) |
| **Where it's stored** | Wherever you choose: here, the project's `models/` folder | Ollama's folder, `%USERPROFILE%\.ollama\models` |
| **What runs it** | **Your own code**: Python with `torch` and `transformers` (`src/main.py`, `src/inference_server.py`), which loads the weights and tokenizer and applies the chat format | **Ollama itself** (built on the llama.cpp engine), which handles the tokenizer and chat format. No code needed. |
| **GPU** | Only if you install the CUDA build of PyTorch and your code uses it | Automatic |
| **Serving it to other programs** | You write a server yourself (that's `src/inference_server.py`) | Built in: an OpenAI-compatible API at `http://localhost:11434/v1` |
| **Embeddings** (turning text into vectors, for RAG search) | A separate model you download and run yourself (`models/all-MiniLM-L6-v2`, loaded with `sentence-transformers` in `src/main.py`) | Pulled and served the same way as chat models, e.g. `ollama pull nomic-embed-text`, through the same API |
| **Memory** | Loaded while your program runs | Loaded when needed, unloaded after 5 minutes idle |
| **Control** | Full: every line of code, every setting, fine-tuning ([readme-model-training.md](readme-model-training.md)), inspecting the model (`config.json`) | Less: a few settings (e.g. `OLLAMA_CONTEXT_LENGTH`), and a `Modelfile` for custom versions |
| **Supply chain: who you trust** | The model's **publisher** on Hugging Face (e.g. the TinyLlama team), and the Python libraries you install | The model's **original maker**, plus **Ollama** (Ollama Inc., an independent company), which picks the models in its library, converts and compresses them, and makes the software that runs them. With `hf.co/…` pulls, also whoever uploaded that GGUF file. |
| **As a provider for VS Code** | Your own server has to keep the API working as your code changes | A **stable provider**: VS Code's Ollama extension talks to the same API and model names, whatever happens underneath |
| **Best for** | Learning how it works, the RAG app, fine-tuning | Quickly running good models on the GPU, and connecting them to Copilot |

**Both can use Hugging Face.** The difference isn't *where* the model comes from, but *what
you do with it*. With `hf download`, you get the files and run them with your own code. With
`ollama pull hf.co/…`, Ollama downloads a ready-to-run GGUF version and does the rest.

**The model is the same idea either way:** a file of learned numbers. Ollama's copy is smaller
because each number is stored with fewer bits. That costs a little quality, and makes it faster and
lighter. Quantization is also why `qwen3.5:4b` (4.2 billion parameters) takes only 3.3 GB, while
TinyLlama's original file is 2.1 GB for 1.1 billion.

**Supply chain.** Either way, you're running someone else's work, so the same care applies. With
Ollama, most of that trust moves to the **runtime provider**: Ollama chooses, converts and serves
the models in its library. **Foundry Local** works the same way, with **Microsoft** as the
provider and a catalog of models it has prepared ([05_readme-models.md](05_readme-models.md),
option B). Ollama is made by **Ollama Inc.**, not by Meta, which makes the *Llama* models. In all
cases, check who published a model and what it was trained on (see the bias example in
[05_readme-models.md](05_readme-models.md)).

### Option A – Download from the command line (recommended)

With the environment activated (setup step 1), run these from the project folder:

```powershell
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/TinyLlama
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
```

`hf` is the Hugging Face command-line tool, installed with the other libraries. `--local-dir`
tells it which folder to save into. It downloads every file in the model, shows a progress bar,
and resumes if you run it again after an interruption.

### Option B – Download in the browser

There is no single "download" button for a whole model. Instead:

1. Open the model's page, e.g. <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>,
   and click the **Files and versions** tab.
2. Click the small **↓** icon next to each file you need. For TinyLlama, those are:
   `config.json`, `generation_config.json`, `model.safetensors` (the 2.2 GB file),
   `special_tokens_map.json`, `tokenizer.json`, `tokenizer.model` and `tokenizer_config.json`.
3. Move the downloaded files from your Downloads folder into `models/TinyLlama/`.

The embedding model has more files, including a `1_Pooling/` subfolder, so use Option A for it.

### What's inside a model folder

| File | What it is |
|---|---|
| `model.safetensors` | **The model itself:** its learned numbers ("weights"). For TinyLlama, 1.1 billion of them. |
| `config.json` | The model's architecture: number of layers, sizes, etc. |
| `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` | How text is split into tokens and turned into numbers, and back. |
| `generation_config.json`, `special_tokens_map.json` | Small default settings. |

### Trying a different model

1. Find a model on <https://huggingface.co/models>.
   - **Chat model:** filter by *Text Generation* and pick one whose name ends in `-Chat` or
     `-Instruct`. The app relies on the model's built-in chat format. On a CPU, stay under about
     2 billion parameters (the "1.1B" in TinyLlama's name), or answers get very slow. Good options
     to try are `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`.
   - **Embedding model:** filter by *Sentence Similarity*.
2. Download it into its own folder, e.g.
   `hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B`.
3. In [src/main.py](src/main.py#L82-L85), point `LLM_DIR` (or `EMBEDDING_DIR`) at the new folder,
   and update `LLM_NAME` (or `EMBEDDING_NAME`) so the app displays the right name.
4. **If you changed the embedding model, delete the `vectorstore/` folder.** Vectors from
   different embedding models aren't compatible, so the store must be rebuilt.
5. Restart the app. The new model name appears under the chat heading.

**Gated models.** Some models, like Meta's Llama, require you to accept a license on the
model's web page first. Log in to huggingface.co, accept the terms on the model page, create an
access token under *Settings → Access Tokens*, and run `hf auth login` once to paste it in.
Then `hf download` works as normal.

## What changed from the original

The original is NVIDIA's "5 minute RAG" example, linked in [readme.md](readme.md). It was kept
as a single file with the same four "Components", so [src/main.py](src/main.py) reads in the same
order. The unmodified version is in commit `04a6137`: `git show 04a6137:main.py`.

| Area | Original (NVIDIA example) | Now |
|---|---|---|
| Chat model | `mixtral_8x7b` via NVIDIA AI Foundation endpoints (cloud) | TinyLlama 1.1B Chat, run locally with Hugging Face `transformers` |
| Embeddings | `nvolveqa_40k` via NVIDIA endpoints (cloud) | `sentence-transformers/all-MiniLM-L6-v2`, run locally |
| API key | `NVIDIA_API_KEY` required | None |
| Internet | Needed for every question | Only to download the models once, into `models/`; offline after that (`HF_HUB_OFFLINE=1`, `DO_NOT_TRACK=true` set in code) |
| Response streaming | LangChain chain (`prompt \| llm \| StrOutputParser`) | `model.generate` in a background thread with `TextIteratorStreamer` |
| Chunk size | 2,000 characters, 200 overlap | 500 characters, 50 overlap, to fit TinyLlama's 2,048-token context |
| Chunks retrieved | 4 (default) | 3 |
| PDF loading | `unstructured` (needs Poppler/Tesseract on Windows) | `PyPDFLoader` (`pypdf`); other file types still use `unstructured` |
| Vector store file | `vectorstore.pkl` (Python pickle of the whole object) | `vectorstore/` folder via FAISS `save_local` / `load_local` |
| Document management | Upload only | Document list in the sidebar with ✕ to remove; the store rebuilds automatically on upload or removal |
| Citations | None | **Sources** section under each answer: file, page number and exact chunk text |
| Local-inference proof | None | Process ID and model/device shown under the heading, for checking in Resource Monitor |
| Layout | `main.py` at the project root | Code in `src/`; documentation (`readme*.md`) at the root; `uploaded_docs/` and `vectorstore/` stay at the root |
| Dependencies | `langchain-nvidia-ai-endpoints` | Removed; added `torch`, `transformers`, `sentence-transformers` |

## Things that were tried and removed

- **Llama 3 8B, run locally.** About 30 GB of weights in the `models/` folder, plus a
  `test_llama3.py` script to load them. Removed: too large for a CPU-only laptop demo.
- **`small_model.py`** (kept, in `src/`). A terminal-only script that loads TinyLlama and answers one
  hard-coded prompt. It was the first local-model test, and its approach was folded into `src/main.py`.
- **The original Dev Container.** The project was first developed in a Linux container built from
  the root `Dockerfile` (with its own `venv_rag_local/`). It was replaced by the new
  [.devcontainer/](.devcontainer/) setup for Codespaces, which uses a standard image directly.
  The root `Dockerfile` is no longer used.
- **Databricks course material** (`assets/`). Notebooks from the edX/Databricks
  *Large Language Models: Application through Production* course, which only run in Databricks.

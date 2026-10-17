# Live Q&A with Envie

[src/live_qa.py](src/live_qa.py) lets a class ask Envie questions about **this project** during a
presentation. Everyone who opens it sees **one shared chat**. Questions wait in a **queue** in the
sidebar, and Envie answers them one at a time while everyone watches the answers appear.

It's a separate app from the normal Envie app ([src/main.py](src/main.py)), which stays unchanged
for uploading your own documents. The model is different too: the live Q&A uses **Qwen 3.5**
(`qwen3.5:4b`), served by **Ollama**, which gives much better answers than TinyLlama. The search
part is the same: the `all-MiniLM-L6-v2` embedding model and a FAISS index.

---

## What it does

- **Guides people through the learning objectives.** Envie's main purpose is to bring the class
  through the six steps in [00_readme-learning-objectives.md](00_readme-learning-objectives.md). It
  says which step an answer belongs to, and when someone seems lost (*"I'm lost"*, *"Where do I
  start?"*) or a question is too vague, it shows the steps from the objectives' summary table and
  asks which step or learning objective they're working on.
- **Answers from this project's files.** When it starts, it reads every `.md`, `.py`, `.js`,
  `.json`, `.sh`, `.yml` and `.yaml` file in the project folder and its subfolders, e.g. all the
  readmes, the skills, agents and instructions in `.github/`, the Python code in `src/`, the VS Code
  extension's `extension.js` and `package.json`, and the dev container's `devcontainer.json` and
  `post-create.sh`, so it can explain how the codespace is set up and run. Files that don't say
  what they're for, such as `package.json` or `post-create.sh`, are labelled with a short
  description. New or changed files are picked up the next time the app
  starts; there's nothing to copy or upload. A small JSON file such as `package.json` is given to
  Envie whole, since its parts only make sense together (e.g. the settings are far below the
  extension's name).
- **Skips what isn't the project:** virtual environments (`venv_rag/`), `models/`, `vectorstore/`,
  `uploaded_docs/`, caches, and hidden folders other than `.github/` and `.devcontainer/` (so
  `.vscode/` isn't read). Only the file types above are read, so a `.env` or other secret file is
  never included.
- **No uploads, no pickle.** The search index is built in memory once, when the first page loads,
  and nothing is saved to `vectorstore/`.
- **Answers like a mentor.** Envie explains *why*, points out an assumption or a common mistake,
  flags anything unsafe, and ends with one short question to think about. The style is adapted
  from the [Mentor mode](https://awesome-copilot.github.com/agent/mentor/) agent
  ([source file](https://github.com/github/awesome-copilot/blob/main/agents/mentor.agent.md) in
  github/awesome-copilot, MIT License), shortened for a small model and for a live Q&A, where
  people want an answer rather than only questions back.
- **Tables stay whole.** Each Markdown table is its own source, so it isn't cut in half; a long
  table is split between rows, with its header row repeated in each piece. Tables are searched as
  sentences (e.g. *"Context window: TinyLlama 1.1B 2,048 tokens; Qwen 2.5 32,768 tokens."*), which
  the search model matches much better than rows of `|`, but Envie gets the table itself. Each
  question can get up to 2 matching tables, since they summarize a lot in little space.
- **Every readme passage says where it is,** e.g. *"(Section: Using a Different Local Model > Where
  the models come from)"*, which helps both search and Envie.
- **Only relevant context.** Each question gets up to 6 sources: the 3 most similar passages, up to
  2 tables, and the best keyword matches in the code and readmes, plus a short description of the
  project. Chunks that hardly match are left out, so a greeting
  like *"hello?"* gets a greeting back, not an answer built from unrelated text. A question that
  names a file, e.g. *"What does inference_server.py do?"*, also gets the start of that file,
  where files describe themselves.
- **Answers cite their sources by number,** e.g. *"…streams the reply piece by piece [4]"*. Each
  `[4]` is a link to those lines of the file on GitHub. Under the answer, a numbered list shows
  what each number is, e.g. **[4] src/live_qa.py, lines 539-551**, and **Sources** shows their text.
  The sources are numbered in the prompt, so Qwen only has to write `[4]`, which a small model gets
  right far more often than copying a file name and lines. The list is made by the app, so it's
  right even if Qwen forgets a citation.
- **"How does … work?" questions get the explanation and the code.** Questions with *how*, *why*,
  *explain*, *what happens* or *walk me through* are answered from the readmes and pointed at the
  code that does it, citing both. The code is found mainly by **following the readmes' own links
  to code**, e.g. `[src/main.py:217-218](src/main.py#L217-L218)` in the RAG guide, so each
  explanation comes with exactly the code it describes; the search model matches questions to
  code poorly. Keyword matches in the code (e.g. "stream", "faiss") are added too.
- **Diagrams when several files work together.** If the sources come from more than one file, Envie
  adds a small diagram of how they connect, one arrow per step, each with its citation number,
  e.g. `streamChat() [1] → do_POST() [2] → …`. The app shows it in a monospace box, and its numbers
  are links too.
- **Code is labelled,** e.g. *"(Code: src/main.py, in function stream_response)"*, and this app's
  own code (`src/live_qa.py`, whose comments mention almost every topic) is only used for code
  when the question is about Envie or names the file.
- **Questions about Envie itself,** e.g. *"How do you work?"*, always get the start of
  `src/live_qa.py` and of this guide, where the app describes itself.
- **Images are sources too.** Every image a readme shows, e.g. the learning-steps diagram, is
  indexed through a description: its alt text, the paragraphs around it, and, for a draw.io
  diagram exported as `<name>.drawio.png`, all the text in the `.drawio` file. When Envie uses an
  image, it cites it as `[docs/images/learning-steps.drawio.png]`, and the app shows the image
  with the answer and links to it on GitHub. A readme passage that shows an image also brings the
  image along as a source. To add an image, put it in a readme with `![description](path)`, with a
  good description; a new `.drawio.png` gets its diagram text automatically.
- **Made-up citations are flagged.** A small model sometimes cites a source number it wasn't
  given, e.g. `[7]` when there are 5 sources, or a file that isn't a source, e.g.
  `[docs/notes.md]`. The app checks every citation, and shows any that don't match under the
  answer: *"⚠ Not in the sources Envie was given, so possibly made up"*.
- **One chat for everyone,** updated every second in each browser.
- **The queue:** each waiting question, shortened; **hover** over one to see all of it. Move
  questions up (**↑**) or down (**↓**), remove them (**✕**), **⏸ Pause** answering, and
  **🗑 Clear chat**. There's one view: everyone who opens the app has the same controls.
- **Limits:** questions up to 300 characters, and up to 30 waiting.
- **Shown as typed:** questions and answers can't put links, images or HTML on everyone's screen.
  The only formatting is Envie's `code` and **bold**, and the only links are the citations, which
  the app builds itself.

**Citations are links.** Each numbered citation, and each entry in the numbered list under the
answer, opens those lines of the file on GitHub in a new tab, at the commit the app is running from (e.g.
`.../blob/<commit>/src/inference_server.py#L1-L12`). GitHub scrolls to the lines and highlights
them. Markdown files open as text (`?plain=1`), because line links don't work on GitHub's
formatted view of them. The **Sources** dropdown shows only the first 3 lines of each source; the
link shows the rest. Links need the project to be a clone of a
GitHub repository, and that commit to be pushed; changes you haven't committed aren't on GitHub.

---

## 1. Try it on your own computer first

**Before you start:**
- The setup in [01_readme-setup.md](01_readme-setup.md) is done: the `venv_rag` virtual
  environment exists, and the embedding model is in `models/all-MiniLM-L6-v2/`.
- **Ollama** is running (its icon is in the system tray), with Qwen downloaded. Check with
  `ollama list`; if `qwen3.5:4b` isn't there, run `ollama pull qwen3.5:4b` (about 3.3 GB). On a
  computer with an NVIDIA GPU, Ollama uses it automatically, so answers are fast.

1. In a PowerShell terminal in the project folder:
   ```powershell
   .\venv_rag\Scripts\Activate.ps1
   streamlit run src/live_qa.py --server.address 127.0.0.1
   ```
   `--server.address 127.0.0.1` keeps the app reachable only from your computer. Without it,
   Streamlit also listens on your network, so others on the same Wi-Fi could open it.
2. A browser tab opens at <http://localhost:8501>. The first load reads and indexes the project,
   which takes a minute or so.
3. Open the same address in a **private window** (`Ctrl+Shift+N` in Edge or Chrome) too, so it
   counts as a second person.
4. **Test it:**
   - **Pause** first, so questions wait in the queue. Then ask a few from both windows, e.g.
     *"What is a skill?"*, *"What does inference_server.py do?"*, *"What are the three safety
     rules?"*. Both windows show them in the queue.
   - Reorder with ↑ / ↓, remove one with ✕, and hover over a long question to see all of it. Both
     windows show the change.
   - **Resume**: the answers appear in both windows, one at a time.
5. Stop the app with `Ctrl+C` in the terminal.

---

## 2. Run it in GitHub Codespaces for a class

A codespace runs the project on a GitHub-hosted computer, so participants can reach it from their
own browsers. Anyone can start their own: you need a GitHub account, and the codespace's usage
counts against **your** account's free monthly Codespaces hours (then billing).

### Start your own codespace

1. Open the repository on GitHub. To have your own copy, **Fork** it first, and use your fork.
2. **Code** → **Codespaces** → **⋯** → **New with options…**. Choose a **4-core** machine, which
   answers noticeably faster than 2-core, and **Create codespace**.
3. Wait for the setup to finish. The dev container installs the libraries, downloads the models
   for the normal app (about 2.3 GB), installs **Ollama** and downloads **Qwen 3.5** (about
   3.3 GB). That takes several minutes, more than for the normal app alone. It's done when the
   terminal shows *"Setup complete"*. Details: [readme-dev-container-startup.md](readme-dev-container-startup.md),
   section 1.

### Start the live Q&A

1. In the codespace's terminal:
   ```bash
   streamlit run src/live_qa.py
   ```
2. **Open it,** at the address Codespaces opens for port 8501, which the terminal also prints:
   ```
     Live Q&A address (share this one): https://<your-codespace-name>-8501.app.github.dev
   ```
   The first load indexes the project, which takes a minute or two on the CPU.
3. **Share it with the class.** In VS Code's **Ports** panel, right-click port **8501** →
   **Port Visibility** → **Public**, and share the address, e.g. as a link or a QR code.

### During the talk

Participants type questions at the bottom of the page. Use **Pause** to look over the queue
before Envie answers, **✕** to remove anything unsuitable, and **↑** / **↓** to choose the order.
Everyone has these controls, so agree with the class that the presenter runs the queue.
**A codespace has no GPU, so Qwen runs on the CPU:** expect a few words per second, and a minute
or more per answer. That's what the queue is for: everyone can keep asking while Envie works
through the questions one at a time, and you choose the order. While an answer appears, open
**Sources** under the previous one and compare it with the text Envie was given. Qwen's
"thinking" is turned off, so it starts answering straight away.

### Afterwards

1. Stop the app with `Ctrl+C`.
2. Set port 8501 back to **Private** (Ports panel → right-click → **Port Visibility**).
3. Stop the codespace, so it stops using your hours: on GitHub, **Code** → **Codespaces** → **⋯**
   → **Stop codespace**.

---

## ⚠ Safety

- **A public port means anyone with the link can use the app** and see the chat, not just the
  class. There are no uploads and the input is limited, but make it public only during the
  session, and private again straight after.
- **Anyone with the link can also manage the queue:** pause, reorder, remove questions and clear
  the chat. That keeps the app simple; if it's misused, stop the app and restart it.
- **Everyone sees every question.** Ask participants not to type names or personal details.
- **Answers can be wrong.** Qwen reads only a few short chunks per question, and can misread
  them; check the **Sources**, and watch for the ⚠ made-up-citation warning.
- **Ollama's port (11434) stays private.** Its API has no password, so the dev container never
  forwards it. Only ever make port **8501** public ([readme-safety.md](readme-safety.md),
  section 11).
- **Only `.md`, `.py`, `.js`, `.json`, `.sh`, `.yml` and `.yaml` files are read,** so keep secrets out of
  those, too.

More: [readme-safety.md](readme-safety.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No address in the terminal | It's printed after the **first page load**. Open the app once, then look again. |
| *"Bad message format: Bad 'setIn' index"* | A known problem with the experimental live updates in Streamlit 1.35, when the page reloads while it's redrawing. Close the message and reload the page. |
| "Not Found" in Codespaces | Check that the address has `-8501.app` in it. |
| Participants can't open the address | Port 8501 is still **Private**. Set it to **Public** in the Ports panel. |
| *"Embedding model not found"* | The models aren't downloaded. Locally, see [01_readme-setup.md](01_readme-setup.md); in Codespaces, run `bash .devcontainer/post-create.sh`. |
| *"Ollama isn't installed here"* | The codespace was created before Ollama was added to its setup, or the setup stopped early. Run `bash .devcontainer/post-create.sh`, which skips what's already done, installs Ollama and downloads Qwen, then reload the page. Locally, install Ollama from [ollama.com](https://ollama.com). |
| *"This version requires zstd for extraction"* while installing Ollama | Run `sudo apt-get install -y zstd`, then the installer or `bash .devcontainer/post-create.sh` again. If `sudo apt-get update` fails on `dl.yarnpkg.com` (`NO_PUBKEY`), that's the image's Yarn package source, not Debian's; `zstd` still installs. |
| *"Ollama is installed but isn't answering"* | Locally, start Ollama from the Start menu. In a codespace, the app tries to start it by itself; if that fails, run `nohup ollama serve >/tmp/ollama.log 2>&1 &`, then reload the page. |
| *"Ollama doesn't have qwen3.5:4b"* | Run `ollama pull qwen3.5:4b`, then reload the page. |
| Answers are very slow | Normal on a CPU. Use a 4-core codespace, keep questions short, and use **Pause** and ✕ to keep the queue manageable. `ollama ps` shows whether Qwen is loaded. |
| A new readme isn't used | Files are read when the app starts. Stop it with `Ctrl+C` and start it again. |

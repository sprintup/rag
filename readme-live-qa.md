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

- **Answers from this project's files.** When it starts, it reads every `.md` and `.py` file in the
  project folder and its subfolders, e.g. all the readmes, the skills, agents and instructions in
  `.github/`, and the code in `src/`. New or changed files are picked up the next time the app
  starts; there's nothing to copy or upload.
- **Skips what isn't the project:** virtual environments (`venv_rag/`), `models/`, `vectorstore/`,
  `uploaded_docs/`, caches, and hidden folders other than `.github/`. Only `.md` and `.py` files are
  read, so a `.env` or other secret file is never included.
- **No uploads, no pickle.** The search index is built in memory once, when the first page loads,
  and nothing is saved to `vectorstore/`.
- **Answers like a mentor.** Envie explains *why*, points out an assumption or a common mistake,
  flags anything unsafe, and ends with one short question to think about. The style is adapted
  from the [Mentor mode](https://awesome-copilot.github.com/agent/mentor/) agent
  ([source file](https://github.com/github/awesome-copilot/blob/main/agents/mentor.agent.md) in
  github/awesome-copilot, MIT License), shortened for a small model and for a live Q&A, where
  people want an answer rather than only questions back.
- **Only relevant context.** Each question gets up to 4 matching chunks of up to 800 characters,
  plus a short description of the project. Chunks that hardly match are left out, so a greeting
  like *"hello?"* gets a greeting back, not an answer built from unrelated text. A question that
  names a file, e.g. *"What does inference_server.py do?"*, also gets the start of that file,
  where files describe themselves.
- **Answers cite files and lines,** e.g. `[src/inference_server.py, lines 1-12]`. Each piece of
  text Envie is given is labelled with its file and line numbers, and Qwen is asked to cite them.
  Under each answer, the app also lists the files and lines it gave Qwen (**From:** …), so the
  list is right even if Qwen forgets to cite one, and **Sources** shows their exact text.
- **Made-up citations are flagged.** A small model sometimes cites lines or files it was never
  given. The app checks every citation against the sources, and shows any that don't match under
  the answer: *"⚠ Not in the sources Envie was given, so possibly made up"*.
- **One chat for everyone,** updated every second in each browser.
- **The queue:** each waiting question, shortened; **hover** over one to see all of it. The
  **presenter** can move questions up (**↑**) or down (**↓**), remove them (**✕**), **⏸ Pause**
  answering and **🗑 Clear chat**. Everyone else can only ask.
- **Limits:** questions up to 300 characters; at most 2 waiting per person and 30 in total.
- **Plain text only:** questions and answers are shown exactly as typed, so nobody can put links,
  images or formatting on everyone's screen.

**The presenter key.** Each time the app starts, it makes a new random key and prints it in the
terminal (after the first page load). Opening the app with `?key=<that key>` at the end of the
address shows the queue controls. Without it, you're a participant.

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
   which takes a minute or so. The terminal then shows:
   ```
     Presenter key: Xq3v9Lk2Tm0
     Add ?key=Xq3v9Lk2Tm0 to the app's address to get the queue controls.
   ```
3. **As the presenter:** open `http://localhost:8501/?key=<your key>`. The sidebar shows the
   **Pause** and **Clear chat** buttons, and ↑ ↓ ✕ next to each question.
4. **As a participant:** open <http://localhost:8501> in a **private window** (`Ctrl+Shift+N` in
   Edge or Chrome), so it counts as a different person. It shows the queue without controls.
5. **Test it:**
   - Ask a few questions from the participant window, e.g. *"What is a skill?"*, *"What does
     inference_server.py do?"*, *"What are the three safety rules?"*. Both windows show them in
     the queue, and the answers appear in both.
   - In the presenter window, **Pause**, reorder with ↑ / ↓, remove one with ✕, and **Resume**.
   - Hover over a long question in the queue to see all of it.
   - Ask a third question from the participant window while two are waiting: it's refused
     (*"You already have questions waiting"*).
6. Stop the app with `Ctrl+C` in the terminal.

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
2. **Open it once,** at the address Codespaces opens for port 8501:
   `https://<your-codespace-name>-8501.app.github.dev`. The first load indexes the project; then
   the terminal shows the **presenter key**.
3. **Open it as the presenter** with your key added:
   ```
   https://<your-codespace-name>-8501.app.github.dev/?key=<your presenter key>
   ```
4. **Share it with the class.** In VS Code's **Ports** panel, right-click port **8501** →
   **Port Visibility** → **Public**. Share the address **without** `?key=…`, e.g. as a link or a
   QR code.

### During the talk

Participants type questions at the bottom of the page. Use **Pause** to look over the queue
before Envie answers, **✕** to remove anything unsuitable, and **↑** / **↓** to choose the order.
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
- **Never share the presenter key**, or the address with `?key=` in it, e.g. on a shared screen.
  Restarting the app makes a new key.
- **Everyone sees every question.** Ask participants not to type names or personal details.
- **Answers can be wrong.** Qwen reads only a few short chunks per question, and can misread
  them; check the **Sources**, and watch for the ⚠ made-up-citation warning.
- **Ollama's port (11434) stays private.** Its API has no password, so the dev container never
  forwards it. Only ever make port **8501** public ([readme-safety.md](readme-safety.md),
  section 11).
- **Only `.md` and `.py` files are read,** so keep secrets out of those, too.

More: [readme-safety.md](readme-safety.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No presenter key in the terminal | The key appears after the **first page load**. Open the app once, then look again. |
| The queue controls are missing | The key in the address is wrong, or the app was restarted and made a new key. Copy the latest one from the terminal. |
| "Not Found" in Codespaces | Check that the address has `-8501.app` in it. |
| Participants can't open the address | Port 8501 is still **Private**. Set it to **Public** in the Ports panel. |
| *"Embedding model not found"* | The models aren't downloaded. Locally, see [01_readme-setup.md](01_readme-setup.md); in Codespaces, run `bash .devcontainer/post-create.sh`. |
| *"Ollama isn't installed here"* | The codespace was created before Ollama was added to its setup, or the setup stopped early. Run `bash .devcontainer/post-create.sh`, which skips what's already done, installs Ollama and downloads Qwen, then reload the page. Locally, install Ollama from [ollama.com](https://ollama.com). |
| *"This version requires zstd for extraction"* while installing Ollama | Run `sudo apt-get install -y zstd`, then the installer or `bash .devcontainer/post-create.sh` again. If `sudo apt-get update` fails on `dl.yarnpkg.com` (`NO_PUBKEY`), that's the image's Yarn package source, not Debian's; `zstd` still installs. |
| *"Ollama is installed but isn't answering"* | Locally, start Ollama from the Start menu. In a codespace, the app tries to start it by itself; if that fails, run `nohup ollama serve >/tmp/ollama.log 2>&1 &`, then reload the page. |
| *"Ollama doesn't have qwen3.5:4b"* | Run `ollama pull qwen3.5:4b`, then reload the page. |
| Answers are very slow | Normal on a CPU. Use a 4-core codespace, keep questions short, and use **Pause** and ✕ to keep the queue manageable. `ollama ps` shows whether Qwen is loaded. |
| A new readme isn't used | Files are read when the app starts. Stop it with `Ctrl+C` and start it again. |

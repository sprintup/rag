# Envie – Running in the Dev Container (Codespaces)

This covers running Envie **inside the dev container**, e.g. in GitHub Codespaces. All commands
are `bash` and go in the VS Code terminal of the codespace (**Terminal → New Terminal**). To
create a codespace in the first place, see the Codespaces section of
[01_readme-setup.md](01_readme-setup.md). For how the app works, see
[02_readme-rag-user-guide.md](02_readme-rag-user-guide.md); everything there applies here except the
PowerShell commands.

**Differences from running on Windows**
- **No virtual environment.** The container itself is the isolated environment, so there's no
  `venv_rag` to activate. Run `python` and `streamlit` directly.
- **The project lives at `/workspaces/rag`.** That's the folder the terminal opens in.
- **"Local" means inside the codespace.** The model runs on the codespace's cloud machine, not
  on your laptop, and not on a hosted AI service.

## 1. Check the setup finished

When the container was created, [.devcontainer/post-create.sh](.devcontainer/post-create.sh) ran
once. It installed the libraries and downloaded both models, then installed Ollama with Qwen 3.5
for the live Q&A (section 5). Check the models are there:

```bash
ls models/TinyLlama models/all-MiniLM-L6-v2
ollama list
```

You should see files including `model.safetensors` in each folder, and `qwen3.5:4b` in Ollama's
list. If a folder is missing or empty,
or the setup failed partway, run the script again. It skips anything already downloaded:

```bash
bash .devcontainer/post-create.sh
```

To see what happened during the original setup, open the Command Palette (`F1`) and run
**Codespaces: View Creation Log**.

## 2. Run the app

```bash
streamlit run src/main.py
```

**Don't use the URLs Streamlit prints.** Its "Local", "Network" and "External" URLs are
addresses inside the codespace's cloud machine, and they time out from your browser. Instead,
open the app at the codespace's forwarded address for port 8501:

1. **Copy the codespace's URL** from your browser's address bar. It looks like
   `https://crispy-tribble-ww94rrr56gh996.github.dev/`.
2. **Open a new browser tab** and paste it into the address bar.
3. **Replace `.github.dev` with `-8501.app.github.dev`.** Leave the codespace name as it is:

   | | |
   |---|---|
   | Codespace URL | `https://crispy-tribble-ww94rrr56gh996.github.dev/` |
   | App URL | `https://crispy-tribble-ww94rrr56gh996-8501.app.github.dev/` |

4. Press Enter. The app opens.

The `-8501` tells Codespaces which port to forward to, and `.app` is GitHub's domain for forwarded
ports. If you get a "Not Found" page, check that the `-8501.app` part is there. Also check that
you're signed in to github.com with the account that owns the codespace, because forwarded
ports are private to that account.

**Loading takes longer than on a PC.** "Loading TinyLlama…" can show for a few minutes the first
time, while the 2.2 GB model is read into memory. After that, the chat heading appears.

(`http://localhost:8501` only works if you opened the codespace in the VS Code desktop app rather
than the browser.)

The heading shows something like
`Process ID: 4321 · Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 on cpu · Location: /workspaces/rag/models/TinyLlama`.

To stop the app, press `Ctrl+C` in the terminal.

## 3. Upload a document and ask questions

`uploaded_docs/` and `vectorstore/` aren't stored in git, so a new codespace starts with no
documents. Upload your PDF again from the sidebar, then use the app as described in
[02_readme-rag-user-guide.md](02_readme-rag-user-guide.md): ask a question, then open **Sources**.

## 4. Demo: proving the model runs inside the codespace

There's no Resource Monitor here, but Linux has terminal equivalents. Keep the app running,
open a **second** terminal (click **+** in the terminal panel), and replace `<PID>` with the
Process ID shown in the app.

**Network connections of the app's process:**

```bash
ss -tnp | grep "pid=<PID>,"
```

Every line should show `127.0.0.1` (this machine) on both ends, i.e. the browser tab talking to
the app through the codespace's port forwarding. Ask a question while watching: no outside
addresses appear. If `ss` isn't found, install it with `sudo apt-get update && sudo apt-get install -y iproute2`.

**CPU usage of the app's process:**

```bash
top -p <PID>
```

`%CPU` jumps while an answer is being generated. Press `q` to quit `top`.

**The model files:**

```bash
ls -lh models/TinyLlama
```

`model.safetensors`, about 2.1 GB, is the model itself. It's the folder shown as **Location** in
the app.

## 5. Live Q&A: the class asks Envie about this project

For a presentation, [src/live_qa.py](src/live_qa.py) runs Envie as one shared chat that the whole
class can ask questions in, answered one at a time from this project's own files. Starting it here,
sharing it, and trying it on your own computer first: [readme-live-qa.md](readme-live-qa.md).

## 6. Useful commands

| Task | Command |
|---|---|
| Run the app | `streamlit run src/main.py` |
| Run the live Q&A ([readme-live-qa.md](readme-live-qa.md)) | `streamlit run src/live_qa.py` |
| Start Ollama, if it isn't running | `nohup ollama serve >/tmp/ollama.log 2>&1 &` |
| See whether Qwen is loaded, and how much memory it uses | `ollama ps` |
| Terminal-only model test | `python src/small_model.py` |
| Re-run setup (install + download models) | `bash .devcontainer/post-create.sh` |
| Download a model manually | `hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/TinyLlama` |
| Rebuild the search index | `rm -rf vectorstore` then restart the app |
| Check memory use | `free -h` |
| Check disk space | `df -h /workspaces` |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model not found in /workspaces/rag/models/...` | Setup didn't finish. Run `bash .devcontainer/post-create.sh`. |
| `ModuleNotFoundError` (e.g. `streamlit`, `torch`) | Same: the library install didn't finish. Run `bash .devcontainer/post-create.sh`. |
| Upload fails with "403" | The container sets `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false` to prevent this. If you changed `devcontainer.json`, run **Codespaces: Rebuild Container** from the Command Palette. |
| The app is killed, or the codespace becomes unresponsive while loading | Not enough memory. Use a codespace with at least 8 GB of RAM (4-core or larger): **Codespaces: Change Machine Type**. |
| Answers are very slow | Normal on a 2-core machine. A 4-core machine is noticeably faster. |
| `cannot import name 'AutoTokenizer' from 'transformers'` right after starting | The browser reconnected while the app was loading. Refresh the browser tab. |
| Terminal warns "You are offline and the cache for model files in Transformers v4.22.0 has been updated…" | Harmless. It's a one-time check for an old Hugging Face cache. The app loads models straight from `models/`, so the warning about `from_pretrained()` failing doesn't apply. |
| Stuck on "Loading TinyLlama…" for more than ~3 minutes | Probably low memory. Run `free -h` in a second terminal. If little is available, or the app terminal says `Killed`, use **Codespaces: Change Machine Type** to pick a larger machine. |
| Browser says "can't reach this page" / `ERR_CONNECTION_TIMED_OUT` | You used a URL printed by Streamlit. Use the codespace URL with `.github.dev` replaced by `-8501.app.github.dev` (section 2). |
| "Not Found" page at `github.dev/pf-signin…` | The URL is missing `-8501.app`, or your browser is signed in to github.com with a different account from the codespace's owner. |
| Port 8501 isn't listed in **Ports** | The app isn't running. Check the terminal for errors. |

**Stop the codespace when you're done.** It uses your free monthly hours while running. Use
**Codespaces: Stop Current Codespace** from the Command Palette, or manage codespaces at
<https://github.com/codespaces>.

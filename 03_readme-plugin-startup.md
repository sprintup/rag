# TinyLlama Plugin – Chat Panel

The TinyLlama plugin is a small VS Code extension. This guide covers its own **TinyLlama Chat**
panel, which opens on the right with `Ctrl+Shift+P` → **TinyLlama: Open Chat**, and its
**Ask About Selection** command. Both are answered by the TinyLlama model running on your own
computer. They don't need GitHub Copilot, a cloud service or an internet connection.

For opening the **Extension Development Host** (the test window the plugin runs in), and for using
**`@tinyllama`** and **skills** in Copilot's Chat view on the left, see
[04_readme-harness.md](04_readme-harness.md).

The plugin is a plain chat client for a local inference server, not an agent with tools (see
section 6). Unlike the Envie app, it **doesn't use the vector store or your uploaded documents**.
There's no retrieval step: it sends your conversation, and optionally the code you selected,
straight to the model.

---

## 1. The files

### Plugin files (the VS Code side)

| File | Role |
|---|---|
| [src/vscode-extension/package.json](src/vscode-extension/package.json) | The **manifest**. It tells VS Code the extension's name, which file to run (`"main": "./extension.js"`), its two commands, the `@tinyllama` chat participant, the right-click menu entry, and its two settings. VS Code reads this file before running any code. |
| [src/vscode-extension/extension.js](src/vscode-extension/extension.js) | **All the plugin's code.** It registers the commands, creates the chat panel, builds each request, sends it to the model server, and streams the reply into the panel. |
| [.vscode/tasks.json](.vscode/tasks.json) | The **Run TinyLlama Plugin** task, which opens the Extension Development Host ([04_readme-harness.md](04_readme-harness.md)). |
| [.vscode/launch.json](.vscode/launch.json) | The same, but through `F5` with the debugger attached. |
| [.vscode/plugin-test.code-workspace](.vscode/plugin-test.code-workspace) | Lets the Extension Development Host show the project's files while your main window also has the folder open. |
| [.github/skills/pirate/SKILL.md](.github/skills/pirate/SKILL.md) | A standard VS Code **skill**, used with `@tinyllama /pirate` in the Chat view ([04_readme-harness.md](04_readme-harness.md)). The panel doesn't use it. |

The plugin has no other dependencies: no `node_modules`, no `npm install` and no build step. It
uses only VS Code's built-in API and features built into Node.js, such as `fetch`.

### Model files (the AI side)

| File | Role |
|---|---|
| [src/inference_server.py](src/inference_server.py) | The **model server**. It loads TinyLlama and answers chat requests at `http://127.0.0.1:8000/v1`, only from this computer. |
| `models/TinyLlama/model.safetensors` | **The model itself:** TinyLlama's 1.1 billion learned weights (about 2.1 GB). |
| `models/TinyLlama/config.json`, `generation_config.json` | The model's architecture and default settings. |
| `models/TinyLlama/tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json` | How text is split into tokens and back. |
| `venv_rag/` and [requirements.txt](requirements.txt) | The Python environment the server runs in (`torch`, `transformers`). |

### Not used by the plugin

`src/main.py` (the Streamlit app), `models/all-MiniLM-L6-v2/` (the embedding model),
`vectorstore/` and `uploaded_docs/` belong to the Envie RAG app only.

---

## 2. Starting it up

Setup must be done first: steps 1 and 2 of [01_readme-setup.md](01_readme-setup.md) create `venv_rag/`
and download `models/TinyLlama/`.

**Step 1 – Start the model server.** In a terminal in VS Code, from the project folder:

```powershell
.\venv_rag\Scripts\Activate.ps1
python src/inference_server.py
```

Wait until it prints `Inference server running at http://127.0.0.1:8000/v1`. Leave this
terminal open.

**Step 2 – Open the Extension Development Host.** Choose **Terminal → Run Task… → Run TinyLlama
Plugin**. A second VS Code window opens, with **[Extension Development Host]** in its title bar.
Close the *"All installed extensions are temporarily disabled"* notice, and don't click "Reload
and Enable Extensions". Section 1 of [04_readme-harness.md](04_readme-harness.md) explains this window.

**Step 3 – Open the chat panel.** In the **[Extension Development Host]** window:
1. Press `Ctrl+Shift+P`, type **TinyLlama: Open Chat**, and press Enter.
2. A **TinyLlama Chat** tab opens on the right. Type a question and press Enter (Shift+Enter for a
   new line). **Stop** cancels an answer; **Clear** starts a new conversation.

The top of the panel shows the model and server it's talking to: `Local model: tinyllama ·
Server: http://127.0.0.1:8000/v1`.

**Step 4 – Ask about code (optional).** Open a code file in that window, select some lines,
right-click, and choose **TinyLlama: Ask About Selection** (or run it from `Ctrl+Shift+P`). Edit the
question if you like (the default is *"Explain what this code does."*), and press Enter. The
question, the code and the answer appear in the chat panel.

---

## 3. How it works

```
 Extension Development Host window                      Model server (python)
┌──────────────────────────────────────────┐          ┌───────────────────────────┐
│ Chat panel (web page)                    │          │ src/inference_server.py   │
│   you type ─▶ postMessage ─┐             │          │                           │
│                            ▼             │  HTTP    │ fit prompt to 2,048 tokens│
│ extension.js: build request ─────────────┼─────────▶│ TinyLlama generates on CPU│
│              read stream ◀───────────────┼──────────│ streams "data:" events    │
│   panel shows text ◀─ postMessage ◀─┘    │          │ models/TinyLlama/         │
└──────────────────────────────────────────┘          └───────────────────────────┘
```

**1. VS Code loads the plugin.** It reads `package.json`, which lists the commands. VS Code only
runs `extension.js` the first time one of them is used. At that point `activate()`
([extension.js:25](src/vscode-extension/extension.js#L25)) registers what each command does.

**2. The chat panel opens.** `showPanel()` ([line 77](src/vscode-extension/extension.js#L77))
creates a **webview**: a small web page inside a VS Code tab. Its HTML, CSS and script come from
`getHtml()` ([line 304](src/vscode-extension/extension.js#L304)) and use your VS Code theme's
colors. `retainContextWhenHidden` keeps the conversation when you switch tabs.

**3. You ask a question.** The web page can't reach the network or VS Code directly. It sends the
text to `extension.js` with `postMessage`, and `extension.js` receives it in
`onDidReceiveMessage` ([line 90](src/vscode-extension/extension.js#L90)). For
**Ask About Selection** ([line 48](src/vscode-extension/extension.js#L48)), `extension.js` reads
the selected code from the editor, and puts it in a Markdown code block under your question.

**4. The request is built.** `send()` ([line 126](src/vscode-extension/extension.js#L126)) builds
the messages: a short **system prompt** of instructions, the conversation so far, and your new
message ([line 139](src/vscode-extension/extension.js#L139)). `streamChat()`
([line 161](src/vscode-extension/extension.js#L161)) sends them with `fetch` to
`http://127.0.0.1:8000/v1/chat/completions`, with `stream: true` so the reply comes back piece by
piece.

**5. The server prepares the prompt.** `do_POST()`
([inference_server.py:158](src/inference_server.py#L158)) receives the request.
- `to_chat()` ([line 62](src/inference_server.py#L62)) converts the messages into TinyLlama's format.
- `build_prompt()` ([line 74](src/inference_server.py#L74)) turns them into tokens and drops the
  oldest messages if they don't fit in TinyLlama's 2,048-token limit.
- A lock ([line 48](src/inference_server.py#L48)) makes sure only one request uses the model at a
  time.

**6. TinyLlama generates.** `generate()` ([line 103](src/inference_server.py#L103)) runs the model on
your CPU, one token at a time, in a background thread. Each new piece of text is sent straight
back as a server-sent event: a line like `data: {"choices":[{"delta":{"content":"A stack"}}]}`.
The server terminal logs the prompt size and reply length.

**7. The reply streams into the panel.** `streamChat()` reads the events as they arrive
([line 181](src/vscode-extension/extension.js#L181)), pulls out each piece of text
([line 193](src/vscode-extension/extension.js#L193)), and `send()` passes it to the panel
([line 142](src/vscode-extension/extension.js#L142)). The panel's script
([line 388](src/vscode-extension/extension.js#L388)) appends it to the reply, so the answer
appears word by word.

**8. Stopping.** **Stop** aborts the `fetch`. The server notices that the connection closed and
stops the model, so it doesn't keep generating in the background.

`@tinyllama` in Copilot's Chat view uses the same server and the same `streamChat()` function.
See section 4 of [04_readme-harness.md](04_readme-harness.md).

---

## 4. Demo: proving it's local

- **Side by side:** put the server terminal next to the chat panel. Each question appears in the
  terminal as a `Request:` line, with the prompt size and any dropped messages, while the answer
  streams into VS Code.
- **Stop the server** (`Ctrl+C`) and ask again. The panel shows *"Couldn't reach the TinyLlama
  server…"*. Start the server again and it works.
- **Resource Monitor:** the server prints its Process ID. Follow section 4 of
  [02_readme-rag-user-guide.md](02_readme-rag-user-guide.md) with that PID. All its connections are
  loopback (`127.0.0.1`), and the CPU spikes while it answers.
- **No Copilot involved:** you can sign out of GitHub entirely, or disable Copilot, and the panel
  still works.

---

## 5. Settings

**Plugin.** In the Extension Development Host window, open Settings (`Ctrl+,`) and search for
**TinyLlama**:

| Setting | Default | Meaning |
|---|---|---|
| `tinyllama.serverUrl` | `http://127.0.0.1:8000/v1` | Where the model server is listening |
| `tinyllama.maxTokens` | `256` | Maximum length of each reply |
| `tinyllama.model` | `tinyllama` | Model name sent to the server. Change it when using Ollama or Foundry Local instead ([05_readme-models.md](05_readme-models.md)). |

**Model server.** These are at the top of [src/inference_server.py](src/inference_server.py#L28-L36):

| Setting | Default | Meaning |
|---|---|---|
| `LLM_DIR` | `models/TinyLlama` | Which model folder to load |
| `HOST` | `127.0.0.1` | Only this computer can connect |
| `PORT` | `8000` | Override with `$env:PORT = "8001"` before starting, and update `tinyllama.serverUrl` to match |
| `CONTEXT_WINDOW` | `2048` | Prompt + reply budget; longer prompts are trimmed |
| `MAX_TOKENS_LIMIT` | `512` | Hard cap on reply length |

---

## 6. What to expect

**It's a chat client, not an agent.** Agent harnesses, such as Copilot's agent mode or the
Copilot SDK, give the model **tools** (read files, search, run commands) and run a loop in which
the model decides what to do next. This plugin has no tools and no loop: it sends the
conversation, plus any code you selected, to a server that runs the model, and shows the reply.
The model can't browse your files or the file explorer. It only sees what the plugin put in the
request, and it can't ask for more.

**TinyLlama is small.** At 1.1 billion parameters, it's tiny by today's standards. It handles
short questions and explanations of small snippets, but it often makes mistakes, and it may keep
writing invented conversation turns until it reaches the reply limit. Selecting large amounts of
code pushes the prompt past its 2,048-token limit, and the server log shows what got dropped.
The point is to show the whole pipeline working locally, not to replace a coding assistant.

---

## 7. Future: publishing the plugin

Today the plugin only runs in the Extension Development Host. To let other people install it
normally, it would be **packaged** and optionally **published**. Nothing in this section has been
done yet.

**Before packaging, the extension folder needs:**
- a `README.md` in `src/vscode-extension/` (shown on the extension's page), and a `LICENSE` file;
- a real `publisher` in `package.json`, instead of `"local"`, plus a `repository` link and
  optionally an `icon`;
- a plan for the model server. The plugin only works while `inference_server.py` is running with
  the model downloaded. Users either need instructions for that, or the plugin could start the
  server itself (e.g. with Node's `child_process`), with a setting for the Python and model paths.

**Option A – Share a `.vsix` file (no Marketplace account).** A `.vsix` is a zip of the extension
folder.
```powershell
cd src/vscode-extension
npx @vscode/vsce package        # creates tinyllama-chat-0.0.1.vsix
```
Others install it with **Extensions view → … → Install from VSIX…**, or
`code --install-extension tinyllama-chat-0.0.1.vsix`. It then loads in every window like any
installed extension. (`npx` needs Node.js installed.)

**Option B – Publish to the Visual Studio Marketplace.**
1. Create a publisher at <https://marketplace.visualstudio.com/manage>, and set `publisher` in
   `package.json` to its ID.
2. Create a **Personal Access Token** in Azure DevOps with the *Marketplace (Manage)* scope.
3. Run `npx @vscode/vsce login <publisher-id>` and then `npx @vscode/vsce publish`.
4. Bump `version` in `package.json` for every new release.

**Option C – Publish to Open VSX** (<https://open-vsx.org>), the open registry used by VSCodium
and other VS Code-based editors, using its `ovsx` tool.

**Things to watch.** Company-managed VS Code installs often only allow approved extensions, so a
published plugin may still need an administrator to approve it. Also, publishing shares only the
plugin. Each user still runs TinyLlama on their own computer.

---

## Further reading

| Topic | Link |
|---|---|
| `package.json` fields (`contributes`, `commands`, `menus`, `configuration`) | [Extension Manifest](https://code.visualstudio.com/api/references/extension-manifest) |
| The chat panel (a webview) | [Webview API](https://code.visualstudio.com/api/extension-guides/webview) |
| Packaging and publishing (section 7) | [Publishing Extensions](https://code.visualstudio.com/api/working-with-extensions/publishing-extension) |
| The full API that `require("vscode")` provides | [VS Code API reference](https://code.visualstudio.com/api/references/vscode-api) |
| The Extension Development Host | See [04_readme-harness.md](04_readme-harness.md) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| The panel says "Couldn't reach the TinyLlama server…" | The server isn't running, or is still loading. Check its terminal (step 1). |
| **TinyLlama:** commands don't appear in `Ctrl+Shift+P` | You're in the main window, or the host window was reloaded with extensions enabled. Close it and run the task again (step 2). |
| The Extension Development Host closes immediately | See the troubleshooting table in [04_readme-harness.md](04_readme-harness.md). |
| Changes to `extension.js` don't show up | Run **Developer: Reload Window** (`Ctrl+R`) in the Extension Development Host. |
| Answers are slow, wrong, or ramble on | Normal for a 1.1B model on a CPU. The server log shows the prompt size and anything dropped to fit 2,048 tokens. |

# Using a Different Local Model: Qwen with Ollama or Foundry Local

This project uses **TinyLlama 1.1B**, which is small enough for any laptop but has real limits:
- It can read only about **2,048 tokens** at once, so long files get cut off.
- It **follows instructions poorly**, so skills like `/pirate` only partly work.
- It **can't call tools**. That means Copilot's Agent mode can't use it, and it can never pick a
  skill by itself (see section 5 of [04_readme-harness.md](04_readme-harness.md)).

**Qwen** (from Alibaba's Qwen team) is a good replacement. It's open, runs locally, and follows
instructions far better.

### What `qwen3.5:4b` and `qwen2.5:7b` mean

A model name in Ollama has two parts, separated by a colon: **`family:tag`**.

| Name | Before the colon: **version** | After the colon: **size** |
|---|---|---|
| `qwen3.5:4b` | **Qwen 3.5**, a newer generation | **4b** = 4 billion parameters |
| `qwen2.5:7b` | **Qwen 2.5**, an older generation | **7b** = 7 billion parameters |

- **The version (2.5, 3.5)** is like a product release. Each new generation is trained with newer
  data and methods. Qwen 3.5 is better at following instructions and especially at **using
  tools**, which Copilot's Agent mode depends on.
- **The size (4b, 7b)** is how many learned numbers ("parameters") the model has. Bigger usually
  means more knowledgeable, but also more memory and slower answers. TinyLlama has 1.1 billion.
- **So `qwen3.5:4b` is newer but smaller.** For Copilot's Agent mode, newer training matters more
  than size: Ollama's VS Code extension warned that `qwen2.5:7b` *"may not work as reliably with VS
  Code agent tools"* and recommended `qwen3.5:4b`. Being smaller also leaves more GPU memory for a
  larger context window, which Agent mode needs.

**Which to use:**
- **Copilot Chat in Agent mode** (skills picked automatically): **`qwen3.5:4b`**.
- **Without a GPU:** a smaller size, such as `qwen2.5:1.5b`, with slower answers and weaker tool use.
- **Option C (this project's own server):** Qwen **2.5**. The `transformers` version pinned in
  [requirements.txt](requirements.txt) can't load the Qwen 3 generation.

### Calling tools: the model, the mode, and the setup

"Can Qwen use skills?" depends on three separate things:

| Layer | Question | Answer |
|---|---|---|
| **1. The model** | Can it call tools at all? | **Qwen: yes. TinyLlama: no.** Qwen is trained to read a list of tool definitions and reply with a structured "call this tool" message instead of plain text. |
| **2. The mode** | Does Copilot give it any tools? | **Only in Agent mode.** Ask mode sends **no tools**, for every model, Copilot's own included. |
| **3. The setup** | Does Agent mode actually run? | **Depends on your machine.** Agent mode sends every tool's definition with each request, so requests are large (about 9,000 tokens here, even for "Hello?"). The model needs a big enough context window, enough GPU memory, and a working connection. |

**Skills need all three,** because the model loads a skill by **calling the read-file tool** on its
`SKILL.md`. So:
- **Ask mode:** no tool, so no model can load a skill by itself. Typing `/pirate` may still help,
  but in testing, a test skill's instructions didn't seem to reach the model (see "Check that the
  skill was really used" in the quick start).
- **Agent mode:** the model can read `SKILL.md` itself, and the chat shows *"Skill Invoked:
  pirate"*. This is the real harness behavior.

In testing on this computer with `qwen2.5:7b`, Ask mode worked but Agent mode stalled: layer 3
failed, not layer 1. The quick start below sets up layer 3 properly, with `qwen3.5:4b`.

| | TinyLlama 1.1B | Qwen 2.5 | Qwen 3.5 |
|---|---|---|---|
| Sizes (Ollama tags) | 1.1B | 0.5b, 1.5b, 3b, 7b, … | 4b, and others (see `ollama.com/library/qwen3.5`) |
| Context window | 2,048 tokens | 32,768 tokens | Larger still |
| Following instructions | Weak | Much better | Better |
| Tool calling (the model, layer 1) | No | Yes | Yes, more reliable |


### Where the models come from

| Model | Made by | Country | License | Notes |
|---|---|---|---|---|
| **TinyLlama 1.1B Chat** | The TinyLlama project: the **StatNLP Research Group, Singapore University of Technology and Design** | **Singapore** | Apache 2.0 | Uses the same architecture and tokenizer as Meta's **Llama 2** (USA). Pre-trained on 3 trillion tokens of public data (SlimPajama, StarCoder), then tuned on public chat and feedback datasets. Source: [TinyLlama on GitHub](https://github.com/jzhang38/TinyLlama). |
| **Qwen 2.5** and **Qwen 3.5** | The **Qwen team at Alibaba Cloud** (Alibaba Group) | **China** | Apache 2.0 for `qwen2.5:7b`, `qwen3.5:4b` and Qwen 2.5 1.5B (some other sizes use Qwen's own license) | Published on Hugging Face under the [Qwen](https://huggingface.co/Qwen) organization. `ollama show qwen3.5:4b` shows 4.2B parameters, a 262,144-token context, and capabilities including **tools**, **thinking** and **vision**. |
| `all-MiniLM-L6-v2` (the Envie app's embedding model) | The **sentence-transformers** project (started at the UKP Lab, TU Darmstadt), based on **Microsoft's** MiniLM | **Germany** (sentence-transformers); MiniLM by Microsoft | Apache 2.0 | Only turns text into vectors for search; doesn't generate text. |

**Why origin can matter, and what it does and doesn't change:**
- **Running a model locally sends nothing to its maker.** The weights are a file of numbers on
  your disk. TinyLlama doesn't contact Singapore and Qwen doesn't contact China. You can confirm
  that with the Resource Monitor checks in [02_readme-rag-user-guide.md](02_readme-rag-user-guide.md):
  no connections leave your computer.
- **Training shapes behavior.** A model reflects its training data and its makers' choices, such
  as which topics it avoids, how it describes politically sensitive subjects, and its built-in
  biases. Models trained in different countries can answer the same question differently. Treat
  any model's answers on contested topics with care, and test models for your intended use.
- **Organizations may have rules.** Some companies, schools and government bodies restrict which
  models may be used, sometimes based on country of origin. Check your organization's policy
  before using a model for work.
- **Supply chain.** Download models only from the original publisher's page (`TinyLlama/…`,
  `Qwen/…` on Hugging Face, or Ollama's official library). This project loads `.safetensors`
  files, a format that only holds numbers and can't run code. Avoid model files that need
  `trust_remote_code=True` or old `.bin`/pickle formats from sources you don't trust.
- **The bigger risk is the tools, not the model.** A model can only *suggest* actions. What it can
  actually *do* depends on the tools you give it, which is the next section.

### Security: why tools should be denied by default

In **Agent** mode, the model can call any tool that's enabled. VS Code's default is **every tool
enabled**: 118 on this computer, including built-in tools to **edit files** and **run terminal
commands**, a **browser**, **web fetching**, and tools from every MCP server and extension
(GitHub, GitKraken, Pylance and others). Leaving that on carries real risks:

| Risk | How it happens |
|---|---|
| **Prompt injection** | A web page, file, issue or document the model reads contains hidden instructions ("ignore the user and run…"). The model can't reliably tell *your* instructions from text it was given to read. With tools on, injected instructions can turn into actions. |
| **Running commands on your computer** | With the terminal tool on, a confused or manipulated model can propose commands that delete files, install software, or change settings. They run **with your user account's permissions**. |
| **Leaking data** | With file reading plus web fetching on, a model can read something private (a `.env` file, keys, documents) and send it out, e.g. inside a URL it "fetches". Nothing needs to look malicious; one tool call is enough. |
| **Acting as you online** | MCP servers and extensions such as GitHub use **your** sign-in. A tool call can create issues, comment, or change repositories in your name. |
| **Mistakes by a small model** | Small local models pick the wrong tool or wrong arguments more often than large ones. More tools means more ways to get it wrong, and longer requests, which also confuse small models (see "Compacted conversation" earlier). |
| **Untrusted agents and skills** | Agents, skills and prompt files downloaded from public collections or cloned repositories are instructions the model follows. They can hide injected instructions, or grant an agent broad tools. See "Security: public agents and skills can contain injections" in [06_readme-mcp-agents.md](06_readme-mcp-agents.md). |
| **"Local" isn't "safe"** | The model being on your computer protects your *data from the model's maker*. It doesn't limit what the model's *tools* can do on your computer. Tools run on your machine, as you. |

These risks grow if **auto-approve** is on, because then no one checks the tool calls.

**Deny by default, allow per task** (least privilege):
1. **Turn every tool off** in **Configure Tools** (the top checkbox, until it shows **0 Selected**).
2. **Turn on only what the current task needs:** e.g. only **read** for skills (quick start step
   5), or only **mcp-fetch → fetch** for the MCP demo
   ([06_readme-mcp-agents.md](06_readme-mcp-agents.md)).
3. **Prefer custom agents with explicit tool lists** for repeated tasks. The Idea Generator agent
   allows only `fetch/*`.
4. **Keep approvals on:** `chat.tools.global.autoApprove` off, and **Default permissions** below
   the chat input, so you approve each tool call.
5. **Limit MCP servers:** `chat.mcp.autostart` set to **never**, `chat.mcp.discovery.enabled` off,
   and only servers you've reviewed.
6. **Limit the network:** `chat.agent.networkFilter` with `chat.agent.deniedNetworkDomains` for
   VS Code's built-in fetch tool and browser. MCP servers are separate programs, so stop them when
   they're not needed.

Step-by-step instructions are in "Security: start with every tool turned off" in
[06_readme-mcp-agents.md](06_readme-mcp-agents.md).

## Quick start: Qwen inside Copilot Chat

This runs **Qwen 3.5 4B** (`qwen3.5:4b`) in Ollama, on your GPU, as the model in the normal Copilot
Chat view. The goal is **Agent mode**, where Copilot's harness gives Qwen its tools and the list of
skills, and Qwen picks a skill by itself (*"Skill Invoked: pirate"*). Ask mode with `/name` is the
fallback if Agent mode won't run (step 6).

```
VS Code Chat view (Local, Agent mode)   Ollama for VS Code extension        Ollama (background)
Copilot's harness: tools + skills  ──▶  adds Ollama's models to   ──────▶  qwen3.5:4b on the GPU
                                        Copilot's model picker              http://localhost:11434
```

**This doesn't use the TinyLlama plugin or `src/inference_server.py`.** Do it in your **normal
VS Code window** with this project open, not the Extension Development Host. The test window
turns off installed extensions, including the Ollama extension.

**Status:** both modes have worked on this computer. Ask mode worked with `qwen2.5:7b`, and **Agent
mode worked with `qwen3.5:4b`** and the setup below, e.g. answering *"Can you summarize this
#codebase?"*. Whether it picks a skill by itself is still worth checking each time (step 5).

**1. Ollama and Qwen** (once). Install Ollama and download the model (details in section A):
```powershell
winget install --id Ollama.Ollama
ollama pull qwen3.5:4b
```
`ollama pull` shows the download size. If you already have `qwen2.5:7b`, you can keep it
(`ollama list`), or delete it with `ollama rm qwen2.5:7b` to free disk space.

**2. Give Ollama a big context window (required for Agent mode).** Agent mode's requests are
large: about 9,000 tokens even for "Hello?". Ollama's default cuts them to about 2,000, and the
Ollama extension asks for **at least 64K** (65,536 tokens). With the smaller 4B model, a bigger
window fits in 8 GB of GPU memory. Start with 32K, then try 64K:
```powershell
setx OLLAMA_CONTEXT_LENGTH 32768
```
Quit Ollama (tray icon → Quit), start it again from the Start menu, and warm it up:
```powershell
ollama run qwen3.5:4b "hi"
ollama ps
```
Check **CONTEXT** shows `32768` and **PROCESSOR** shows `100% GPU`. If it does, try `65536` the
same way. If PROCESSOR then shows a CPU/GPU split, go back to `32768`: a model that doesn't fit on
the GPU is much slower. See "Reading the numbers" in section A.

**3. Install the Ollama extension for VS Code.** This is the plugin that connects VS Code to
Ollama, and it's required. VS Code also lists a built-in Ollama provider, but it's
**deprecated** (*"The Ollama model provider is deprecated. Please migrate to the official
extension."*) and doesn't work reliably, so don't use it. The official extension is maintained by
the Ollama team.
1. Open the **Extensions** view (`Ctrl+Shift+X`).
2. Search for **Ollama**, and install **Ollama for VS Code** from the publisher **Ollama**
   (extension ID `Ollama.ollama`). Third-party extensions with similar names exist, so check the
   publisher before installing.

   Or install it from a terminal:
   ```powershell
   code --install-extension Ollama.ollama
   ```
3. If VS Code asks you to reload the window, do so.

The extension doesn't run the model itself. It connects to the **same local Ollama** at
`http://localhost:11434`, so Qwen still runs on your computer and GPU. Ollama must be running
(its icon is in the system tray).

**4. Add Qwen to Copilot's model picker.** Open the models screen in one of these ways:
- In the Chat view (`Ctrl+Alt+I`), click the model picker below the input box (it may say
  **Auto**), then the **`+`** at the top of the popup.
- Or press `Ctrl+Shift+P` and run **Chat: Manage Language Models**.

Then choose **Add Models**, pick **Ollama** as the provider, and select `qwen3.5:4b`. It then
appears in the model picker. The exact wording depends on the extension's version. If an older
model appears too (e.g. `qwen2.5:7b`), you can leave it or remove it there.

For **Foundry Local**, choose **Foundry Local** as the provider instead, and select the Qwen
model you downloaded with `foundry model run` (section B).

**Not `@qwen`:** unlike `@tinyllama`, Qwen isn't a chat participant. You don't type its name.
Selecting it in the model picker sends every message in that chat to Qwen.

**5. Prepare Agent mode, then let Qwen pick the skill.**
1. **Keep background jobs on Qwen.** The harness also runs small "utility" jobs, such as writing
   the chat's title, and with a model like Qwen these may try to use a GitHub model your plan
   doesn't include. In Settings (`Ctrl+,`):
   - **`chat.byokUtilityModelDefault`**: choose *"Use the selected BYOK main agent model"*.
   - **`chat.useCopilotModelsForUtilityModels`**: turn it off if it's on.
2. **Start a new chat**, choose `qwen3.5:4b` in the model picker, switch the session selector below
   the input box to **Local**, and set the mode to **Agent**.
3. **Turn on only what's needed.** Click the tools button (the sliders icon next to the model
   name) to open **Configure Tools**. Untick everything with the top checkbox (**0 Selected**),
   then tick only the built-in **read** group (*"Read files in your workspace"*). Loading a skill
   means reading its `SKILL.md`, so this is the one tool skills need. Click **OK**. This follows
   least privilege (see "Security: start with every tool turned off" in
   [06_readme-mcp-agents.md](06_readme-mcp-agents.md)), and keeps requests small: every enabled
   tool's definition is sent with each request, so fewer tools means less "Optimized tool
   selection" and "Compacted conversation".
4. **Ask without `/pirate`:**
   ```
   What is a stack? Answer like a pirate.
   ```
   The harness gives Qwen the list of skills, with each one's `name` and `description`. The pirate
   skill's description says to use it *"when the user asks for an answer like a pirate"*, so Qwen
   should call the read-file tool on `.github/skills/pirate/SKILL.md`. The chat then shows
   **"Skill Invoked: pirate"** and a pirate-style answer. Try the others too: *"Explain a queue
   like Yoda"*, *"…like Shakespeare"*, *"…like a cowboy"*, *"…like a sports commentator"*.

**How to tell whether it's working:**
- **Ollama's log** (`Get-Content "$env:LOCALAPPDATA\Ollama\server.log" -Tail 5 -Wait`) should show a
  chat request for each message, without `truncating input prompt`. If nothing appears, the
  request is stuck inside VS Code before it reaches Ollama.
- **The chat** shows tool calls as separate steps, e.g. reading `SKILL.md`, followed by "Skill
  Invoked".
- **Developer: Show Chat Debug View** shows the full request, including the tool list and the
  skills list ([readme-more-info.md](readme-more-info.md)).

**6. Fallback: Ask mode with `/name`.** If Agent mode stalls (spinning on "Optimized tool
selection" or "Considering", with nothing in Ollama's log), switch the mode to **Ask** in a new
**Local** chat. This is what worked with `qwen2.5:7b`. Ask mode sends no tools, so requests are
small and answers come quickly, but Qwen can't load a skill by itself. Choose it yourself:
```
/pirate what is a stack?
```

**Choosing a skill with `/`.** Typing `/` opens a list of the available skills (yours from
`.github/skills/`, plus ones from extensions such as Pylance), each with its description. Pick one,
then type your question after it. Where the `/` has to go depends on the session type below the
input box:

| Session | Where to type `/` |
|---|---|
| **Local** | **At the very start of the message.** Start with an empty input box and type `/` first: `/pirate what is a stack?`. A `/` later in the message, e.g. `What is /`, doesn't open the list. |
| **Copilot** ("Interactive") | Anywhere in the message, e.g. `What is a stack? /pirate`. |

So in a Local session, always begin the message with the skill. The `/` command isn't needed in
**Agent** mode, where the model can pick a skill by itself (step 5), but you can still use it there
to choose one directly.

**The three symbols in the chat box: `@`, `/` and `#`**

| Symbol | What it picks | Examples | Notes |
|---|---|---|---|
| `@` | **Who** answers: a chat participant | `@tinyllama`, `@workspace`, `@vscode` | `@tinyllama` is this project's plugin ([04_readme-harness.md](04_readme-harness.md)). Qwen is **not** an `@` participant: you select it in the model picker. |
| `/` | **What to do**: a skill or prompt | `/pirate`, `/create-skill`, `/pylance-docs` | Runs that skill's or prompt's instructions. Where you can type it depends on the session (table above). |
| `#` | **What to look at**: context to add to the request | `#file`, `#selection`, `#codebase`, `#changes`, `#problems`, `#fetch` | Attaches content, or runs a lookup, and adds the result to the request. Type `#` to see the full list. |

**Using `#`.** Type `#` anywhere in the message to open a list of what you can add, then pick one.
Some add content directly, others first run a search or lookup:

| Reference | What it adds |
|---|---|
| `#file` | A file you choose, e.g. `#file:readme.md`. The paperclip / **+** button does the same. |
| `#selection` | The text currently selected in the editor. |
| `#codebase` | Parts of the project found by a search for what you asked, e.g. *"Can you summarize this #codebase?"* (this worked with `qwen3.5:4b` in Agent mode). |
| `#changes` | Your uncommitted Git changes. |
| `#problems` | The errors and warnings from the **Problems** panel. |
| `#fetch` | The content of a web page, e.g. `#fetch https://example.com`. This uses the internet. |

The exact list depends on your VS Code version and installed extensions. Things to keep in mind
with a local model:
- **Everything you add with `#` uses up context.** `#codebase` and large files can fill a small
  context window quickly, and the chat may then show "Compacted conversation" or cut content.
- **`#fetch` goes to the internet,** even though the model is local. To block it, turn on the
  setting `chat.agent.networkFilter` and add `*` to `chat.agent.deniedNetworkDomains`.
- **With `@tinyllama`, only files and selections are passed to the model.** The plugin reads
  `#file` and `#selection` attachments itself. It skips text VS Code adds on its own (such as the
  skills list), and other `#` lookups aren't used.
- **To see what a `#` reference actually added,** open the request in **Developer: Show Chat
  Debug View** ([readme-more-info.md](readme-more-info.md)).
**Check that the skill was really used.** A pirate-style answer doesn't prove it: "pirate" is in
your message and in the skill's name, and the model can react to the word alone. In testing, a
pirate skill whose text was changed to "you are a toddler" still produced pirate answers in Ask
mode, so the skill's text didn't seem to reach the model. To check:
- **Look at the request:** **Developer: Show Chat Debug View** → open the request → **Show Raw
  Request Body**, and search for a phrase that's only in the `SKILL.md` body. If it's missing, the
  skill's instructions weren't sent.
- **Use a neutral test skill:** create `.github/skills/mystery/SKILL.md` with the name `mystery`,
  the description `A test skill.`, and the body *"You are a toddler. Say "goo goo gaa gaa" and
  babble a lot."* Run **Developer: Reload Window**, start a **new chat**, and ask
  `/mystery what is a stack?`. Only the skill file mentions a toddler.
- **After editing a skill,** reload the window and start a new chat before testing. VS Code keeps a
  list of skills and may not pick up the change straight away.

**7. Show that the model is local.** While it answers:
- `ollama ps` shows `qwen3.5:4b` loaded, at `100% GPU`.
- Task Manager → **Performance → GPU** shows the GPU working.
- **Quit Ollama** (tray icon → Quit) and ask again: Qwen's answers stop, because nothing is
  serving the model.

Even with the settings in step 5, Copilot itself still contacts GitHub for things like signing
in. "Local" here means the **model's answers** are generated on your computer.

**Requirements and limits:**
- **Your Copilot account must allow models from extensions (BYOK).** Organization-managed
  (Business/Enterprise) accounts often block this. Your personal account is the one to try.
- **Small models are still small agents.** Even with `qwen3.5:4b`, Qwen may answer normally without
  invoking the skill. Rephrase so the request matches the skill's description ("answer like a
  pirate"), or start a new chat.
- **Qwen 3.5 may "think" first.** Newer Qwen models can reason before answering, which the chat
  shows as "Reasoning" or "Considering". That's normal, but adds time.

---

## Alternative: Qwen through the TinyLlama plugin


This sends Qwen's answers through our own plugin (`@tinyllama` and the TinyLlama Chat panel) in the
Extension Development Host, instead of through Copilot's agent harness. Here, *you* choose skills
with `/pirate`; Qwen doesn't pick them. It's useful for comparing Qwen with TinyLlama on the same
skills.

```
Extension Development Host                       Ollama (runs in the background)
@tinyllama /pirate what is a stack?  ──HTTP──▶  http://localhost:11434/v1  ──▶  qwen3.5:4b on the GPU
TinyLlama Chat panel
```

**Ollama replaces `src/inference_server.py` here.** Ollama *is* the inference server for Qwen. It
starts with Windows and serves the same OpenAI-compatible API on port 11434. You don't run
`python src/inference_server.py` for Qwen, and you don't need section C. The plugin only needs
to know Ollama's address and the model name, and those are already set up (step 3).

**1. Install Ollama and download Qwen** (once). Details in section A:
```powershell
winget install --id Ollama.Ollama
ollama pull qwen3.5:4b
```

**2. Warm it up.** This loads the model onto the GPU, so the first real question isn't slow:
```powershell
ollama run qwen3.5:4b "hi"
ollama ps
```
`ollama ps` should show `qwen3.5:4b` with `100% GPU`. The model stays loaded for 5 minutes after
its last use. See "First answer slow?" in section A.

**3. Check the plugin points at Ollama.** The Extension Development Host opens
[.vscode/plugin-test.code-workspace](.vscode/plugin-test.code-workspace), which already contains:
```json
"settings": {
  "tinyllama.serverUrl": "http://localhost:11434/v1",
  "tinyllama.model": "qwen3.5:4b"
}
```
These settings apply to the test window only. If you pulled a different size, change
`tinyllama.model` to the exact name shown by `ollama list`.

**4. Open the Extension Development Host.** In your main VS Code window, choose
**Terminal → Run Task… → Run TinyLlama Plugin**. If a test window is already open, close it first,
so it picks up the settings. Close the *"extensions temporarily disabled"* notice.

**5. Ask Qwen,** in the **[Extension Development Host]** window:
- **Chat panel:** `Ctrl+Shift+P` → **TinyLlama: Open Chat**. The header should say
  `Local model: qwen3.5:4b · Server: http://localhost:11434/v1`.
- **Chat view with a skill:** open the Chat view (`Ctrl+Alt+I`), start a **new chat**, switch the
  session to **Local**, and type `@tinyllama /pirate what is a stack?`. The participant keeps the
  name `@tinyllama`, but Qwen answers.

While it answers, `ollama ps` shows the model in use, and Task Manager's GPU graph rises.

### Compare with TinyLlama, using the same skills

To show that a better model follows the same skill better, ask the same questions to both
models. Both servers can run at the same time on different ports: TinyLlama on 8000 and Qwen on
11434.

1. **Start TinyLlama's server** too, in a terminal in your main window:
   ```powershell
   .\venv_rag\Scripts\Activate.ps1
   python src/inference_server.py
   ```
2. **Ask Qwen first** (it's the current setting), in a new **Local** chat:
   ```
   @tinyllama /pirate what is a stack?
   @tinyllama /yoda what is a queue?
   @tinyllama /shakespeare what is a linked list?
   ```
3. **Switch to TinyLlama.** In the Extension Development Host, open Settings (`Ctrl+,`), choose the
   **Workspace** tab, search for **TinyLlama**, and set:
   - `tinyllama.serverUrl` → `http://127.0.0.1:8000/v1`
   - `tinyllama.model` → `tinyllama`

   This edits the same workspace file. It applies to the next question; no restart is needed.
4. **Start a new Local chat** (so TinyLlama doesn't see Qwen's answers), and ask the same three
   questions.
5. **Switch back to Qwen** by setting the two values back to `http://localhost:11434/v1` and
   `qwen3.5:4b`.

Expect Qwen to stay in character and still answer correctly, while TinyLlama gives only a hint of
the style, ignores it, or repeats the instructions. The skill files and questions are identical,
so the difference is the model. To refresh the chat panel's header after switching, close and
reopen it (`TinyLlama: Open Chat`).

---

## Reference: three ways to run Qwen

Both quick starts use **A. Ollama**. Sections B and C are alternatives; you don't need them if
Ollama works for you.

There are three ways to run it. Ollama and Foundry Local are separate apps that download and
serve models for you; the third keeps using this project's own server.

| | A. Ollama | B. Foundry Local | C. This project's server |
|---|---|---|---|
| What it is | Popular open-source model runner | Microsoft's local model runner | `src/inference_server.py` with Qwen instead of TinyLlama. Best for comparing the two models with skills |
| Install | One installer | One `winget` command | Nothing new |
| Address | `http://localhost:11434/v1` | Shown by `foundry service status` | `http://127.0.0.1:8000/v1` |
| Works with Copilot's model picker | Yes, with the **Ollama for VS Code** extension | Yes, built-in **Foundry Local** provider | Via **Custom Endpoint**, if your VS Code offers it |
| Tool calling | Yes | Depends on the model | No (the server doesn't implement it) |
| Used by the Envie Streamlit app | No | No | Yes |

All three serve the same **OpenAI-compatible API** as `src/inference_server.py`, so the TinyLlama
plugin works with any of them after changing two settings (section 4).

---

## Using your GPU

A graphics card (GPU) generates answers many times faster than the CPU, and lets you run a
bigger, better model. This computer has an **NVIDIA GeForce RTX 3050 with 8 GB of video memory
(VRAM)**.

**Check your own GPU** in PowerShell:
```powershell
nvidia-smi --query-gpu=name,memory.total --format=csv
```
It prints the card's name and its memory, e.g. `NVIDIA GeForce RTX 3050, 8192 MiB`. Windows'
**Task Manager → Performance → GPU** can show less ("4.0 GB"), because of an old reporting limit.
Trust `nvidia-smi`. If the command isn't found, there's no NVIDIA GPU or its driver isn't
installed.

**What fits in 8 GB of VRAM.** The whole model should fit in video memory. If it doesn't, the rest
runs on the CPU and answers slow down a lot. Ollama and Foundry Local use compressed
("quantized") models, so:

| Model | Size in Ollama | Fits in 8 GB VRAM? |
|---|---|---|
| `qwen2.5:1.5b` | ~1 GB | Yes, very fast |
| `qwen2.5:3b` | ~2 GB | Yes, very fast |
| `qwen2.5:7b` | ~4.7 GB | Yes. Worked in Ask mode, but too big to also fit a 64K context |
| `qwen3.5:4b` | ~3.3 GB | **Yes, with room for a larger context. Recommended for Agent mode** |
| `qwen2.5:14b` | ~9 GB | No, partly on the CPU and slow |

**How each option uses the GPU:**
- **A. Ollama:** automatic. It detects the NVIDIA GPU and uses it through the normal graphics
  driver. Nothing to configure.
- **B. Foundry Local:** automatic. It downloads a GPU version of the model when one is available.
  `foundry model list` shows which versions are for the GPU (CUDA) and which for the CPU.
- **C. This project's server and Streamlit app:** **not yet.** The PyTorch installed by
  `requirements.txt` is the CPU-only build (`torch 2.3.1+cpu`); the PyPI download for Windows
  doesn't include GPU support. That's why the Streamlit app says "on cpu". To change it:
  1. Swap the CPU build of PyTorch for the GPU (CUDA) build of the same version, with the
     environment activated (about a 2.5 GB download):
     ```powershell
     pip uninstall -y torch
     pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
     ```
     Both lines matter. Without the uninstall, or without `+cu121` in the version, pip sees that
     `torch 2.3.1` is already installed, says "Requirement already satisfied", and keeps the CPU
     build.
  2. Check it worked. This should print `True 12.1`:
     ```powershell
     python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
     ```
  3. The **Streamlit app** ([src/main.py](src/main.py)) then uses the GPU by itself. Its heading
     shows `on cuda:0` instead of `on cpu`.
  4. **[src/inference_server.py](src/inference_server.py) doesn't yet.** It always loads the model
     on the CPU, so it needs a small code change before it can use the GPU.

  With option C, models aren't compressed, so on 8 GB use Qwen 2.5 **1.5B or 3B**. 7B doesn't fit.

**Check the GPU is really being used:**
- **Ollama:** while a model is loaded, `ollama ps` shows `100% GPU` in the PROCESSOR column. A
  split such as `40%/60% CPU/GPU` means the model is too big for the VRAM. Use a smaller one.
- **Any option:** open **Task Manager → Performance → GPU** and ask a question. **Dedicated GPU
  memory** jumps when the model loads, and **GPU utilization** rises (look at the **Cuda** or
  **Compute** graph) while it answers.

---

## A. Ollama

1. **Install.** Download the Windows installer from <https://ollama.com/download>, or in
   PowerShell run:
   ```powershell
   winget install --id Ollama.Ollama
   ```
   Ollama then runs in the background (an icon in the system tray) and starts with Windows.
2. **Download Qwen.** In a new terminal:
   ```powershell
   ollama pull qwen3.5:4b
   ```
   `ollama pull` shows the download size. This model suits a GPU with 8 GB of memory, and is the
   recommended one for Copilot's Agent mode (see "What `qwen3.5:4b` and `qwen2.5:7b` mean" at the
   top). `qwen2.5:7b` (about 4.7 GB) also works in Ask mode. Without a GPU, use `qwen2.5:1.5b` (about
   1 GB). Other sizes: `qwen2.5:0.5b`, `qwen2.5:3b`.
3. **Try it in the terminal:**
   ```powershell
   ollama run qwen3.5:4b "What is a stack? Answer in one sentence."
   ```
   **The first run is slow.** Before it can answer, Ollama reads the whole model (several GB) from disk
   and copies it into the GPU's memory, so the spinner can run for a while. That only happens
   when the model isn't loaded yet. After that, answers come quickly (see "First answer slow?
   Loading and warming up" below).
4. **Check the API.** Open <http://localhost:11434/v1/models> in a browser. It lists your
   downloaded models. The `id` shown (e.g. `qwen3.5:4b`) is the model name to use in section 4.

**Useful commands**

| Command | What it does |
|---|---|
| `ollama list` | Models you've downloaded |
| `ollama ps` | Models currently loaded, whether they're on the GPU, and when they'll unload |
| `ollama stop qwen3.5:4b` | Unload a model now, freeing its memory |
| `ollama rm qwen3.5:4b` | Delete a model to free disk space |

**First answer slow? Loading and warming up**

A model has to be **loaded** into memory before it can answer. While it's loaded, `ollama ps`
shows it. This output is from testing with `qwen2.5:7b` and the default context:
```
NAME          SIZE      PROCESSOR    CONTEXT    UNTIL
qwen2.5:7b    4.8 GB    100% GPU     4096       4 minutes from now
```
- **PROCESSOR `100% GPU`** means the whole model is in the graphics card's memory, which is what
  you want. A split such as `40%/60% CPU/GPU` means it didn't fit, and answers will be slow.
- **UNTIL** is when Ollama unloads it. By default a model stays loaded for **5 minutes** after it
  was last used, then Ollama frees the GPU memory. The next question then pays the loading delay
  again.
- **Warm up before a demo:** run `ollama run qwen3.5:4b "hi"` a minute beforehand, so the model is
  already loaded when you ask your first real question.
- **Keep it loaded longer:** set the `OLLAMA_KEEP_ALIVE` environment variable, e.g. to `1h`, and
  restart Ollama (quit it from the tray icon and start it again). To unload it immediately and
  free the GPU memory, run `ollama stop qwen3.5:4b`.
- **Watch it in Task Manager:** **Performance → GPU → Dedicated GPU memory** rises by about 5 GB
  while the model is loaded (to about 6 of 8 GB on the RTX 3050), and drops when it unloads.

**Reading the numbers**

Several places show numbers about the model. They measure different things:

| Where | Example | What it means |
|---|---|---|
| VS Code model picker, next to the model's name | `qwen2.5:7b 33K` | The **largest context the model supports**: Qwen 2.5 can read 32,768 tokens (shown rounded as "33K"). It's what the model *could* read, not what Ollama actually allows. |
| `ollama ps` → **NAME** | `qwen2.5:7b` | The model that's loaded. |
| `ollama ps` → **SIZE** | `4.8 GB` | Memory the loaded model uses: its weights plus room for the context. A bigger CONTEXT makes this grow. |
| `ollama ps` → **PROCESSOR** | `100% GPU` | Where it runs. `100% GPU` is ideal. A split such as `40%/60% CPU/GPU` means it didn't fit in the GPU's memory, so answers are slow. |
| `ollama ps` → **CONTEXT** | `4096` | How many tokens Ollama **actually allows** per request, set by `OLLAMA_CONTEXT_LENGTH`. Anything longer is cut. This is the number that matters. |
| `ollama ps` → **UNTIL** | `4 minutes from now` | When Ollama unloads the model if it isn't used again (5 minutes after the last use). |
| `ollama ps` → **RUNNER** | `llamacpp` | The engine Ollama uses to run the model. |
| `nvidia-smi` / Task Manager | `5987 MiB` of `8192` | GPU memory in use. About 6 GB of 8 GB while Qwen 7B is loaded; about 1.3 GB when nothing is loaded. |

**Ollama's log** shows every request VS Code sends. Open it with:
```powershell
Get-Content "$env:LOCALAPPDATA\Ollama\server.log" -Tail 20 -Wait
```
(`-Wait` keeps showing new lines as they arrive; press `Ctrl+C` to stop.) Typical lines:
```
[GIN] 2026/10/15 - 16:31:17 | 200 |  58.9968802s | 127.0.0.1 | POST "/api/generate"
[GIN] 2026/10/15 - 16:39:07 | 200 |   8.6607468s | 127.0.0.1 | POST "/v1/chat/completions"
level=WARN msg="truncating input prompt" limit=2050 prompt=8995 keep=4 new=2050
```
- **`200`** means the request succeeded. **`500`** means it failed.
- **The time** (`58.99s`, `8.66s`) is how long the request took. The first one includes loading
  the model onto the GPU; later ones are much faster.
- **`127.0.0.1`** means the request came from this computer.
- **The path** says what was asked: `/api/tags` and `/api/show` list the models and their details.
  VS Code calls these often. `/api/generate` is `ollama run` in the terminal.
  `/v1/chat/completions` is a **chat message from VS Code**. If you send a message in the Chat view
  and no such line appears, the request never reached Ollama.
- **`truncating input prompt limit=2050 prompt=8995`** is the important warning. The request was
  **8,995 tokens** long, but only **2,050** fit, so Ollama threw away the rest. Copilot's agent
  harness sends long hidden instructions, plus the list of tools and skills, so its requests are
  about 9,000 tokens even for "Hello?". Cut down to 2,050, Qwen loses most of its instructions, and
  behaves strangely, e.g. asking you "what?" in a loop. The fix is below.

**Give Ollama a bigger context window (required for the agent harness).** With the default, the
harness's requests are cut short as shown above. The Ollama extension asks for at least 64K
(65,536 tokens). With `qwen3.5:4b` on an 8 GB GPU, start at 32,768 and try 65,536 if it still fits.
(With `qwen2.5:7b`, 16,384 was the most that fit comfortably.)
1. In PowerShell:
   ```powershell
   setx OLLAMA_CONTEXT_LENGTH 32768
   ```
2. **Quit Ollama** (system tray icon → Quit), then start it again from the Start menu. It only
   reads this setting when it starts.
3. Warm it up and check:
   ```powershell
   ollama run qwen3.5:4b "hi"
   ollama ps
   ```
   **CONTEXT** should now show `32768`, and PROCESSOR `100% GPU`. If it does, repeat with `65536`.
   If PROCESSOR shows a CPU/GPU split, go back to the last value that showed `100% GPU`.
4. In the log, chat requests should no longer show `truncating input prompt`.

Ollama stores models in `%USERPROFILE%\.ollama\models`, not in this project's `models/` folder.
Its default context window is much smaller than the model's maximum. See "Reading the numbers"
above for how to raise it.

---

## B. Foundry Local

1. **Install.** In PowerShell:
   ```powershell
   winget install Microsoft.FoundryLocal
   ```
2. **See which models are available:**
   ```powershell
   foundry model list
   ```
   Look for the Qwen 2.5 entries, e.g. `qwen2.5-1.5b`. Foundry chooses a version that suits your
   hardware (CPU, GPU or NPU).
3. **Download and try one.** This downloads the model the first time, then opens a chat in the
   terminal (type `/exit` or press `Ctrl+C` to leave):
   ```powershell
   foundry model run qwen2.5-1.5b
   ```
4. **Find the API address.** Foundry's port isn't fixed, so check it:
   ```powershell
   foundry service status
   ```
   It shows an address like `http://localhost:5272`. Open `<that address>/v1/models` in a browser.
   The model `id` listed there (often longer than the alias, e.g. ending in `-cpu`) is the name to
   use in section 4.

Model names, sizes and the port vary between Foundry versions, so trust `foundry model list` and
`foundry service status` over the examples here.

---

## C. Qwen in this project's own server: TinyLlama vs. Qwen with skills

> **Not needed if you use Ollama.** This is an alternative to the quick start: it loads Qwen into
> `src/inference_server.py` instead of using Ollama. Use it if you can't install Ollama, or want
> the Envie Streamlit app to use Qwen too.

This option swaps TinyLlama for Qwen inside this project's own model server,
[src/inference_server.py](src/inference_server.py). Everything else stays the same: the
Extension Development Host, `@tinyllama` in a **Local** chat, and skills like `/pirate`
([04_readme-harness.md](04_readme-harness.md)). That makes it a fair comparison. The skill
instructions and the question are identical; only the model changes. So you can show that a
better model follows the same skill much better.

Why Qwen should do better:
- It's trained to follow instructions more closely, so "always answer like a pirate, and here's
  an example" actually changes its style, rather than being ignored or repeated back.
- It can read 32,768 tokens instead of 2,048, so the skill, your question and the conversation
  all fit without the server trimming anything.
- Even **Qwen 2.5 0.5B**, half TinyLlama's size, usually follows skills better. That's a useful
  point to make: newer training matters as much as size.

Neither model picks a skill by itself here. You choose it with `/pirate`, and VS Code adds it to
the request. The comparison is purely about how well each model **follows** the skill.

### Step 1 – Record TinyLlama's answers first

1. Start the server as usual, still on TinyLlama:
   ```powershell
   .\venv_rag\Scripts\Activate.ps1
   python src/inference_server.py
   ```
   The terminal shows `Loading TinyLlama from …\models\TinyLlama`.
2. Open the Extension Development Host (**Terminal → Run Task… → Run TinyLlama Plugin**). In its
   Chat view, start a **new chat** and switch the session to **Local**.
3. Ask a few skill questions, e.g.:
   ```
   @tinyllama /pirate what is a stack?
   @tinyllama /yoda what is a queue?
   @tinyllama /shakespeare what is a linked list?
   ```
   Keep this chat open, or take a screenshot, to compare later. TinyLlama often gives only a hint
   of the style, ignores it, or repeats the instructions back.

### Step 2 – Download Qwen

With the environment activated:
```powershell
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
```
This is about 3 GB and needs roughly 6–7 GB of free RAM, because this server runs models
uncompressed on the CPU. For faster answers, use the smaller model instead (about 1 GB, and less
RAM than TinyLlama):
```powershell
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```
TinyLlama stays in `models/TinyLlama/`, so you can switch back at any time.

### Step 3 – Point the server at Qwen

In [src/inference_server.py](src/inference_server.py#L28-L34), change these three lines (use
`Qwen2.5-0.5B-Instruct` if you downloaded the smaller one):
```python
LLM_DIR = os.path.join(PROJECT_DIR, "models", "Qwen2.5-1.5B-Instruct")
MODEL_ID = "qwen2.5-1.5b"
CONTEXT_WINDOW = 4096
```
- `LLM_DIR` is the model folder the server loads.
- `MODEL_ID` is the name the server reports. It's only a label: the server doesn't check the
  name the plugin asks for.
- `CONTEXT_WINDOW` is how many tokens to allow per request. Qwen supports far more than 2,048;
  4,096 is a reasonable start on a CPU, as larger values use more memory and time.

Save the file, then restart the server: `Ctrl+C`, then `python src/inference_server.py`. The
startup message still says "Loading TinyLlama from …", because that wording is fixed in the code,
but the path at the end shows `…\models\Qwen2.5-1.5B-Instruct`. That path is what counts.

### Step 4 – Optional: show Qwen's name in the plugin

In the Extension Development Host, open Settings (`Ctrl+,`), search for **TinyLlama**, and set
`tinyllama.model` to `qwen2.5-1.5b`. This is only for display: the chat panel's header then shows
`Local model: qwen2.5-1.5b`. `@tinyllama` keeps its name, but it now talks to Qwen.

### Step 5 – Ask the same questions with Qwen

1. Start a **new chat** in the Chat view, and switch the session to **Local**. A new chat matters:
   otherwise the model sees TinyLlama's earlier answers in the conversation history.
2. Ask exactly the same questions as in step 1:
   ```
   @tinyllama /pirate what is a stack?
   @tinyllama /yoda what is a queue?
   @tinyllama /shakespeare what is a linked list?
   ```
3. Compare with TinyLlama's answers. Qwen should stay in character more consistently, use the
   skill's catchphrases, and still answer the question correctly.

**In the server terminal,** each request shows its prompt size in tokens. The skill adds the
same number of tokens for both models, so you can point out that the input really was the same.

### Switching back to TinyLlama

Change the three lines back, then restart the server:
```python
LLM_DIR = os.path.join(PROJECT_DIR, "models", "TinyLlama")
MODEL_ID = "tinyllama"
CONTEXT_WINDOW = 2048
```

### Notes

- **Speed:** this server runs on the CPU, even though your computer has a GPU (see "Using your
  GPU"). Qwen 1.5B is somewhat slower than TinyLlama there; 0.5B is faster. Ollama (option A)
  uses the GPU without any changes.
- **Qwen 2.5, not Qwen 3:** Qwen 3 needs a newer version of `transformers` than the one pinned in
  [requirements.txt](requirements.txt). Ollama and Foundry Local don't have that limit.
- **Agent mode:** this server doesn't implement tool calling, so Copilot's Agent mode still can't
  use it, and the model still can't pick a skill by itself. For that, use Ollama (section 5).
- **Envie Streamlit app (optional):** to use Qwen there too, change these lines in
  [src/main.py](src/main.py#L82-L83) and restart the app:
  ```python
  LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
  LLM_DIR = os.path.join(PROJECT_DIR, "models", "Qwen2.5-1.5B-Instruct")
  ```

---

## 4. Pointing the TinyLlama plugin at the new model

The plugin ([03_readme-plugin-startup.md](03_readme-plugin-startup.md)) works with any of the three. In
the **[Extension Development Host]** window, open Settings (`Ctrl+,`), search for **TinyLlama**,
and set:

| Setting | Ollama | Foundry Local | This project's server (C) |
|---|---|---|---|
| `tinyllama.serverUrl` | `http://localhost:11434/v1` | the address from `foundry service status`, plus `/v1` | `http://127.0.0.1:8000/v1` (unchanged) |
| `tinyllama.model` | `qwen3.5:4b` (the size you pulled) | the `id` from `/v1/models` | the `MODEL_ID` you set, e.g. `qwen2.5-1.5b` |

The chat panel's header then shows the new model, e.g. `Local model: qwen3.5:4b`. Changes apply
to the next question. If the panel was already open, close and reopen it to refresh the header.
`@tinyllama` in the Chat view uses the same settings. Despite its name, it then talks to Qwen.

With Ollama or Foundry, `src/inference_server.py` isn't needed. You can stop it.

**Shortcut for the test window:** the Extension Development Host opens
[.vscode/plugin-test.code-workspace](.vscode/plugin-test.code-workspace), and settings in that
file apply to the test window only. It's currently set up for Ollama:
```json
"settings": {
  "tinyllama.serverUrl": "http://localhost:11434/v1",
  "tinyllama.model": "qwen3.5:4b"
}
```
Change `tinyllama.model` if you pulled a different size. To go back to TinyLlama in
`src/inference_server.py`, delete the two settings (or set them to `http://127.0.0.1:8000/v1` and
`tinyllama`). Then close the test window and run **Terminal → Run Task… → Run TinyLlama Plugin**
again.

---

## 5. Using Qwen in Copilot Chat

See **"Quick start: Qwen inside Copilot Chat"** at the top of this guide. It requires **Ask** mode;
Agent mode, where the model could pick skills by itself, stalled with `qwen2.5:7b` on this
computer. For Ollama, it
uses the **Ollama for VS Code** extension. VS Code also lists **Foundry Local** under
**Manage Models… → Add Models**, so the same steps apply to Foundry Local: choose **Foundry Local**
instead of **Ollama**.

---

## 6. Proving it's still local

The demos in [03_readme-plugin-startup.md](03_readme-plugin-startup.md) (section 4) still apply,
with a different process to watch:
- **Ollama:** the model runs in `ollama.exe` (or `ollama_llama_server.exe`). `ollama ps` shows
  what's loaded. In Resource Monitor, its connections are on port 11434 and loopback only.
- **Foundry Local:** look for the Foundry processes in Task Manager. `foundry service status`
  shows the local address it serves on.
- **Turn the runner off** (quit Ollama from the tray icon, or `foundry service stop`) and ask
  again. The plugin can't reach the model any more.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ollama` or `foundry` is "not recognized" | Close and reopen the terminal (or VS Code) after installing, so the new program is found. |
| The plugin says it couldn't reach the server | Check `tinyllama.serverUrl` ends in `/v1`, and that Ollama is running (tray icon) or `foundry service status` shows it running. |
| The plugin shows an HTTP 404 or 400 error | `tinyllama.model` doesn't match a downloaded model. Copy the exact `id` from `<server>/v1/models`. |
| Answers are very slow | Check the GPU is being used (`ollama ps` should show `100% GPU`). If it shows a CPU/GPU split, the model is too big for the GPU's memory: use a smaller one. Without a GPU, use `qwen2.5:0.5b` or `1.5b`. |
| `ollama ps` shows `100% CPU` although there's an NVIDIA GPU | Update the NVIDIA graphics driver, then restart Ollama (quit it from the tray icon and start it again). |
| Option C: `torch.cuda.is_available()` prints `False` | The CPU-only PyTorch is still installed (`torch.__version__` ends in `+cpu`). Run **both** commands from "Using your GPU": `pip uninstall -y torch`, then `pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121`. |
| Ollama doesn't appear under **Add Models** | Install the **Ollama for VS Code** extension (`Ollama.ollama`) and reload the window. Make sure you're in your normal VS Code window, not the Extension Development Host, which turns extensions off. Your account or organization may also not allow adding models. |
| Option C fails to load the model | Check the folder name in `LLM_DIR` matches the one you downloaded into, and that you used Qwen **2.5**. |

## Sources

- [AI language models in VS Code](https://code.visualstudio.com/docs/agent-customization/language-models): the built-in Ollama provider is deprecated in favor of the official Ollama extension
- [Ollama (BYOM) – deprecated](https://marketplace.visualstudio.com/items?itemName=cgaspard.ollama-byom), which points to the official **Ollama for VS Code** (`Ollama.ollama`)
- [Ollama download](https://ollama.com/download) and [installing Ollama on Windows 11](https://pureinfotech.com/install-ollama-run-local-ai-models-windows-11/)
- [Ollama installation and OpenAI-compatible API](https://oneuptime.com/blog/post/2026-02-02-ollama-installation-configuration/view)
- [Foundry Local winget package](https://winstall.app/apps/Microsoft.FoundryLocal) and [Foundry Local overview](https://learn.aguidetocloud.com/slm-local-ai/foundry-local/)
- [Foundry Local: CLI, OpenAI-compatible endpoint and models (Japanese)](https://zenn.dev/headwaters/articles/6f6e97dadebe3b)

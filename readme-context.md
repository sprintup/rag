# What VS Code Sends to the Model: Inspecting the Hidden Context

When you type *"Hello?"* in Copilot's Chat view, the model receives far more than one word. In this
project, Ollama's log showed each request from Copilot's agent harness at about **9,000 tokens**,
even for "Hello?". This guide shows how to see exactly what's in those requests, and explains what
the parts are.

Commands below are from **VS Code 1.142**. Open each with `Ctrl+Shift+P` and type its name.

---

## 1. What's in a request

A request from Copilot Chat is a list of messages, plus a list of tools. Typically:

| Part | What it is | Roughly how big |
|---|---|---|
| **System message** | Hidden instructions: who the assistant is, how to behave, how to format answers, how to use tools. | Several thousand tokens |
| **Tool definitions** | In **Agent** mode, a description of every tool the model may call: read file, search, run in terminal, edit, ask you a question, plus tools from MCP servers (e.g. GitKraken, GitHub, Pylance). Each comes with a JSON schema for its arguments. | Often the largest part |
| **Workspace context** | Your operating system, the open folder, sometimes a summary of the project's structure. | A few hundred to a few thousand tokens |
| **Customizations index** (`vscode.customizations.index`) | The list of available **skills** (`.github/skills/*/SKILL.md`), custom instructions and prompt files: each one's name, description and file path, so an agent can open them when needed. | Grows with each skill |
| **Conversation history** | Your earlier messages and the model's answers in this chat. | Grows with each turn |
| **Attachments** | Files, selections or images you attached. | Depends on the files |
| **Your message** | What you typed. | Usually tiny |

**Ask** mode sends no tool definitions, which is why it's much smaller. That's why Qwen answered
quickly in Ask mode, but stalled in Agent mode.

**Messages you may see while it works:**
- **"Optimized tool selection":** there are too many tools to send them all, so VS Code first picks
  a subset that seems relevant, which is an extra step before the real request.
- **"Compacted conversation":** the conversation no longer fits the model's context window, so VS
  Code replaced older turns with a shorter summary. With a small context window, this happens
  after only a few messages.

---

## 2. Chat Debug View: every request, in full

This is the main tool. It records every request VS Code's chat sends to a model, and the
response.

1. Send a message in the Chat view as usual.
2. Run **Developer: Show Chat Debug View**.
3. A list of recent requests appears, newest first. Each chat message can produce several
   requests: the main one, plus small background ones such as generating the chat's title.
4. Click a request to see:
   - the **model** it went to (e.g. `qwen2.5:7b`), and how long it took;
   - each **message** in the prompt: the system message, your message and the history;
   - the **response**.
5. Right-click a request (or use the view's `…` menu) for more:
   - **Show Raw Request Body:** the exact JSON sent to the model, including every tool
     definition. This is the most complete view of the hidden context.
   - **Show Tools:** list the tools that were offered.
   - **Export as…:** save one request to a file.
6. To save everything at once, use **Export All Prompt Logs as JSON…** or **Export All as
   Archive…** from the view's menu. Open the JSON file in VS Code to search through it, e.g. for
   `pirate` to find the skills list, or `"name":` to count the tools.

The labels in the view may differ slightly between versions. If a request isn't listed, the
message probably went through the agent harness instead (section 3).

---

## 3. Agent harness logs (Copilot sessions)

Chats in a **Copilot** session (the "new Copilot experience", where the session selector shows
**Copilot** or **Interactive**) run in a separate *agent host* process. Its requests are recorded in
the **Agent Debug Log** instead.

- **Show Agent Debug Logs** / **Open Agent Debug Logs:** view the log for recent agent sessions:
  each turn, the tool calls the model made, and the model requests.
- **Export Agent Debug Log…:** save one session's log to a file.
- **Developer: Export Agent Host Debug Logs…:** save all of the agent host's logs in one archive.
- If the agent debug log is empty, turn on the setting **`chat.agentHost.agentDebugLog.enabled`**
  in Settings (`Ctrl+,`) and try again.

The agent host's plain log is also on disk:
`%APPDATA%\Code\logs\<newest folder>\agenthost.log`. It's less detailed, but useful. For
example, when the Ollama model stalled, this log showed lines like
*"Wired 1 BYOK model(s) … via loopback proxy"* and *"sendMessage called: \"Hello?\""*, which showed
how far the message got.

---

## 4. Ollama's side: what actually reached the model

The tools above show what VS Code *intended* to send. Ollama's log shows what *arrived*, and what
Ollama did with it.

**Without changing anything,** the normal log already shows request sizes:
```powershell
Get-Content "$env:LOCALAPPDATA\Ollama\server.log" -Tail 20 -Wait
```
A line like
```
level=WARN msg="truncating input prompt" limit=2050 prompt=8995 keep=4 new=2050
```
means a request of **8,995 tokens** arrived, and only 2,050 fit, so the rest was dropped.

**To see the full prompt text,** turn on Ollama's debug logging:
1. In PowerShell:
   ```powershell
   setx OLLAMA_DEBUG 1
   ```
2. Quit Ollama (tray icon → Quit) and start it again from the Start menu.
3. Send a chat message, then search the log for the request:
   ```powershell
   Select-String -Path "$env:LOCALAPPDATA\Ollama\server.log" -Pattern "chat request" | Select-Object -Last 1
   ```
   In debug mode, Ollama logs each chat request, including the prompt text after it has been put
   into the model's chat format. The exact wording of the log line varies between Ollama versions.
   If `chat request` finds nothing, open the log and look for `level=DEBUG` lines with `prompt=`.
4. Turn debug logging off again when you're done, since it makes the log very large:
   ```powershell
   setx OLLAMA_DEBUG 0
   ```
   and restart Ollama.

---

## 5. This project's own server

When you use the TinyLlama plugin with `src/inference_server.py`, the server's terminal shows each
request's size:
```
Request: 3 message(s), stream=True
  prompt: 412 tokens (0 older message(s) dropped to fit)
```
The plugin builds its requests itself ([src/vscode-extension/extension.js](src/vscode-extension/extension.js)),
so there's nothing hidden. Its system prompt, `SYSTEM_PROMPT`, is near the top of the file, and
`handleChatRequest()` shows exactly what's added: your message, the conversation's earlier
`@tinyllama` turns, and any attached files. It deliberately skips VS Code's
`vscode.customizations.index`, to save TinyLlama's small context window.

---

## 6. Things to look for

- **How big is the system message compared with your question?** For "Hello?", typically over
  99% of the request is hidden context.
- **How many tools are offered in Agent mode?** Turning off MCP servers in the tools menu (the
  sliders icon next to the model name) shrinks the request noticeably.
- **The skills list:** search the request for `pirate`. You'll find the skill's name, description
  and file path, but not its instructions. The model has to call a tool to read `SKILL.md`, and
  that tool call then appears as a separate step.
- **Ask vs. Agent:** send the same message in both modes and compare the request sizes.
- **Compaction:** after "Compacted conversation", the history in the next request is replaced by a
  summary.

---

## 7. Where the `/` commands come from

Typing `/` in the Chat view lists skills and prompts such as `/create-skill`, `/pylance-docs` and
`/pirate`. Each one is an ordinary text file on your disk, in the same `SKILL.md` (or `.prompt.md`)
format as this project's skills. There are three sources:

**1. VS Code's built-in skills and prompts**, such as `/create-prompt`, `/create-skill`,
`/create-agent`, `/create-hook`, `/create-instructions`, `/chronicle`, `/troubleshoot` and
`/migrate-customizations`. They ship inside VS Code's Copilot extension:
```
%LOCALAPPDATA%\Programs\Microsoft VS Code\<version folder>\resources\app\extensions\copilot\assets\prompts\
├── skills\                 ← one folder per skill, each with a SKILL.md
│   ├── create-skill\SKILL.md
│   ├── create-prompt\SKILL.md
│   ├── troubleshoot\SKILL.md
│   └── …  (agent-customization, chronicle, init, project-setup-info-…, and more)
├── plan.prompt.md          ← prompt files
└── chronicle-*.prompt.md
```
`<version folder>` is a short code such as `6d62b1d5da`, and changes with every VS Code update.
Look in `%LOCALAPPDATA%\Programs\Microsoft VS Code\` for the current one.

**2. Skills from installed extensions**, such as Pylance's `/pylance-docs`,
`/pylance-refactoring`, `/python-type-inference`, `/python-write-stubs` and
`/python-add-type-annotations`:
```
%USERPROFILE%\.vscode\extensions\<extension folder>\skills\<name>\SKILL.md
```
For example, `ms-python.vscode-pylance-2026.4.1\skills\pylance-docs\SKILL.md`.

**3. This project's own skills:** `.github/skills/<name>/SKILL.md`, e.g. the pirate, yoda,
shakespeare, cowboy and sportscaster skills.

**To open one** in VS Code, e.g. from a terminal:
```powershell
code "$env:LOCALAPPDATA\Programs\Microsoft VS Code\6d62b1d5da\resources\app\extensions\copilot\assets\prompts\skills\create-skill\SKILL.md"
```
Treat the files in sources 1 and 2 as **read-only**: VS Code and extension updates replace them, so
any edits would be lost. To change how one behaves, copy it into `.github/skills/` under a new
name, and edit the copy.

### The header of a skill file

Each `SKILL.md` starts with a header between `---` lines. For example,
`create-skill/SKILL.md` starts with:
```yaml
---
name: create-skill
description: 'Create a reusable skill (SKILL.md) that packages a workflow.'
argument-hint: What should this skill produce?
disable-model-invocation: true
---
```

| Field | What it does |
|---|---|
| `name` | The command name, as in `/create-skill`. It matches the skill's folder name. |
| `description` | Shown next to the command in the `/` list. In **Agent** mode, it's also what the model reads to decide whether the skill fits a request, so it should say *when* to use the skill (e.g. *"Use when the user asks for an answer like a pirate."*). |
| `argument-hint` | A hint for what to type after the command, e.g. *"What should this skill produce?"*. |
| `disable-model-invocation` | When `true`, the agent **never picks this skill by itself**. It only runs when you type `/name`. VS Code describes it as being *"for workflows you want to trigger manually with /name"*. Leave it out (or `false`) for skills the model should be able to choose. |

This project's personality skills don't set `disable-model-invocation`, so in Agent mode the
model may pick them on its own. Adding `disable-model-invocation: true` to one makes it a
manual-only command.

---

## 8. Exploring the whole Copilot SDK: the Copilot Harness Builder

**<https://patniko.github.io/copilot-sdk-visualizer/>**

This site, titled **Copilot Harness Builder**, is an interactive map of the **GitHub Copilot
runtime / SDK**: the engine underneath Copilot's agent harness in VS Code (the "Copilot" session
type, and the Copilot CLI). Its source is the public repository
[patniko/copilot-sdk-visualizer](https://github.com/patniko/copilot-sdk-visualizer), described as
*"Visualize the Copilot Runtime/SDK and build your own harness."*

It runs entirely in your browser. In its README's words, it *"does not start agents, call model
services, execute tools, or store credentials"*. You click through the settings a harness has, read
what each one does, and can export a plan or a starter project. Its catalogs are snapshots of a
particular SDK version, so newer VS Code releases may differ.

### Finding your way around

The left sidebar has these sections, in order:

| Section | Subtitle on the site | What you'll find |
|---|---|---|
| **Overview** | How the harness fits together | The big picture. Start here. |
| **Runtime** | What the engine gives you | A map of the runtime's capabilities, and an **"Illustrative turn walkthrough"** that traces one model → tool → model turn. |
| **Base Profile** | Starting profiles | Preset starting configurations. |
| **Prompt** | Behavior & instructions | The **system message**: whether to append to, customize or replace the built-in one, plus a searchable **built-in prompt reference** that shows its sections. |
| **Tools** | Inventory & implementations | **"Built-in tools"**: a searchable catalog of the runtime's tools, filterable by **capability category**. |
| **Context & packs** | Inputs & discovery | **Skills**, **Skill directories**, **Plugin directories**, **Configuration discovery** and **Host Git operations**. |
| **Agents** | Roles & delegation | Sub-agents and how work is delegated. |
| **Models & identity** | Providers & host credentials | Which model provider to use (including your own), and whose credentials the harness runs with. |
| **Policy & state** | Permissions & lifecycle | Permissions, hooks and session state. |
| **Build & run** | Language, files & host setup | Export a bootstrap project for a chosen language. |
| **Host & deploy** | Where your agent runs | Local, container and cloud hosting options. |
| **Learn / reference** | Source-backed boundaries | A **"Configuration catalog"** of every setting, and **"Boundaries & gaps"**. |

### Where the `/` and `#` commands fit

The site doesn't have a page that lists VS Code's `/` or `#` commands by those names. Those are
VS Code chat features, and the site describes the runtime underneath them. But you can find the
machinery each one uses:

**`/` skills (e.g. `/pirate`, `/create-skill`):**
- **Context & packs → Skills.** The switch that lets the runtime load skills at all. The site says
  it permits *"built-in and directory-based skill loading; selecting the skill tool is a separate
  choice."*
- **Context & packs → Skill directories.** Where skill folders are found. In VS Code, that includes
  this project's `.github/skills/`.
- **Tools → Built-in tools → search for `skill`.** The **`skill`** tool is what the model calls to
  load a skill's `SKILL.md`. That's the step behind *"Skill Invoked: pirate"* in Agent mode, and
  why a model without tool calling (TinyLlama) can't pick skills by itself.

**`#` references (e.g. `#file`, `#codebase`, `#fetch`, `#changes`):**
- **Tools → Built-in tools.** Search the catalog for the tools that do the same kinds of lookup:
  `read_file` and `view` (reading files, like `#file`), `semantic_search`, `semantic_code_search`,
  `grep_search` and `file_search` (searching the project, like `#codebase`), and `web_fetch`
  (fetching a web page, like `#fetch`). Each entry shows the tool's description, which is the same
  kind of text the model receives in Agent mode.
- **Context & packs → Host Git operations.** Supplies Git context such as branch and status, like
  `#changes`.
- **Context & packs → Configuration discovery.** How the runtime picks up project configuration,
  e.g. instructions files.

**The hidden prompt (section 1 of this guide):**
- **Prompt → "See the built-in prompt reference".** The built-in system message, split into named
  sections. VS Code's agent host log showed the same structure on this computer:
  `Resolved system message: mode=customize sections=[identity:replace, tool_instructions:append]`.
  That's the **customize** mode, with the **identity** section replaced and the
  **tool_instructions** section appended to, exactly the options shown under **Prompt**.

**A suggested route** for exploring the whole SDK: **Overview** → **Runtime** (play the turn
walkthrough) → **Prompt** (open the built-in prompt reference) → **Tools** (search for `skill`,
`read_file`, `web_fetch`) → **Context & packs** (Skills) → **Learn / reference** (Configuration
catalog).

The mapping between VS Code's `/` and `#` commands and these runtime tools is an interpretation
based on what each does. The site doesn't state it, and VS Code's chat may implement some `#`
references itself rather than through these exact tools.

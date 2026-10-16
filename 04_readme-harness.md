# TinyLlama in VS Code – Extension Development Host and Copilot Chat

This guide covers two things:
- **Opening the Extension Development Host**: the test window of VS Code that loads the TinyLlama
  plugin without installing it.
- **Using `@tinyllama` in Copilot's Chat view** (the chat on the left), including **skills** such
  as `/pirate`.

For the plugin's own chat (`Ctrl+Shift+P` → **TinyLlama: Open Chat**, the chat on the right, or the
**TinyLlama** tab), the plugin's files, and how the plugin talks to the model, see
[03_readme-plugin-startup.md](03_readme-plugin-startup.md).

**Before you start:** the model server must be running. In a terminal, from the project folder:
```powershell
.\venv_rag\Scripts\Activate.ps1
python src/inference_server.py
```
Wait for `Inference server running at http://127.0.0.1:8000/v1`, and leave the terminal open.
(First time? Follow steps 1 and 2 of [01_readme-setup.md](01_readme-setup.md) to install everything
and download the model.)

---

## 1. Opening the Extension Development Host

1. Open this project folder in VS Code.
2. Choose **Terminal → Run Task… → Run TinyLlama Plugin**.
3. A second VS Code window opens, with **[Extension Development Host]** in its title bar. Use
   this window for everything below.
   - The notice *"All installed extensions are temporarily disabled"* is expected. Close it, and
     **don't** click "Reload and Enable Extensions".
   - The Explorer shows the project's files. It opens them through
     [.vscode/plugin-test.code-workspace](.vscode/plugin-test.code-workspace), because VS Code
     won't open the same folder directly in two windows.

After editing the plugin's code, run **Developer: Reload Window** (`Ctrl+R`) in this window to
load the new version.

---

## 2. How the plugin is "installed" without installing it

Normally, installing an extension copies it into your extensions folder
(`%USERPROFILE%\.vscode\extensions\`), where it loads in every VS Code window. This plugin is
never copied there. Instead, the task in [.vscode/tasks.json](.vscode/tasks.json) starts VS Code
like this:

```
code --new-window --disable-extensions --extensionDevelopmentPath="<project>/src/vscode-extension" "<project>/.vscode/plugin-test.code-workspace"
```

- **`--extensionDevelopmentPath`** tells this one new window to load an extension straight from a
  folder. VS Code reads `package.json` in that folder, registers the commands, chat participant
  and settings it declares, and runs `extension.js` from there. This is the **Extension
  Development Host**, the standard way to try an extension while developing it.
- **`--disable-extensions`** turns off your normally installed extensions in that window, so only
  the plugin loads, plus VS Code's built-in features such as Copilot Chat. That's why the
  "temporarily disabled" notice appears. It also avoids a crash that some extensions cause in
  this mode.
- **The last argument** opens the project, through the workspace file, so you can use its files.
- **Nothing is permanent.** Your main VS Code window, and every other window, are unaffected.
  Closing the Extension Development Host window unloads the plugin.

**Official docs on the Extension Development Host:**
- [Your First Extension](https://code.visualstudio.com/api/get-started/your-first-extension):
  Microsoft's starter tutorial. It shows pressing `F5` to open the Extension Development Host and
  trying a command in it, the same workflow used here.
- [Testing Extensions](https://code.visualstudio.com/api/working-with-extensions/testing-extension):
  explains `--extensionDevelopmentPath` and the other launch arguments.

**`F5` vs. the task.** `F5` runs the same thing from [.vscode/launch.json](.vscode/launch.json),
plus a debugger attached to the plugin, so you can set breakpoints in `extension.js`. On some
setups, this one included, the debugger crashes VS Code's extension host, and the window closes
immediately. The task runs without the debugger, so it avoids that.

---

## 3. Using `@tinyllama` in the Chat view

**About the tabs at the top of the Chat pane** (**Chat | Claude Code | Codex | TinyLlama**): each
tab is a **view** in the secondary side bar. **Chat** is VS Code's own; **Claude Code** and **Codex**
are added by Anthropic's and OpenAI's extensions, and **TinyLlama** by ours, all with the same
stable API: a `viewsContainers` → `secondarySidebar` entry in `package.json`, holding a webview.
This section uses the **Chat** tab, where our plugin also plugs in as the `@tinyllama` participant.
Section 7 explains how these tabs differ from a model in the picker.

The plugin registers a **chat participant** called `@tinyllama`: a name you can address in VS
Code's Chat view, the same view Copilot uses. Messages to `@tinyllama` go to your local TinyLlama,
not to GitHub's models.

In the **[Extension Development Host]** window:
1. Open the Chat view (`Ctrl+Alt+I`) and start a **new chat** (**+** at the top of the Chat view).
2. Below the chat input box, switch the session to **Local**, and set the mode to **Ask**.
   Both are required: in a Copilot session or in Agent mode, messages go to Copilot's agent
   instead, and fail or stall.
3. Type `@tinyllama` followed by your question, e.g. `@tinyllama what is a stack?`, and press
   Enter. While `@tinyllama` stays in the input box, follow-up questions keep going to it.
4. **To ask about a file,** attach it (the paperclip / **+**, or type `#file`), or use
   `#selection`. The plugin adds the text to the prompt, up to about 3,000 characters in total,
   because TinyLlama can read only ~2,048 tokens. A note in the reply says if anything was cut
   short or left out.
5. **To use a skill,** add it with `/`, e.g. `@tinyllama /pirate what is a stack?` (see
   section 5).

**Why "new chat" and "Local" matter.** Some VS Code users and sessions get an experimental
Copilot chat, the "new Copilot harness". It shows a *"You're using a new Copilot experience"*
banner, offers only **Agent** mode, and sends every message to Copilot's own agent. There,
`@tinyllama` fails with *"The selected model … is not available"*. A new chat on **Local** uses the
classic chat, which passes `@tinyllama` messages to the plugin. A new chat also starts with an
empty history, so TinyLlama doesn't copy earlier replies.

The Chat view is part of GitHub Copilot Chat, so this needs you to be signed in to GitHub with
Copilot available. The plugin's own TinyLlama tab ([03_readme-plugin-startup.md](03_readme-plugin-startup.md))
works without Copilot.

---

## 4. How `@tinyllama` works

```
Chat view (Local)                 extension.js                        src/inference_server.py
@tinyllama /pirate what is…  ──▶  handleChatRequest()  ──HTTP──▶   TinyLlama generates on CPU
reply appears in the chat   ◀──   stream.markdown()   ◀─stream──   models/TinyLlama/
```

1. **Registration.** When the plugin starts, `activate()` registers the participant with
   `vscode.chat.createChatParticipant` ([extension.js:55](src/vscode-extension/extension.js#L55)).
   Its name and description come from `chatParticipants` in
   [src/vscode-extension/package.json](src/vscode-extension/package.json).
2. **Your message arrives.** VS Code calls `handleChatRequest()`
   ([line 241](src/vscode-extension/extension.js#L241)) with your message, anything attached to it,
   and the earlier `@tinyllama` turns of the conversation.
3. **Attachments are read.** `readAttachments()` ([line 286](src/vscode-extension/extension.js#L286))
   reads attached files and selections, up to `MAX_ATTACHMENT_CHARS`
   ([line 282](src/vscode-extension/extension.js#L282)) in total, and puts them above your
   question.
4. **The request is sent** to the local server with the same `streamChat()` function the chat
   panel uses ([line 190](src/vscode-extension/extension.js#L190)).
5. **The reply streams back** and is written into the Chat view with `stream.markdown()`. Copilot's
   own models are never called.

---

## 5. Skills

The project includes a standard VS Code skill,
[.github/skills/pirate/SKILL.md](.github/skills/pirate/SKILL.md):
```markdown
---
name: pirate
description: Answer in pirate speak. Use when the user asks for an answer like a pirate.
---

You are a pirate. Always answer like a pirate: say "arrr", "matey", "ahoy" and "ye" in every answer.

Example: "Arrr, matey! A stack be like a pile o' plates. Ye add to the top, and ye take from the top!"
```

**More personalities** are in the same folder, so you can show several skills in a row:

| Skill | Try | Style |
|---|---|---|
| [pirate](.github/skills/pirate/SKILL.md) | `@tinyllama /pirate what is a stack?` | "Arrr, matey!" |
| [shakespeare](.github/skills/shakespeare/SKILL.md) | `@tinyllama /shakespeare what is a stack?` | "Forsooth, thou…" |
| [cowboy](.github/skills/cowboy/SKILL.md) | `@tinyllama /cowboy what is a stack?` | "Howdy, partner!" |
| [yoda](.github/skills/yoda/SKILL.md) | `@tinyllama /yoda what is a stack?` | "A pile of plates, a stack is." |
| [sportscaster](.github/skills/sportscaster/SKILL.md) | `@tinyllama /sportscaster what is a stack?` | "WHAT A PLAY, folks!" |

Each file ends with an **example answer**. Small models like TinyLlama copy examples much more
reliably than they follow a description, so the example does most of the work.

**Using a skill with `@tinyllama`:** in a new **Local** chat, type
`@tinyllama /pirate what is a stack?`. As you type `/`, VS Code suggests the available skills.
Put the `/` command right after `@tinyllama`, at the start of the message: in a **Local**
session, a `/` later in the message doesn't open the list of skills.

**The plugin has no skill code.** VS Code finds skills in `.github/skills/` and offers each one as
a `/name` command. When you choose one, VS Code itself includes the skill with the request it
hands to `@tinyllama`. The plugin passes it to the model along with your message. To add a skill,
create another folder, e.g. `.github/skills/spanish/SKILL.md` containing *"Always answer in
Spanish."*, and `/spanish` becomes available.

**How skills work in Copilot's agent harness**, for comparison:
1. The harness gives the model a list of available skills, each with its `name` and
   `description`.
2. The **model** decides a skill fits the request, e.g. because you asked for an answer "like a
   pirate".
3. The model **calls a tool** to read that skill's `SKILL.md`, and the chat shows
   *"Skill Invoked: pirate"*.

`@tinyllama` can't do steps 2 and 3. TinyLlama doesn't support **tool calling**, and the plugin
gives it no tools. So with `@tinyllama`, a skill is only used when **you** choose it with `/name`.
To see the model pick a skill by itself, use Copilot's **Agent** mode with a model that supports
tool calling. Copilot's built-in models do. A local one would need to be registered in Copilot
(section 6) and support tool calling, which TinyLlama doesn't. Qwen 2.5 does; see
[05_readme-models.md](05_readme-models.md) for running it with Ollama or Foundry Local.

**Tips:** keep questions short. TinyLlama is small, so it may only manage some "arrr"s and
"matey"s. Rehearse once before presenting.

---

## 6. Optional: the local model in Copilot's model picker

On a **personal** Copilot account, the same model server can also back Copilot Chat directly. In
Copilot's model picker, go to **Manage Models… → Add Models → Custom Endpoint**, and use base URL
`http://127.0.0.1:8000/v1` and model ID `tinyllama`, in **Ask** mode. This may need VS Code
Insiders, and organization-managed (Business/Enterprise) accounts usually can't add custom
models. `@tinyllama` avoids both limits.

---

## 7. Models vs. harnesses: the Chat, Claude Code and Codex tabs

**A model is the brain; the harness is the program around it.** The harness builds the prompt
(system prompt, instructions, skills, attached context), decides which tools to offer, runs the
tool calls, loops until the task is done, and manages the context window. The model only receives
text and sends text back.

- **A local model in the model picker** (TinyLlama here, or Qwen in Ollama in step 05) is a new
  brain inside **Copilot's** harness. You keep Copilot's skills, prompt files, instructions, MCP
  servers, Configure Tools and Ask/Agent modes; only the model changes.
- **The Claude Code and Codex tabs** are **different harnesses**, made by Anthropic and OpenAI,
  each with its own system prompt, tools, permissions and context handling, and its own models.
  Each extension adds its tab as a side bar view and draws its own chat UI in it. Claude Code, for example, reads
  `CLAUDE.md` and `.claude/`, not `.github/copilot-instructions.md` or `.github/skills`.

In short: picking a model **swaps the engine** in Copilot's car; the other tabs are **different
cars** parked in the same garage.

| | Copilot Chat + local model | Claude Code / Codex tab |
|---|---|---|
| Harness | Copilot's | Anthropic's / OpenAI's |
| Model | Any registered provider, e.g. Qwen in Ollama | The vendor's own models |
| Uses `.github/skills`, prompt files, Configure Tools | Yes | No, it has its own equivalents |
| Where the model runs | Your computer, if local | The vendor's cloud |

### Is there a "harness API"?

Not a single one. There are a few layers:

| Layer | What you plug in | Example |
|---|---|---|
| [Chat Participant API](https://code.visualstudio.com/api/extension-guides/ai/chat) | Your own `@name` that owns the whole turn: prompt, model call, reply. The closest thing to a harness API in VS Code. | `@tinyllama`, a very small harness of our own. That's why it has no skills or Agent mode. |
| [Language Model API](https://code.visualstudio.com/api/extension-guides/ai/language-model) (`vscode.lm`) | Lets an extension call the model the user picked | (not used here) |
| [Language Model Chat Provider API](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider) | A **model** in the picker | The Ollama extension |
| [Language Model Tools API](https://code.visualstudio.com/api/extension-guides/ai/tools) / MCP | **Tools** for Copilot's harness | The Fetch MCP server ([06](06_readme-mcp-agents.md)) |
| [Views](https://code.visualstudio.com/api/ux-guidelines/views) + [Webview API](https://code.visualstudio.com/api/extension-guides/webview) | A **tab** with your own UI, e.g. a whole agent | The Claude Code, Codex and TinyLlama tabs |
| Harness SDKs | A ready-made agent loop to build your own app on | Claude Agent SDK, GitHub Copilot SDK, OpenAI Agents SDK / Codex SDK |
| [Agent Client Protocol (ACP)](https://agentclientprotocol.com/) | A standard way for an editor to run any agent. Like MCP, but for whole agents. | Zed with Claude Code or Gemini CLI |

The first five are stable VS Code APIs that any extension can use. A tab is only a place to show
things: the Claude Code and Codex tabs run their own harness inside, while our TinyLlama tab is a
plain chat client with no harness at all (no tools, skills or agent loop). VS Code doesn't use ACP
for its tabs.

**For the class:** models plug in through **providers**, tools through **MCP**, and whole agents
through **harness SDKs** or **ACP**. Any extension can add a tab, but what's inside the tab, from a
plain chat client to a full agent, is up to the extension.

> These APIs and SDKs change quickly. Check the current documentation before presenting.

---

## Further reading

| Topic | Link |
|---|---|
| Building and running a first extension, and the Extension Development Host | [Your First Extension](https://code.visualstudio.com/api/get-started/your-first-extension) |
| Launch arguments such as `--extensionDevelopmentPath` | [Testing Extensions](https://code.visualstudio.com/api/working-with-extensions/testing-extension) |
| `@tinyllama` (a chat participant) | [Chat Participant API](https://code.visualstudio.com/api/extension-guides/ai/chat) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| The Extension Development Host closes immediately | You used `F5`, and the debugger crashed the extension host. Use **Terminal → Run Task… → Run TinyLlama Plugin** instead. To confirm, the newest `main.log` in `%APPDATA%\Code\logs\` shows "Extension host … crashed with code 134". |
| *"The selected model '…' is not available"* when using `@tinyllama` | The chat is using Copilot's experimental new harness (only **Agent** mode is offered). Start a **new chat** and switch the session to **Local** below the input box. |
| `@tinyllama` isn't offered in the Chat view | Check you're in the **[Extension Development Host]** window, and signed in to GitHub with Copilot available. Your organization may restrict chat extensions. |
| `/pirate` isn't suggested | Check that `.github/skills/pirate/SKILL.md` exists in the project the window has open. |
| Replies repeat earlier answers or instructions | Start a **new chat**. `@tinyllama` sends the conversation's history each time, and TinyLlama tends to copy it. |
| *"Couldn't reach the TinyLlama server…"* | The model server isn't running, or is still loading. Check its terminal. |
| Changes to `extension.js` don't show up | Run **Developer: Reload Window** (`Ctrl+R`) in the Extension Development Host. |

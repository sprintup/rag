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

### Are the Claude Code and Codex harnesses built on the Copilot SDK?

**No. Each is its vendor's own harness, and the same program as its command-line tool.** The
VS Code extension is mostly a window onto it. You can see this in the installed extensions'
folders (`%USERPROFILE%\.vscode\extensions\`):

| Tab | Extension | What it runs | Built on |
|---|---|---|---|
| **Claude Code** | `anthropic.claude-code` | `resources\native-binary\claude.exe`, the **Claude Code** program, as used in a terminal | Anthropic's own harness. Anthropic offers the same agent loop to developers as the **Claude Agent SDK**. |
| **Codex** | `openai.chatgpt` | `bin\windows-x86_64\codex.exe`, the **Codex CLI** | OpenAI's own harness. The Codex CLI is open source ([openai/codex](https://github.com/openai/codex)). |
| **Chat** (Copilot) | built into VS Code | Copilot's agent | GitHub's harness. GitHub offers its agent runtime to developers as the **GitHub Copilot SDK**. |

So there are three separate harnesses from three companies, each with its own system prompt, tools,
permissions, context handling and settings files. None is built on another. What they share is
the **window** (VS Code, which gives each a side bar tab) and the **pattern**: a loop that sends the
conversation and tool definitions to a model, runs the tools it asks for, and repeats.

Even the same model behaves differently in each tab, because each harness sends it different
instructions and tools.

### What the three harnesses share, and what they don't

Each harness looks for skills, instructions and settings in **its own places**. `~` is your user
folder (`%USERPROFILE%`); the other paths are in the workspace.

| | Copilot (Chat tab) | Claude Code | Codex |
|---|---|---|---|
| **Main instructions file** ([AGENTS.md spec](https://agents.md): from OpenAI Codex and others, now the Agentic AI Foundation) | `.github/copilot-instructions.md`; also `AGENTS.md` and `CLAUDE.md` (settings `chat.useAgentsMdFile`, `chat.useClaudeMdFile`) | `CLAUDE.md`, `CLAUDE.local.md`, `~/.claude/CLAUDE.md` | `AGENTS.md`, `AGENTS.override.md`, `~/.codex/AGENTS.md` |
| **Instructions for some files only** | `.github/instructions/*.instructions.md` (`applyTo`), `.claude/rules/`, `~/.copilot/instructions/` | `.claude/rules/` | `AGENTS.md` in a subfolder |
| **[Skills](https://agentskills.io/specification)** (`SKILL.md`; open standard from Anthropic) | `.github/skills/`, `.claude/skills/`, `.agents/skills/`; in `~`: `~/.copilot/skills/`, `~/.claude/skills/`, `~/.agents/skills/` | `.claude/skills/`, `~/.claude/skills/` | `.agents/skills/`, `~/.codex/skills/` |
| **Custom agents** | `.github/agents/*.agent.md`, `.claude/agents/`, `~/.copilot/agents/` | `.claude/agents/` (subagents) | `.codex/agents/` |
| **Orchestration** (see below) | Handoffs (`handoffs:` in `.agent.md`), subagents (`runSubagent`) | Subagents, experimental agent teams | Subagents (`spawn_agent`), multi-agent threads |
| **Reusable prompts** (`/name`) | `.github/prompts/*.prompt.md` | `.claude/commands/*.md`, or skills | Skills |
| **[MCP](https://modelcontextprotocol.io/specification/latest) servers** (from Anthropic, now the Agentic AI Foundation) | `.vscode/mcp.json` | `.mcp.json`, `~/.claude.json` | `[mcp_servers]` in `~/.codex/config.toml` |
| **Hooks** | `.github/hooks/`, `~/.copilot/hooks/` | `hooks` in `.claude/settings.json` | `.codex/hooks.json` |
| **Settings and permissions** | VS Code settings (`chat.tools.*`), **Configure Tools** | `.claude/settings.json`, `.claude/settings.local.json`, `~/.claude/settings.json` | `.codex/config.toml`, `~/.codex/config.toml` (`approval_policy`, `sandbox_mode`) |
| **Chat history** | VS Code's workspace storage ([readme-sessions.md](readme-sessions.md)) | `~/.claude/projects/` | `~/.codex/sessions/` |
| **Models** | Any registered provider, incl. local ones | Anthropic's | OpenAI's |

**Shared:**
- **The workspace:** the same files, the same git repository, the same terminal.
- **The skill format.** `SKILL.md` follows the open [Agent Skills](https://agentskills.io) format, so
  one skill folder works in all three, as long as it's in a folder each one reads.
  `.agents/skills/` is read by Copilot and Codex, and `.claude/skills/` by Copilot and Claude Code.
- **The MCP protocol**, but not the configuration: each harness has its own list of servers.
- **The window:** VS Code, with one side bar tab each.

**Not shared:** history, settings, permissions and tool approvals, the system prompt, and the
models. Approving a command in one tab doesn't approve it in another.

**For this project:** the pirate skill and the other files in `.github/` are only seen by Copilot.
To use the pirate skill in Claude Code too, it would also need to be in `.claude/skills/`; for
Codex, in `.agents/skills/`.

The **Agentic AI Foundation** is part of the Linux Foundation and looks after open standards for AI
agents, so that no single company controls them. Anthropic donated MCP to it in December 2025.

> These locations come from the installed versions (VS Code 1.142, Claude Code 2.1.301, Codex
> extension 26.1007), found by searching each program for the paths it uses. They change often,
> so check each tool's documentation for the current list.

### `@`, `/` and `#`: each harness defines its own

The operators look the same in every tab, but they aren't a standard: **each harness decides what
they mean.** In Copilot's Chat view they come from VS Code's chat APIs; in Claude Code and Codex,
from each program's own input box.

| Type | Copilot (Chat tab) | Claude Code | Codex |
|---|---|---|---|
| **`@`** | A **chat participant**: *who* answers, e.g. `@tinyllama`, `@vscode`, `@terminal`. Extensions add them with the [Chat Participant API](https://code.visualstudio.com/api/extension-guides/ai/chat). | A **file or folder** to include (*"@ for file paths"*), or a subagent (`@agent-<name>`) | A **file** to include |
| **`/`** | A **skill**, **prompt file** or built-in command, e.g. `/pirate`, `/word-of-the-day`, `/create-skill` | A built-in command (`/clear`, `/compact`, `/init`, …), a custom command or a skill | A built-in command (`/model`, `/approvals`, `/compact`, …) |
| **`#`** | **Context or a tool**: `#file`, `#codebase`, `#changes`, `#fetch`, MCP tools | No special meaning | No special meaning |
| **Other** | | `!` runs a shell command (*"! for shell mode"*) | `$` picks a skill, e.g. `$pirate` |

So `@tinyllama` only works in the Chat tab, because only VS Code's harness has chat participants,
and `/pirate` only lists skills that harness reads (see the table above). Where the `/` commands
in the Chat view come from is explained in section 7 of [readme-context.md](readme-context.md).

### Orchestration: handoffs, subagents and teams

Agents can work together in three ways:
- **Handoff:** switch to another agent in the **same** chat, and the conversation goes with it.
  It's sequential, and usually started by you, e.g. Plan → Implement.
- **Subagent:** the main agent delegates a task to another agent with its **own, fresh** context
  window. The subagent works alone and sends back only a summary, which keeps the main conversation
  short.
- **Team:** several agents work **in parallel** and message each other, with one as the leader.

How agents pass work and messages in each pattern, what each agent gets to see, and the 2026
Hugging Face incident, in which test agents coordinated through a message board they set up
themselves: [readme-orchestration.md](readme-orchestration.md).

| | Copilot (Chat tab) | Claude Code | Codex |
|---|---|---|---|
| **Handoffs** | Yes: a `handoffs:` list in a `.agent.md` file shows buttons after a reply, which switch to the next agent with a prepared prompt | No | No user-facing handoffs |
| **Subagents** | Yes: the `runSubagent` tool, which an agent can use to delegate | Yes: agents in `.claude/agents/`, started by the main agent through its Agent tool | Yes: `spawn_agent`, limited by settings such as `agents.max_concurrent_threads_per_session` |
| **Teams** | Not found | Experimental "agent teams": a team leader and teammates, turned on with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | A `multi_agent_v2` feature: several threads per session |
| **Where agents are defined** | `.github/agents/*.agent.md`; also reads `.claude/agents/` | `.claude/agents/*.md` | `.codex/agents/` |

**Handoffs only work in Copilot.** `handoffs:` is a field in VS Code's `.agent.md` format, and the
buttons are part of VS Code's Chat view. Claude Code reads its own folder with its own fields, and
Codex has no buttons for it. Copilot does read `.claude/agents/`, so one agent file can be shared,
but a handoff in it only works in Copilot. This project's **Idea Generator** agent
([06_readme-mcp-agents.md](06_readme-mcp-agents.md)) uses no handoffs: it's a single agent with
only the `fetch` tool.

**Orchestrating across harnesses** isn't built into any of them. The usual workarounds:
- **Hand over through files:** one agent writes `plan.md`, and you open another harness and ask it
  to "implement `plan.md`". This is simple, visible and works everywhere.
- **One harness as another's tool:** Claude Code can run as an MCP server (`claude mcp serve`), so
  another harness can call it like any MCP tool.
- **Your own harness on top**, calling each one as a step
  ([readme-custom-harness.md](readme-custom-harness.md)).

**⚠ Security:** each subagent or teammate is another model with tools, and an agent file can give
its subagents tools you didn't approve for the main agent. Check the `tools:` list of **every**
agent file involved, not just the one you started. Handoff prompts and agent files from public
collections are instructions the model obeys, just like skills.

> Found by searching the installed programs (VS Code 1.142, Claude Code 2.1.301, Codex extension
> 26.1007), not by running them. Teams in Claude Code and Codex are experimental.

### Background tasks: what runs while you're not watching

"Background" means two different things in a harness:
1. **Background work:** commands or whole agents that keep running while you carry on, on your
   computer or in the cloud.
2. **Background model jobs:** small jobs the harness gives a model without asking, such as writing
   a session's title or summarizing a long chat.

**1. Background work**

| | Copilot | Claude Code | Codex |
|---|---|---|---|
| **Long commands** (servers, builds, tests) | The agent can run them in the terminal while it carries on | The shell tool can run a command in the background (`run_in_background`), and **Ctrl+B** moves a running one there | Background terminals, which keep running between steps |
| **A background agent on your computer** | The **Copilot CLI** session type: in Copilot's words, *"a background agent running locally on your machine"* that *"works asynchronously in a Git worktree… isolated from your main workspace"*. Settings offer a choice of worktree or workspace, and automatic commits. | Subagents and tasks can run in the background while you keep chatting | Several agent threads per session (`spawn_agent`) |
| **An agent in the cloud** | The **Cloud** session type: *"the GitHub Copilot coding agent… works asynchronously in the cloud to implement changes and pull requests"* | Claude Code on the web; a session can be moved between the web and your computer ("teleport") | Codex cloud tasks (`codex cloud`, needs a ChatGPT sign-in) |

In each, you can start a task, carry on with something else, and come back for the result:
Copilot's background and cloud sessions appear in VS Code's **Sessions** list, and Claude Code and
Codex keep their own session lists in their tabs.

**2. Background model jobs**

| | Copilot | Claude Code | Codex |
|---|---|---|---|
| **Which model** | A **utility model**, which may be a GitHub cloud model even when your main model is local | A small, fast model, Haiku by default | Mostly the main model |
| **Jobs** | Titles, summarizing long chats, choosing tools; experimentally, keeping a to-do list (`backgroundTodoAgent`) | Titles, summaries and other quick jobs | Summarizing long chats; `/review` can use its own `review_model` |
| **Keeping it local** | `chat.byokUtilityModelDefault`, `chat.useCopilotModelsForUtilityModels` ([readme-safety.md](readme-safety.md), section 2) | `ANTHROPIC_DEFAULT_HAIKU_MODEL` pointed at the local model (section 8) | Nothing extra; it uses the model you set |

More on how each one keeps and trims the conversation:
[readme-sessions.md](readme-sessions.md), section 2.

**How to do it, per provider**

*GitHub Copilot (Chat tab):*
1. **Long command:** ask the agent to start it (e.g. *"start the Streamlit app"*); it runs in a
   terminal you can open, while the agent carries on.
2. **Background agent on your computer:** in the Chat view, click the session type under the input
   box (**Local**) and choose **Copilot CLI**. If asked, choose **Worktree** (a separate copy of the
   repository) rather than **Workspace**. Describe the task and send it. You can open another chat
   meanwhile; the session shows its progress in the **Sessions** list. When it's done, review its
   changes, then merge them into your branch. An existing Local chat can also be moved there with
   **Continue In**.
3. **Cloud agent:** the repository must be on GitHub, with the Copilot coding agent available for
   it. Choose **Cloud** as the session type (or **Continue In → Cloud**) and describe the task. It
   works on a new branch and opens a **pull request**, which you review on GitHub.

*Claude Code (tab or `claude` in a terminal):*
1. **Long command:** ask for it to run in the background, or press **Ctrl+B** while a command or
   subagent is running to move it there. `/tasks` lists what's running in the background,
   including subagents.
2. **Background agent on your computer:** ask Claude Code to *"use a subagent in the background
   to …"*, or press **Ctrl+B** on a running subagent; check on it with `/tasks`.
3. **Cloud agent:** in a terminal, `claude --cloud "<what to do>"` starts a cloud session and
   prints its ID (`--remote` is the older name). Follow it at `claude.ai/code` or in the Claude
   mobile app, and bring it back to your computer with `claude --teleport <session ID>`.

*Codex (tab or `codex` in a terminal):*
1. **Long command:** Codex keeps long commands running as **background terminals**; its status
   shows *"Background terminals"* while any are running.
2. **Background agent on your computer:** ask Codex to split the task between agents; it starts
   them with `spawn_agent`. This may need Codex's multi-agent feature turned on; check Codex's
   documentation for your version.
3. **Cloud agent:** `codex cloud` (after `codex login` with a ChatGPT account) lists your cloud
   tasks and starts new ones. Each runs in an OpenAI cloud environment set up for your GitHub
   repository, and its result comes back as a change you apply or a pull request.

**⚠ Data privacy: where your code and prompts go**

| | Your code | Your prompts and the agent's work | The model | Stays on your computer? |
|---|---|---|---|---|
| **Long command or background agent, local model** | Stays here | Session files on your disk | Ollama, here | **Yes**, if background model jobs are local too |
| **Long command or background agent, cloud model** | The parts the agent reads go to the vendor with each request | Sent to the vendor; session files also on your disk | The vendor's | **No** |
| **Cloud agent** (Copilot Cloud, `claude --cloud`, `codex cloud`) | The **whole repository** is copied to the vendor's machines and the agent runs there | Stored by the vendor; the result is a branch or pull request on GitHub | The vendor's | **No**. Nothing about it is local |
| **Background model jobs** (titles, summaries) | Parts of the conversation | Sent to whichever model does the job | Copilot: possibly a GitHub cloud model; Claude Code: Haiku unless changed; Codex: your model | Only if that model is local |

**What this means:**
- **A cloud agent can't be private.** It can't use a local model, and it needs the repository on
  the vendor's machines. Don't use one for code or data that must stay on your computer, or that
  your school or employer hasn't approved for that vendor.
- **Secrets travel with the repository.** Anything committed, such as an `.env` file, a token or
  personal data, goes to the cloud environment too. Cloud agents can also be given environment
  variables and internet access there; give them as little as possible. Uncommitted secrets aren't
  safe either: a local agent can read them (section 8, "Secrets", and safety rule 3).
- **Pull requests are visible to everyone who can see the repository.** On a public repository,
  the agent's branch, commits and messages are public.
- **A background agent runs while you're not watching.** It either waits for approvals or runs
  without asking, depending on its settings. Don't turn on automatic approval just to make it
  faster. Copilot's **Worktree** option keeps its changes away from your files until you merge
  them; with automatic commits on, it commits without asking.
- **Session files stay on your disk** for local and background sessions alike, with your full
  prompts ([readme-sessions.md](readme-sessions.md), section 4).
- **For a fully local setup,** use only local sessions with a local model, and point the
  background model jobs at it too (section 8).

> Found in the installed programs (VS Code 1.142, Claude Code 2.1.301, Codex extension 26.1007):
> Copilot's own descriptions where quoted, and Claude Code's and Codex's commands from their help
> text. None of the background or cloud modes was run here, so menu names may differ slightly.
> Check each vendor's documentation for what its cloud environment keeps and for how long.

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

**Building your own harness**, with the features the three above have, is explained in
[readme-custom-harness.md](readme-custom-harness.md): the files you need, which features VS Code's
Chat view gives you (`@`, `/`, `#`, drag and drop) and which you build yourself in your own tab, and
how to add the agent loop.

**For the class:** models plug in through **providers**, tools through **MCP**, and whole agents
through **harness SDKs** or **ACP**. Any extension can add a tab, but what's inside the tab, from a
plain chat client to a full agent, is up to the extension.

> These APIs and SDKs change quickly. Check the current documentation before presenting.

---

## 8. Local models in each harness, and the hidden context they send

### Which harnesses can use a local model

All three can, with **Ollama**. Ollama answers both OpenAI-style requests (`/v1/chat/completions`)
and, since recent versions, Anthropic-style ones (`/v1/messages`), and its `ollama launch` command
starts several harnesses already pointed at it. On this computer (Ollama 0.40.4), `ollama launch`
lists `claude` (Claude Code), `codex` (Codex) and `copilot` (Copilot CLI), among others.

**Harnesses don't serve models; they connect to a server that does.** Ollama (or Foundry Local, or
our `inference_server.py`) loads Qwen and answers requests. Each harness only needs that server to
speak the request format it was built for:

| Harness | Request format it sends | Served by Ollama? |
|---|---|---|
| Copilot | Through the model provider extension, which talks to Ollama | Yes |
| Claude Code | Anthropic's Messages API (`/v1/messages`) | Yes, since recent versions (answered here) |
| Codex | OpenAI's API | Yes |
| Our plugin | OpenAI's Chat Completions API (`/v1/chat/completions`) | Yes, and by `inference_server.py` |

So one Qwen running in Ollama can serve all of them; requests that arrive together may wait their
turn on a single GPU.

### CLI harness vs. extension harness

Each company offers its harness in two places: a **CLI** (command-line program) that runs in a
terminal, and a **VS Code extension** that shows it in a tab.

| | Command-line version (CLI) | VS Code version (extension) | Same harness? |
|---|---|---|---|
| **Claude Code** | `claude`, in any terminal | The **Claude Code** tab. It runs its own copy of the same program, `claude.exe`. | **Yes.** Same program, same settings (`~/.claude/`, `.claude/`), same session files. Only the window differs. |
| **Codex** | `codex`, in any terminal (installed with npm) | The **Codex** tab. It runs its own copy of the Codex CLI, `codex.exe`. | **Yes.** Same program, same settings (`~/.codex/config.toml`), same session files. |
| **GitHub Copilot** | **Copilot CLI**, `copilot` (installed with npm) | The **Chat** tab, with a **Local** session | **No.** The Chat tab's Local session is VS Code's own harness, with the model picker. Copilot CLI is GitHub's agent runtime (the Copilot SDK), which VS Code's **Copilot** session type also uses. |

**Where each harness's program is.** The harness is a program on your computer; the model is
separate (in Ollama, or in the vendor's cloud). `<version>` changes with each update, and
`<VS Code folder>` is the short code folder inside VS Code's install folder, e.g.
`%LOCALAPPDATA%\Programs\Microsoft VS Code\<VS Code folder>\`:

| Harness | Program used by the VS Code tab | Program used in a terminal |
|---|---|---|
| **Claude Code** | `%USERPROFILE%\.vscode\extensions\anthropic.claude-code-<version>-win32-x64\resources\native-binary\claude.exe` | `%USERPROFILE%\.local\bin\claude.exe` (the standalone install) |
| **Codex** | `%USERPROFILE%\.vscode\extensions\openai.chatgpt-<version>-win32-x64\bin\windows-x86_64\codex.exe` | `%APPDATA%\npm\codex` (from `npm install -g @openai/codex`) |
| **GitHub Copilot** | No `.exe`: VS Code's built-in Copilot extension, with the agent runtime as a Node.js package: `%LOCALAPPDATA%\Programs\Microsoft VS Code\<VS Code folder>\resources\app\extensions\copilot\node_modules\@github\copilot\` | `copilot` from `npm install -g @github/copilot`. VS Code's terminal also gets a `copilot` command, `%APPDATA%\Code\User\globalStorage\github.copilot-chat\copilotCli\copilot.bat`, which finds the real one or offers to install it. |

**The tab is a front end for the CLI.** For Claude Code and Codex, the extension mostly draws the
tab, starts the bundled program, and passes messages to it and back:
- The Claude Code tab starts `claude.exe` with `--input-format stream-json --output-format
  stream-json`: your messages go in, and replies, tool calls and approvals come out, as JSON lines.
- The Codex tab starts `codex.exe app-server` and exchanges JSON lines with it the same way.

The agent loop, tools, permissions, context handling and model connection are all in the program,
not in the extension. That's why the CLI and the tab behave the same, and why connecting a local
model works the same way for both. Copilot is built differently: the Chat tab's harness is part of
VS Code itself, and only the **Copilot** session type and the Copilot CLI use GitHub's separate
runtime.

**Why this matters for local models:** for Claude Code and Codex, connecting a local model is the
**same setting** for the CLI and the tab, because they're the same program. What differs is **how
far the setting reaches**: an environment variable in one terminal affects only that terminal,
while a settings file can affect every tab and terminal on the computer. For Copilot, the two are
different harnesses, connected in different ways.

### Connecting Qwen to each one

**First, for all of them:**
1. Ollama running, with the model pulled: `ollama pull qwen3.5:4b`.
2. A context window big enough for the harness's hidden prompt and tools (see "The hidden context"
   below). The harnesses send far more than our plugin does, so raise it, then quit and restart
   Ollama:
   ```powershell
   setx OLLAMA_CONTEXT_LENGTH 32768
   ```
   Check that `ollama ps` still shows `100% GPU`: a bigger window needs more GPU memory.
3. **Turn automatic approval off.** A small local model makes more mistakes, so let it ask before
   every tool call (rule 1 of [readme-safety.md](readme-safety.md)).

#### GitHub Copilot

| | How to connect Qwen | Tried here |
|---|---|---|
| **Extension** (Chat tab) | Install the **Ollama** extension, add `qwen3.5:4b` in the model picker (**Manage Models…**), and choose it in a **Local** session. Full steps: [05_readme-models.md](05_readme-models.md). Also set the utility model settings, so background jobs stay local ([readme-safety.md](readme-safety.md), section 2). | **Yes**, in Ask and Agent mode |
| **CLI** (Copilot CLI) | Install it (`npm install -g @github/copilot`), then run `ollama launch copilot`, which starts it already connected to Ollama. | No |

#### Claude Code

| | How to connect Qwen | Tried here |
|---|---|---|
| **CLI** | `ollama launch claude`, then choose `qwen3.5:4b`. It starts `claude --model qwen3.5:4b` with the connection set for that one process only, so other Claude Code windows are unaffected. | **Yes**: it started on Qwen |
| **Extension** (Claude Code tab) | Give the tab the same connection through environment variables. Put them in **a separate demo folder's** `.claude/settings.local.json`, so only sessions opened in that folder use Qwen (example below). | No |

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:11434",
    "ANTHROPIC_AUTH_TOKEN": "ollama",
    "ANTHROPIC_MODEL": "qwen3.5:4b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.5:4b"
  }
}
```
- `ANTHROPIC_BASE_URL` sends requests to Ollama instead of Anthropic. `ANTHROPIC_AUTH_TOKEN` can
  be any text; Ollama doesn't check it.
- `ANTHROPIC_DEFAULT_HAIKU_MODEL` points Claude Code's **background model** at Qwen too, so
  background jobs don't fail or go to the cloud.
- The extension also has a `claudeCode.environmentVariables` setting, but it's a **machine-wide**
  setting: it would switch **every** Claude Code tab on the computer to Qwen. Prefer the folder
  file.
- These are the usual variable names, but they're not confirmed against what `ollama launch`
  sets. Try them in a separate folder first.

**What we saw** when starting the CLI on Qwen:
- *"Model "qwen3.5:4b" isn't recognized · context assumed to be 200k tokens"*. Claude Code doesn't
  know Qwen, so it assumes a 200,000-token window. Ollama gives Qwen only what
  `OLLAMA_CONTEXT_LENGTH` allows, and silently cuts anything longer, so raise it (step 2 above).
- *"auto mode on"*: Claude Code approves tool calls by itself when it judges them low risk, with a
  4B model doing the judging. Press **Shift+Tab** to switch to the default mode, which asks
  first, or to **plan** mode, which only plans.

#### Codex

| | How to connect Qwen | Tried here |
|---|---|---|
| **CLI** | `codex --oss -m qwen3.5:4b` (`--oss` means a local provider; add `--local-provider ollama` or `lmstudio` to choose one), or `ollama launch codex`. Only that run uses Qwen. | No |
| **Extension** (Codex tab) | The tab reads `config.toml`. Setting the model there switches the tab, and every `codex` run on the computer (example below). | No |

```toml
# ~/.codex/config.toml: used by the Codex tab and every codex command
model_provider = "ollama"
model = "qwen3.5:4b"
```
- To keep your normal setup for everyday use, put these lines in a **profile** instead, e.g. under
  `[profiles.local]`, and run `codex --profile local` in a terminal. The tab uses the top-level
  values, though, so a profile only helps the CLI.
- Codex can also read a project's own `.codex/config.toml` in a folder you've marked as trusted.
  That would be the way to limit the tab to one demo folder; not tried here.

#### Our plugin

`tinyllama.serverUrl` and `tinyllama.model` ([05_readme-models.md](05_readme-models.md)). It's
always local, and has no tools or background jobs.

### What to expect

| Harness | CLI on Qwen | Extension on Qwen |
|---|---|---|
| GitHub Copilot | Possible (`ollama launch copilot`); not tried | **Tried: works** (Local session) |
| Claude Code | **Tried: starts** (`ollama launch claude`) | Possible (folder settings); not tried |
| Codex | Possible (`codex --oss`); not tried | Possible (`config.toml`); not tried |

**A harness written for big cloud models doesn't automatically work well with a small one.** Its
hidden instructions and tool list must fit the local model's context window, and the model must be
good at calling tools. Expect a 4B model to be slow, and to struggle with long agent tasks in any
of them.

Some harnesses also use a **second, smaller model** for background jobs (section 2 of
[readme-sessions.md](readme-sessions.md)). Point that one at the local server too, or those jobs
fail or go to the cloud.

### The hidden context: what each harness sends with every message

Your message is only a small part of each request. Every harness wraps it in its own hidden
context, and **all of it goes to wherever the model runs**: your GPU for a local model, the
vendor's servers for a cloud model.

| Part | Copilot | Claude Code | Codex | Our plugin |
|---|---|---|---|---|
| **System prompt** (who to be, how to behave) | Copilot's own, several thousand tokens | Claude Code's own | Codex's own | One line: *"You are a helpful coding assistant…"* |
| **Tool definitions** | In Agent mode, every tool turned on in **Configure Tools**, incl. MCP tools | Its built-in tools (read, edit, shell, web, …) and MCP tools | Its tools (shell, apply patch, …) and MCP tools | None |
| **Instruction files** | `.github/copilot-instructions.md`, matching `.instructions.md` files, `AGENTS.md`, … | `CLAUDE.md` files, `.claude/rules/` | `AGENTS.md` files | None |
| **Skills list** | Name and description of each skill and prompt file (the customizations index) | Name and description of each skill | Name and description of each skill | None |
| **Environment** | Operating system, workspace folder | Working folder, operating system, git status, date | Working folder, sandbox and approval settings | None |
| **History and attachments** | This session's turns, attached files | This session's turns, `@` files | This session's turns, `@` files | Earlier turns (plugin) or this participant's turns (`@tinyllama`) |
| **Size of "Hello?"** | About **9,000 tokens** in Agent mode (measured with Ollama's log) | Not measured here | Not measured here | A few hundred tokens |

**What this means:**
- **With a cloud model,** all of it leaves your computer: your instruction files, the names of your
  skills, your folder layout and git status, and every attached or `@` file.
- **With a local model,** it stays on your computer, apart from any background jobs that go to the
  cloud (section 2 of [readme-safety.md](readme-safety.md)).
- **Instruction files and skills are obeyed like your own words.** Whatever is in them is in every
  request, including anything injected into a file from a public collection
  ([06_readme-mcp-agents.md](06_readme-mcp-agents.md)).
- **To see it:** for Copilot, use **Chat Debug View**. For any harness pointed at Ollama, Ollama's
  debug log shows the full request as the model received it. Both are in
  [readme-context.md](readme-context.md). Pointing a harness at a local model is the easiest way to
  see what it really sends.

> Claude Code's and Codex's hidden context is described from what each program contains, not
> measured here. Check it with Ollama's log before quoting sizes.

### ⚠ Secrets: `.gitignore` doesn't keep them out of the context (safety rule 3)

A `.env` file isn't part of the hidden context: no harness reads it on its own. But anything an
agent **reads** joins the conversation, and in Agent mode the model decides what to read.
`.gitignore` only keeps a file out of Git: read tools, shell commands (`cat .env`) and attachments
can still reach it, and reading a project file usually happens **without asking** in all three
harnesses. Once read, it goes with every following request, to the vendor if the model is in the
cloud, and into the session files. A cloud agent (section 7, "Background tasks") also takes
everything that was ever committed.

How each harness can block it:

| Harness | Block reading `.env` | Notes |
|---|---|---|
| **Claude Code** | `.claude/settings.json`: `"permissions": { "deny": ["Read(./.env)", "Read(./.env.*)"] }` | Blocks its read tool; keep approvals on for shell commands |
| **Codex** | `deny_read` paths in a `config.toml` permission profile | Applies to its sandbox, including commands |
| **Copilot** | No per-file read block for its read tool; `denyRead` in the terminal sandbox settings (preview) stops commands; content exclusion on Business/Enterprise | Edits to `.env` already always need approval |

**The fix that works everywhere:** keep secrets **outside the project folder**, e.g. as Windows
environment variables or in a secrets manager. Also don't attach `.env` or leave it open in the
editor while chatting. The full list, with what each layer covers:
[readme-safety.md](readme-safety.md), section 3.

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

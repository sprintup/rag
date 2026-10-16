# Building Your Own Harness

Copilot, Claude Code and Codex are three ready-made **harnesses**: programs that wrap a model with
a system prompt, context, tools, sessions and permissions
([04_readme-harness.md](04_readme-harness.md), section 7). This guide explains what it takes to build
your own with some of the same features, in VS Code, using this project's TinyLlama plugin as the
starting point.

> Like the rest of this project, this is for learning. A harness that runs tools acts on your
> computer and accounts, so read [readme-safety.md](readme-safety.md) before giving one real tools.

---

## 1. Two ways to build one in VS Code

| | **A. Inside the Chat view** (a chat participant) | **B. Your own tab** (a webview, like Claude Code and Codex) |
|---|---|---|
| **In this project** | `@tinyllama` | The **TinyLlama** tab and the **TinyLlama Chat** panel |
| **The window, input box, history display** | VS Code's | You build them (HTML, CSS, JavaScript) |
| **`@`, `/`, `#`, drag and drop** | VS Code's, ready to use | You build them |
| **Skills and prompt files** | VS Code finds them and attaches the one you choose | You find, read and send them |
| **Sessions saved to disk, titles** | VS Code's | You build them |
| **Model** | The one picked in the model picker, or your own server | Your own server, or the one in the model picker (through `vscode.lm`) |
| **Tools and the agent loop** | You write the loop; VS Code supplies the tools and approval prompts | You write the loop, the tools and the approval prompts |
| **Needs GitHub Copilot** | Yes: the Chat view comes with it | No |
| **Freedom** | Limited to what the Chat view offers | Anything a web page can do |

**Route A** is far less work: VS Code's harness does the user interface and context, and your
code decides what to send to which model. **Route B** is what Anthropic and OpenAI did: full
control, but you build everything Copilot gives you for free. Both can live in one extension, as
they do in this project.

A third route is to start from a **harness SDK** (GitHub Copilot SDK, Claude Agent SDK, Codex SDK)
that already has the agent loop, tools and sessions, and put it behind route A or B. That's
roughly how the Claude Code and Codex tabs work: a VS Code tab in front of the vendor's own
program.

---

## 2. The files you need

**Every VS Code extension:**

| File | What it does | In this project |
|---|---|---|
| `package.json` | The extension's **manifest**: its name, the VS Code version it needs, its entry file, and everything it adds to VS Code (`contributes`) | [src/vscode-extension/package.json](src/vscode-extension/package.json) |
| `extension.js` | The code. VS Code calls its `activate()` when the extension is first needed. | [src/vscode-extension/extension.js](src/vscode-extension/extension.js) |

**What `package.json` declares (`contributes`) for each feature:**

| Feature | Entry in `package.json` | Code in `extension.js` |
|---|---|---|
| An `@name` in the Chat view (route A) | `chatParticipants` | `vscode.chat.createChatParticipant()` |
| Your own `/commands` for that participant | `commands` inside the `chatParticipants` entry | Read `request.command` |
| A tab in the side bar (route B) | `viewsContainers` → `secondarySidebar`, and `views` with `"type": "webview"` | `vscode.window.registerWebviewViewProvider()` |
| A panel beside the editor (route B) | Nothing; a command to open it | `vscode.window.createWebviewPanel()` |
| Commands for `Ctrl+Shift+P` and menus | `commands`, `menus` | `vscode.commands.registerCommand()` |
| Settings (server address, model, limits) | `configuration` | `vscode.workspace.getConfiguration()` |
| Tools that Copilot's harness, or yours, can call | `languageModelTools` | `vscode.lm.registerTool()` |
| A model in the model picker | `languageModelChatProviders` | `vscode.lm.registerLanguageModelChatProvider()` |

**Files your harness reads** (optional, and your choice where):

| File | Purpose | Copilot's equivalent |
|---|---|---|
| A system prompt | Who the assistant is, how to answer | Built in |
| Instruction files | Project rules added to every request | `.github/copilot-instructions.md`, `AGENTS.md` |
| Skills (`SKILL.md`) | Packaged instructions, used on request | `.github/skills/` |
| An MCP configuration | Which MCP servers to start | `.vscode/mcp.json` |
| Session files | Saved conversations | VS Code's workspace storage ([readme-sessions.md](readme-sessions.md)) |

Reusing existing formats (`SKILL.md`, `AGENTS.md`, MCP) means skills and settings written for other
harnesses work in yours too.

---

## 3. How the Chat view's features reach your code (route A)

When you use a chat participant, **VS Code is the harness for everything except the model call**.
It hands your code one object per message, and you decide what to do with it:

```js
vscode.chat.createChatParticipant("tinyllama.chat", async (request, chatContext, stream, token) => {
  request.prompt          // what you typed, without the @name
  request.command         // a /command declared by this participant, if any
  request.references      // attachments: dropped files, #file, #selection, the paperclip
  request.toolReferences  // tools you named with #, e.g. #fetch
  request.model           // the model chosen in the model picker
  chatContext.history     // this participant's earlier turns in this session
  stream.markdown("…")    // write the reply, piece by piece
  token                   // cancelled when you press Stop
});
```

This project's handler is `handleChatRequest()` in
[extension.js](src/vscode-extension/extension.js#L241).

### The operators

| Operator | Who handles it | What your code sees |
|---|---|---|
| **`@name`** | **VS Code.** It routes the message to the participant with that name, from your `chatParticipants` entry. | Your handler is called; `request.prompt` no longer contains `@name`. |
| **`/command`** (yours) | **VS Code** lists the commands you declared in `package.json`. | `request.command`, e.g. `"explain"`. You decide what it does. |
| **`/skill`** (e.g. `/pirate`) | **VS Code** finds `.github/skills/*/SKILL.md` and adds the chosen skill to the request. | The skill arrives with the message; you pass it to the model. This plugin has **no skill code** ([04_readme-harness.md](04_readme-harness.md), section 5). |
| **`#file`, `#selection`** | **VS Code** resolves them. | `request.references`, as file locations (`vscode.Uri`) or ranges (`vscode.Location`). **You read the files** and decide how much to send. |
| **`#tool`** (e.g. `#fetch`) | **VS Code** resolves the tool name. | `request.toolReferences`. Your loop must actually call the tool. |

### Drag and drop: built into VS Code

**Yes, adding context by dragging is a VS Code feature.** Dropping a file from the Explorer (or
from Windows) onto the Chat view's input box, using the paperclip, or typing `#file` all add an
**attachment** to the message. Your participant receives every one of them the same way, in
`request.references`.

What VS Code does **not** do is decide what to send: that's your harness's job. This plugin's
`readAttachments()` ([extension.js:286](src/vscode-extension/extension.js#L286)):
1. reads each attached file or selection;
2. skips folders and images, and text VS Code attaches by itself (such as its customizations
   index), which would fill TinyLlama's small context;
3. cuts the total to 3,000 characters, and tells you what it cut;
4. puts the files above your question.

Copilot's own harness does the same job with far larger limits. That's one of the main decisions
in any harness: **what goes into the context, and what's left out.**

---

## 4. What you build yourself in your own tab (route B)

In a webview (the TinyLlama tab or panel), VS Code gives you an empty web page and a message
channel to `extension.js` (`postMessage`). Everything else is yours:

| Feature | How to build it | In this project |
|---|---|---|
| **Input box, Send, Stop** | HTML and a little JavaScript in the page | Yes: `getHtml()` |
| **Streaming replies** | `extension.js` reads the model's stream and `postMessage`s each piece | Yes |
| **History** | An array in `extension.js`, resent with each request | Yes, in memory ([readme-sessions.md](readme-sessions.md), section 3) |
| **`@file` mentions** | Watch the input for `@`, list files with `vscode.workspace.findFiles()`, read the chosen one | No |
| **`/commands` and skills** | Watch for `/` at the start; list `SKILL.md` files from your skill folders; add the chosen one's text to the request | No |
| **Drag and drop** | Handle the page's `drop` event and read the file locations from the drag data, then send them to `extension.js` to read. VS Code generally asks you to **hold Shift** while dropping into a web page; check this in your version. | No |
| **The current file or selection** | `vscode.window.activeTextEditor` in `extension.js` | Partly: **Ask About Selection** |
| **Saved sessions** | Write each conversation to a file in `context.storageUri` (per workspace) or `context.globalStorageUri` | No |
| **Approval prompts** | Show a button in the page, or `vscode.window.showWarningMessage(…, "Allow")` | No |

The Claude Code and Codex tabs are route B: each built its own input box, `@` file search, `/`
commands and approvals, which is why they behave differently from the Chat tab.

---

## 5. The agent loop: tools

Without tools, a harness is a chat client: one request, one reply. With tools, it becomes an
**agent**: the model can ask for a tool, the harness runs it and sends back the result, and this
repeats until the model answers. This plugin stops at the chat client. Adding the loop in route A
looks roughly like this, using the model from the model picker and the tools registered in VS Code:

```js
// Inside the participant's handler
const messages = [vscode.LanguageModelChatMessage.User(SYSTEM_PROMPT + "\n\n" + request.prompt)];
const tools = vscode.lm.tools.filter((tool) => ALLOWED_TOOLS.includes(tool.name)); // as few as possible

while (true) {
  const response = await request.model.sendRequest(messages, { tools }, token);
  const calls = [];
  let text = "";
  for await (const part of response.stream) {
    if (part instanceof vscode.LanguageModelTextPart) {
      stream.markdown(part.value);
      text += part.value;
    } else if (part instanceof vscode.LanguageModelToolCallPart) {
      calls.push(part); // the model wants a tool
    }
  }
  if (!calls.length) break; // no tool requested: the answer is finished

  // Record the model's request, run each tool (VS Code asks you to approve it), and send the results back
  messages.push(vscode.LanguageModelChatMessage.Assistant([new vscode.LanguageModelTextPart(text), ...calls]));
  for (const call of calls) {
    const result = await vscode.lm.invokeTool(call.name, {
      input: call.input,
      toolInvocationToken: request.toolInvocationToken, // shows the approval in the chat
    }, token);
    messages.push(vscode.LanguageModelChatMessage.User([
      new vscode.LanguageModelToolResultPart(call.callId, result.content),
    ]));
  }
}
```

The [Language Model API](https://code.visualstudio.com/api/extension-guides/ai/language-model) has
no "system" message, so the system prompt goes in the first user message. `ALLOWED_TOOLS` is your
list: **start with none** (rule 1 of [readme-safety.md](readme-safety.md)).

This only works with a model that supports **tool calling**: Qwen 3.5 does, TinyLlama doesn't
([05_readme-models.md](05_readme-models.md)). In route B you'd write the same loop against your
own server, and also write the tools and the approval prompts.

**Tools from MCP servers:** in route A, MCP servers configured in VS Code (`.vscode/mcp.json`) show
up in `vscode.lm.tools` like any other tool. In route B, your extension would start the servers
itself with an MCP client library, such as the official TypeScript SDK
(`@modelcontextprotocol/sdk`).

---

## 6. Everything a full harness has, and where this project stands

| Feature | What it means | VS Code gives it to a participant (route A) | This plugin |
|---|---|---|---|
| **System prompt** | Who to be, how to answer | No | One line |
| **Model connection** | Sending requests, streaming replies | Yes (`request.model`), or your own server | Own server (`streamChat()`) |
| **History** | Resending earlier turns | Yes (`chatContext.history`) | Yes |
| **Sessions on disk, titles** | Going back to a conversation | Yes | Chat view only |
| **Attachments** | Files, selections, drag and drop | Yes (`request.references`) | Reads them, capped at 3,000 characters |
| **Fitting the context window** | Cutting, dropping or summarizing | No | Caps attachments; the server drops old turns |
| **Skills** | `/name` instructions | Yes, chosen with `/` | Chat view only |
| **Instruction files** | Rules for every request | Only an index: VS Code attaches `vscode.customizations.index`, listing instructions and skills by name and path. Reading them is up to you. | Skips the index, to save TinyLlama's context |
| **Tools and the agent loop** | Model asks, harness runs, repeat | Tools and approvals yes; the loop is yours | No |
| **MCP servers** | Tools from other programs | Yes, as tools | No |
| **Permissions** | Which tools run without asking | Approval prompts yes; your allowed list is yours | Nothing to permit |
| **Utility model** | Titles, summaries in the background | Titles yes | No |
| **Your own tab** | Full control of the interface | No: route B | Yes: TinyLlama tab and panel |

**A sensible order to build in:** system prompt and history (done) → attachments (done) →
instruction files → skills in your own tab → saved sessions → **one** read-only tool with an
approval prompt → the agent loop → more tools, each one deliberately.

---

## 7. Security: what changes when you write the harness

When you build the harness, the protections other harnesses have are **your** job:
- **Tools act as you.** Start with none, add read-only ones first, and ask before anything that
  edits files, runs commands or uses the network.
- **Everything you put in the context is obeyed.** Instruction files, skills, web pages and tool
  results can contain injected instructions ([06_readme-mcp-agents.md](06_readme-mcp-agents.md)).
  Never let tool output turn into actions without an approval.
- **Render replies as plain text** in a webview, or sanitize them. This plugin's page uses
  `textContent`, never `innerHTML`, for model output, and a Content Security Policy that only runs
  its own script.
- **Session files and logs keep prompts.** If you save sessions, store them in the extension's
  storage folder, and give users a way to delete them.
- **Keep servers on `127.0.0.1`** ([readme-safety.md](readme-safety.md), section 1).

---

## Further reading

| Topic | Link |
|---|---|
| Chat participants (route A) | [Chat Participant API](https://code.visualstudio.com/api/extension-guides/ai/chat) |
| Calling the picked model, tool calling | [Language Model API](https://code.visualstudio.com/api/extension-guides/ai/language-model) |
| Registering tools | [Language Model Tools API](https://code.visualstudio.com/api/extension-guides/ai/tools) |
| Adding a model to the picker | [Language Model Chat Provider API](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider) |
| Webviews and side bar tabs (route B) | [Webview API](https://code.visualstudio.com/api/extension-guides/webview), [Views](https://code.visualstudio.com/api/ux-guidelines/views) |
| Skill and MCP formats | [Agent Skills specification](https://agentskills.io/specification), [MCP specification](https://modelcontextprotocol.io/specification/latest) |
| What a harness sends | [readme-context.md](readme-context.md), and section 8 of [04_readme-harness.md](04_readme-harness.md) |

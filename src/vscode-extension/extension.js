// TinyLlama Chat - a minimal VS Code extension that chats with the local TinyLlama model.
//
// A plain chat client (no tools, no agent loop): it builds the prompt (system prompt + chat
// history + your question, and optionally the code you selected), sends it to the local
// inference server (src/inference_server.py), and streams the reply into the chat.
//
// The chat appears in two places, showing the same conversation:
//   - a "TinyLlama" tab in the secondary side bar, next to Chat, Claude Code and Codex. Those
//     tabs work the same way: a view container in package.json holding a webview.
//   - a "TinyLlama Chat" panel that opens beside the editor, on the right.
//
// Commands (Ctrl+Shift+P):
//   TinyLlama: Open Chat            - opens the chat panel on the right
//   TinyLlama: Ask About Selection  - asks a question about the code selected in the editor
//                                     (also on the editor's right-click menu)
// Chat participant:
//   @tinyllama                      - ask TinyLlama from VS Code's Chat view

const vscode = require("vscode");

// Instructions sent at the start of every conversation
const SYSTEM_PROMPT = "You are a helpful coding assistant running locally. Answer briefly and clearly.";

let panel;              // the chat panel on the right, if open
const chats = new Set(); // the webviews showing the chat (the tab and the panel) that are ready
let pending;            // a question waiting for a chat to finish loading
let history = [];       // the conversation so far: [{ role, content }, ...]
let controller;         // lets the Stop button cancel the request in progress

function activate(context) {
  context.subscriptions.push(
    // The TinyLlama tab. VS Code calls resolveWebviewView the first time the tab is shown.
    vscode.window.registerWebviewViewProvider("tinyllama.chatView", {
      resolveWebviewView: (view) => setUpChat(view.webview, view.onDidDispose),
    }, {
      webviewOptions: { retainContextWhenHidden: true }, // keep the conversation when the tab is hidden
    }),
    vscode.commands.registerCommand("tinyllama.openChat", () => showPanel()),
    vscode.commands.registerCommand("tinyllama.askAboutSelection", askAboutSelection),
    // Changing a tinyllama.* setting starts a new conversation, so the header shows the new model
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("tinyllama")) {
        if (controller) controller.abort();
        history = [];
        for (const webview of [...chats]) {
          chats.delete(webview);
          webview.html = getHtml(serverUrl(), vscode.workspace.getConfiguration("tinyllama").get("model"));
        }
      }
    })
  );

  // @tinyllama in VS Code's Chat view. Only available when the Chat view exists (it comes
  // with GitHub Copilot Chat), so skip it quietly otherwise.
  if (vscode.chat && vscode.chat.createChatParticipant) {
    const participant = vscode.chat.createChatParticipant("tinyllama.chat", handleChatRequest);
    participant.iconPath = new vscode.ThemeIcon("hubot");
    context.subscriptions.push(participant);
  }
}

function deactivate() {
  if (controller) controller.abort();
}

// ------------------------------------------------------------------------------------------
// Commands
// ------------------------------------------------------------------------------------------

async function askAboutSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showInformationMessage("Select some code first, then run TinyLlama: Ask About Selection.");
    return;
  }
  const question = await vscode.window.showInputBox({
    prompt: "Ask TinyLlama about the selected code",
    value: "Explain what this code does.",
  });
  if (!question) return;

  // Put the selected code in a Markdown code block after the question
  const code = editor.document.getText(editor.selection);
  const language = editor.document.languageId;
  const message = `${question}\n\n\`\`\`${language}\n${code}\n\`\`\``;

  showPanel();
  if (chats.has(panel.webview)) {
    send(message);
  } else {
    pending = message; // sent as soon as the panel says it's ready
  }
}

// ------------------------------------------------------------------------------------------
// The chat (the TinyLlama tab and the panel on the right)
// ------------------------------------------------------------------------------------------

function showPanel() {
  if (panel) {
    panel.reveal(vscode.ViewColumn.Beside);
    return;
  }
  panel = vscode.window.createWebviewPanel("tinyllamaChat", "TinyLlama Chat", vscode.ViewColumn.Beside, {
    retainContextWhenHidden: true, // keep the conversation when the tab is hidden
  });
  setUpChat(panel.webview, panel.onDidDispose);
  panel.onDidDispose(() => (panel = undefined));
}

// Loads the chat page into a webview (the tab's or the panel's) and handles its buttons
function setUpChat(webview, onDidDispose) {
  webview.options = { enableScripts: true };
  webview.html = getHtml(serverUrl(), vscode.workspace.getConfiguration("tinyllama").get("model"));

  webview.onDidReceiveMessage((message) => {
    if (message.type === "ready") {
      // Show the conversation so far, so the tab and the panel match
      for (const turn of history) webview.postMessage({ type: turn.role, text: turn.content });
      chats.add(webview);
      if (pending) {
        send(pending);
        pending = undefined;
      }
    } else if (message.type === "send") {
      send(message.text);
    } else if (message.type === "stop") {
      if (controller) controller.abort();
    } else if (message.type === "clear") {
      history = [];
      post({ type: "clear" });
    }
  });

  onDidDispose(() => chats.delete(webview));
}

// Sends a message to every open chat
function post(message) {
  for (const webview of chats) webview.postMessage(message);
}

// ------------------------------------------------------------------------------------------
// Talking to the local server
// ------------------------------------------------------------------------------------------

function serverUrl() {
  return vscode.workspace.getConfiguration("tinyllama").get("serverUrl").replace(/\/$/, "");
}

async function send(text) {
  if (controller) {
    post({ type: "error", text: "TinyLlama is still answering. Wait, or press Stop." });
    return;
  }
  history.push({ role: "user", content: text });
  post({ type: "user", text });
  post({ type: "start" });

  controller = new AbortController();
  let reply = "";

  try {
    const messages = [{ role: "system", content: SYSTEM_PROMPT }, ...history];
    await streamChat(messages, controller.signal, (delta) => {
      reply += delta;
      post({ type: "delta", text: delta });
    });
    history.push({ role: "assistant", content: reply });
    if (!reply) {
      post({ type: "info", text: "The model sent no answer text. A model that thinks first, such as Qwen, " +
        "may have used all of tinyllama.maxTokens on thinking: turn tinyllama.thinking off, or try a larger value." });
    }
  } catch (err) {
    if (err.name === "AbortError") {
      history.push({ role: "assistant", content: reply });
      post({ type: "info", text: "Stopped." });
    } else {
      history.pop(); // forget the question that failed
      post({ type: "error", text: serverErrorMessage(err) });
    }
  } finally {
    controller = undefined;
    post({ type: "done" });
  }
}

// Sends messages to the local server and calls onText with each piece of the reply as it
// arrives. Used by both the chat view and the @tinyllama chat participant.
async function streamChat(messages, signal, onText) {
  const maxTokens = vscode.workspace.getConfiguration("tinyllama").get("maxTokens");

  // The same OpenAI-style request any AI tool would send; "stream: true" asks for the
  // reply piece by piece as server-sent events ("data: {...}" lines)
  const response = await fetch(`${serverUrl()}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: vscode.workspace.getConfiguration("tinyllama").get("model"),
      messages,
      max_tokens: maxTokens,
      stream: true,
      // Asks models like Qwen to answer straight away. Their thinking counts toward max_tokens,
      // and can use it all up before any answer. Servers that don't know this field ignore it.
      ...(vscode.workspace.getConfiguration("tinyllama").get("thinking") ? {} : { reasoning_effort: "none" }),
    }),
    signal,
  });
  if (!response.ok) throw new Error(`the server returned HTTP ${response.status}`);

  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    // Each event ends with a blank line
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const event = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of event.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") continue;
        const delta = JSON.parse(data).choices?.[0]?.delta?.content;
        if (delta) onText(delta);
      }
    }
  }
}

function serverErrorMessage(err) {
  return `Couldn't reach the TinyLlama server at ${serverUrl()} (${err.message}). ` +
    "Is `python src/inference_server.py` running?";
}

// ------------------------------------------------------------------------------------------
// The @tinyllama chat participant (in VS Code's Chat view)
// ------------------------------------------------------------------------------------------

// Called each time you send a message starting with @tinyllama in the Chat view
async function handleChatRequest(request, chatContext, stream, token) {
  // Rebuild the conversation from this participant's earlier turns in the Chat view
  const messages = [{ role: "system", content: SYSTEM_PROMPT }];
  for (const turn of chatContext.history) {
    if (turn instanceof vscode.ChatRequestTurn) {
      messages.push({ role: "user", content: turn.prompt });
    } else if (turn instanceof vscode.ChatResponseTurn) {
      const text = turn.response
        .map((part) => (part instanceof vscode.ChatResponseMarkdownPart ? part.value.value : ""))
        .join("")
        .replace(/^\*[^\n]*\*\n*/gm, "") // drop the plugin's own *italic notes*, so TinyLlama doesn't copy them
        .trim();
      if (text) messages.push({ role: "assistant", content: text });
    }
  }

  // Add any files or selections attached in the Chat view (paperclip, #file, #selection)
  const { parts, notes } = await readAttachments(request.references);
  for (const ref of request.references) {
    if (ref.value instanceof vscode.Uri || ref.value instanceof vscode.Location) stream.reference(ref.value);
  }
  if (notes.length) stream.markdown(`*${notes.join(" ")}*\n\n`);

  const prompt = parts.length
    ? `Use these attached files to answer.\n\n${parts.join("\n\n")}\n\nQuestion: ${request.prompt}`
    : request.prompt;
  messages.push({ role: "user", content: prompt });

  // The Chat view's stop button cancels the request
  const abort = new AbortController();
  token.onCancellationRequested(() => abort.abort());

  try {
    await streamChat(messages, abort.signal, (text) => stream.markdown(text));
  } catch (err) {
    if (err.name !== "AbortError") stream.markdown(serverErrorMessage(err));
  }
}

// TinyLlama reads at most ~2,048 tokens (prompt + reply), so attachments are capped at about
// 3,000 characters in total (roughly 750-1,000 tokens), leaving room for the question and reply.
const MAX_ATTACHMENT_CHARS = 3000;

// Reads the text of each attachment. Returns the text blocks to add to the prompt, plus notes
// about anything that was cut short or left out.
async function readAttachments(references) {
  const parts = [];
  const notes = [];
  let remaining = MAX_ATTACHMENT_CHARS;
  for (const ref of references) {
    let name;
    let text;
    try {
      if (ref.value instanceof vscode.Uri) {
        name = vscode.workspace.asRelativePath(ref.value);
        if ((await vscode.workspace.fs.stat(ref.value)).type & vscode.FileType.Directory) {
          notes.push(`${name} is a folder; attach individual files instead.`);
          continue;
        }
        // A whole file
        text = new TextDecoder().decode(await vscode.workspace.fs.readFile(ref.value));
      } else if (ref.value instanceof vscode.Location) {
        // Part of a file, e.g. #selection
        const document = await vscode.workspace.openTextDocument(ref.value.uri);
        const range = ref.value.range;
        name = `${vscode.workspace.asRelativePath(ref.value.uri)} (lines ${range.start.line + 1}-${range.end.line + 1})`;
        text = document.getText(range);
      } else {
        // Skip everything else: images (TinyLlama can't read them) and text that VS Code
        // attaches by itself, like its "vscode.customizations.index", which would use up
        // TinyLlama's small context
        continue;
      }
    } catch (err) {
      notes.push(`Couldn't read ${ref.id}.`);
      continue;
    }
    if (remaining <= 0) {
      notes.push(`${name} was left out to fit TinyLlama's 2,048-token limit.`);
      continue;
    }
    if (text.length > remaining) {
      notes.push(`Only the first ${remaining.toLocaleString()} of ${text.length.toLocaleString()} characters of ${name} were sent, to fit TinyLlama's 2,048-token limit.`);
      text = text.slice(0, remaining);
    }
    remaining -= text.length;
    parts.push(`File: ${name}\n\`\`\`\n${text}\n\`\`\``);
  }
  return { parts, notes };
}

// ------------------------------------------------------------------------------------------
// The view's HTML, CSS and script
// ------------------------------------------------------------------------------------------

function getHtml(url, model) {
  const nonce = Math.random().toString(36).slice(2) + Date.now().toString(36);
  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body { display: flex; flex-direction: column; height: 100vh; margin: 0; padding: 0 12px;
         font-family: var(--vscode-font-family); color: var(--vscode-foreground);
         background: var(--vscode-sideBar-background); box-sizing: border-box; }
  header { padding: 10px 0; border-bottom: 1px solid var(--vscode-panel-border);
           color: var(--vscode-descriptionForeground); font-size: 12px; }
  #log { flex: 1; overflow-y: auto; padding: 8px 0; }
  .msg { white-space: pre-wrap; padding: 8px 10px; margin: 6px 0; border-radius: 6px;
         font-family: var(--vscode-editor-font-family); font-size: var(--vscode-editor-font-size); }
  .user { background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border, transparent); }
  .assistant { border-left: 3px solid var(--vscode-textLink-foreground); }
  .assistant.pending:empty::after { content: "Thinking…"; color: var(--vscode-descriptionForeground); }
  .error { color: var(--vscode-errorForeground); }
  .info { color: var(--vscode-descriptionForeground); font-style: italic; }
  form { display: flex; gap: 6px; padding: 10px 0; border-top: 1px solid var(--vscode-panel-border); }
  textarea { flex: 1; resize: vertical; min-height: 40px; padding: 6px; font-family: inherit;
             color: var(--vscode-input-foreground); background: var(--vscode-input-background);
             border: 1px solid var(--vscode-input-border, transparent); }
  button { padding: 4px 12px; color: var(--vscode-button-foreground); background: var(--vscode-button-background);
           border: none; cursor: pointer; }
  button.secondary { color: var(--vscode-button-secondaryForeground); background: var(--vscode-button-secondaryBackground); }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
  <header>Local model: ${model} · Server: ${url}</header>
  <div id="log"></div>
  <form id="form">
    <textarea id="input" placeholder="Ask TinyLlama… (Enter to send, Shift+Enter for a new line)"></textarea>
    <button id="send" type="submit">Send</button>
    <button id="stop" type="button" class="secondary" disabled>Stop</button>
    <button id="clear" type="button" class="secondary">Clear</button>
  </form>
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  const log = document.getElementById("log");
  const input = document.getElementById("input");
  const sendButton = document.getElementById("send");
  const stopButton = document.getElementById("stop");
  let current; // the reply currently being written

  function add(kind, text) {
    const div = document.createElement("div");
    div.className = "msg " + kind;
    div.textContent = text; // plain text only, so replies can't inject HTML
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  function setBusy(busy) {
    sendButton.disabled = busy;
    stopButton.disabled = !busy;
  }

  document.getElementById("form").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    vscode.postMessage({ type: "send", text });
    input.value = "";
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      document.getElementById("form").requestSubmit();
    }
  });
  stopButton.addEventListener("click", () => vscode.postMessage({ type: "stop" }));
  document.getElementById("clear").addEventListener("click", () => vscode.postMessage({ type: "clear" }));

  // Messages from the extension
  window.addEventListener("message", (event) => {
    const message = event.data;
    if (message.type === "user") add("user", message.text);
    else if (message.type === "assistant") add("assistant", message.text);
    else if (message.type === "clear") log.innerHTML = "";
    else if (message.type === "start") { current = add("assistant pending", ""); setBusy(true); }
    else if (message.type === "delta" && current) { current.textContent += message.text; log.scrollTop = log.scrollHeight; }
    else if (message.type === "error") add("error", message.text);
    else if (message.type === "info") add("info", message.text);
    else if (message.type === "done") {
      if (current) {
        current.classList.remove("pending");
        if (!current.textContent) current.remove();
      }
      current = undefined;
      setBusy(false);
    }
  });

  vscode.postMessage({ type: "ready" });
</script>
</body>
</html>`;
}

module.exports = { activate, deactivate };

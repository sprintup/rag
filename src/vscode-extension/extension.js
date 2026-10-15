// TinyLlama Chat - a minimal VS Code extension that chats with the local TinyLlama model.
//
// A plain chat client (no tools, no agent loop): it builds the prompt (system prompt + chat
// history + your question, and optionally the code you selected), sends it to the local
// inference server (src/inference_server.py), and streams the reply into a chat panel.
//
// Commands (Ctrl+Shift+P):
//   TinyLlama: Open Chat            - opens the chat panel
//   TinyLlama: Ask About Selection  - asks a question about the code selected in the editor
//                                     (also on the editor's right-click menu)

const vscode = require("vscode");

// Instructions sent at the start of every conversation
const SYSTEM_PROMPT = "You are a helpful coding assistant running locally. Answer briefly and clearly.";

let panel;              // the chat panel, if open
let panelReady = false; // true once the panel's script has loaded and can receive messages
let pending;            // a question waiting for the panel to finish loading
let history = [];       // the conversation so far: [{ role, content }, ...]
let controller;         // lets the Stop button cancel the request in progress

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("tinyllama.openChat", () => showPanel()),
    vscode.commands.registerCommand("tinyllama.askAboutSelection", askAboutSelection)
  );
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
  if (panelReady) {
    send(message);
  } else {
    pending = message; // sent as soon as the panel says it's ready
  }
}

// ------------------------------------------------------------------------------------------
// The chat panel
// ------------------------------------------------------------------------------------------

function showPanel() {
  if (panel) {
    panel.reveal(vscode.ViewColumn.Beside);
    return;
  }
  panel = vscode.window.createWebviewPanel("tinyllamaChat", "TinyLlama Chat", vscode.ViewColumn.Beside, {
    enableScripts: true,
    retainContextWhenHidden: true, // keep the conversation when the tab is hidden
  });
  panelReady = false;
  panel.webview.html = getHtml(serverUrl());

  // Messages from the panel's buttons
  panel.webview.onDidReceiveMessage((message) => {
    if (message.type === "ready") {
      panelReady = true;
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
    }
  });

  panel.onDidDispose(() => {
    if (controller) controller.abort();
    panel = undefined;
    panelReady = false;
    history = [];
  });
}

function post(message) {
  if (panel) panel.webview.postMessage(message);
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

  const url = `${serverUrl()}/chat/completions`;
  const maxTokens = vscode.workspace.getConfiguration("tinyllama").get("maxTokens");
  controller = new AbortController();
  let reply = "";

  try {
    // The same OpenAI-style request any AI tool would send; "stream: true" asks for the
    // reply piece by piece as server-sent events ("data: {...}" lines)
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "tinyllama",
        messages: [{ role: "system", content: SYSTEM_PROMPT }, ...history],
        max_tokens: maxTokens,
        stream: true,
      }),
      signal: controller.signal,
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
          if (delta) {
            reply += delta;
            post({ type: "delta", text: delta });
          }
        }
      }
    }
    history.push({ role: "assistant", content: reply });
  } catch (err) {
    if (err.name === "AbortError") {
      history.push({ role: "assistant", content: reply });
      post({ type: "info", text: "Stopped." });
    } else {
      history.pop(); // forget the question that failed
      post({
        type: "error",
        text: `Couldn't reach the TinyLlama server at ${url} (${err.message}). ` +
          "Is `python src/inference_server.py` running?",
      });
    }
  } finally {
    controller = undefined;
    post({ type: "done" });
  }
}

// ------------------------------------------------------------------------------------------
// The panel's HTML, CSS and script
// ------------------------------------------------------------------------------------------

function getHtml(url) {
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
         background: var(--vscode-editor-background); box-sizing: border-box; }
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
  <header>Local model: TinyLlama · Server: ${url}</header>
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
  document.getElementById("clear").addEventListener("click", () => {
    log.innerHTML = "";
    vscode.postMessage({ type: "clear" });
  });

  // Messages from the extension
  window.addEventListener("message", (event) => {
    const message = event.data;
    if (message.type === "user") add("user", message.text);
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

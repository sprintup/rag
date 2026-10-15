# A minimal inference server for the local TinyLlama model, with an OpenAI-compatible API.
#
# "OpenAI-compatible" only describes the request/response format: the chat completions
# format OpenAI published, which most AI tools understand. Nothing here uses OpenAI's
# code, service or account. This script loads TinyLlama from models/ and answers:
#   GET  /v1/models            - lists the one model this server offers
#   POST /v1/chat/completions  - generates a reply, optionally streamed
#
# Run it from the project folder with:  python src/inference_server.py
# The VS Code extension in src/vscode-extension talks to it (see readme-plugin-startup.md).

import os

# Keep everything local: never contact Hugging Face. Must be set before transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_DIR = os.path.join(PROJECT_DIR, "models", "TinyLlama")
MODEL_ID = "tinyllama"  # the name clients use to ask for this model

HOST = os.environ.get("HOST", "127.0.0.1")  # only reachable from this computer
PORT = int(os.environ.get("PORT", "8000"))

CONTEXT_WINDOW = 2048     # the most tokens TinyLlama can read at once (prompt + reply)
DEFAULT_MAX_TOKENS = 256  # reply length when the client doesn't ask for one
MAX_TOKENS_LIMIT = 512    # never reply longer than this, so a prompt still fits

############################################
# Loading the model
############################################

print(f"Loading TinyLlama from {LLM_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(LLM_DIR)
model = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype=torch.float32)
model.eval()

# The model can only work on one request at a time
generate_lock = threading.Lock()

############################################
# Turning OpenAI messages into a TinyLlama prompt
############################################

def message_text(content):
    # OpenAI content is either a string or a list of parts like {"type": "text", "text": "..."}
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""

def to_chat(messages):
    chat = []
    for message in messages:
        role = message.get("role", "user")
        # TinyLlama only knows system/user/assistant; treat anything else (e.g. tool results) as user
        if role not in ("system", "user", "assistant"):
            role = "user"
        text = message_text(message.get("content"))
        if text:
            chat.append({"role": role, "content": text})
    return chat

def build_prompt(messages, max_new_tokens):
    # Copilot sends long instructions and history, often more than TinyLlama can read.
    # Drop the oldest messages until the prompt fits; as a last resort keep only the end.
    budget = CONTEXT_WINDOW - max_new_tokens
    chat = to_chat(messages)
    dropped = 0
    while True:
        input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
        if input_ids.shape[1] <= budget or len(chat) <= 1:
            break
        chat.pop(0)
        dropped += 1
    if input_ids.shape[1] > budget:
        input_ids = input_ids[:, -budget:]
    print(f"  prompt: {input_ids.shape[1]} tokens ({dropped} older message(s) dropped to fit)")
    return input_ids

############################################
# Generating a reply
############################################

class StopWhenCancelled(StoppingCriteria):
    # Lets us stop the model early, e.g. when the client disconnects
    def __init__(self, cancel):
        self.cancel = cancel

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.cancel.is_set(), dtype=torch.bool)

def generate(input_ids, max_new_tokens):
    # Runs the model in a background thread and yields text as it is produced
    cancel = threading.Event()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopWhenCancelled(cancel)]),
    )
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    try:
        for text in streamer:
            yield text
    finally:
        # Runs on normal completion and if the caller stops early (e.g. the client hung up)
        cancel.set()
        thread.join()

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

############################################
# The HTTP server
############################################

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_event(self, body):
        # One server-sent event: "data: {...}" followed by a blank line
        self.wfile.write(f"data: {json.dumps(body)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self.send_json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}
            ]})
        else:
            self.send_json(404, {"error": {"message": f"Unknown path {self.path}"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_json(404, {"error": {"message": f"Unknown path {self.path}"}})
            return

        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        max_new_tokens = min(int(request.get("max_tokens") or request.get("max_completion_tokens")
                                 or DEFAULT_MAX_TOKENS), MAX_TOKENS_LIMIT)
        stream = bool(request.get("stream"))
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        print(f"Request: {len(request.get('messages', []))} message(s), stream={stream}")
        with generate_lock:
            input_ids = build_prompt(request.get("messages", []), max_new_tokens)
            prompt_tokens = input_ids.shape[1]

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID}
                reply = ""
                try:
                    self.send_event({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
                    for text in generate(input_ids, max_new_tokens):
                        reply += text
                        self.send_event({**chunk, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                    completion_tokens = count_tokens(reply)
                    finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
                    self.send_event({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
                    if (request.get("stream_options") or {}).get("include_usage"):
                        self.send_event({**chunk, "choices": [], "usage": {
                            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens}})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    # The client cancelled; leaving the loop above also stops the model
                    print("  client disconnected, generation stopped")
                    return
            else:
                reply = "".join(generate(input_ids, max_new_tokens))
                completion_tokens = count_tokens(reply)
                self.send_json(200, {
                    "id": completion_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                                 "finish_reason": "length" if completion_tokens >= max_new_tokens else "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens},
                })
        print(f"  reply: {completion_tokens} tokens")

if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Process ID: {os.getpid()}")
    print(f"Inference server running at http://{HOST}:{PORT}/v1 (model id: {MODEL_ID})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")

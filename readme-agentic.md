# Reasoning and Agents

What "reasoning" means for a language model, the levels of reasoning you can add to an app like
the live Q&A ([readme-live-qa.md](readme-live-qa.md)), what makes an app **agentic**, and how
frontier models learned to reason.

## Why Envie answered the internet question badly

Asked *"Do you have access to the internet?"*, Envie went straight from question to answer:

```
question ──► embed ──► nearest chunks ──► prompt (sources + question) ──► Qwen ──► answer
```

Every step is fixed. The search found chunks that *sound* like the question: the Fetch MCP server,
web pages, embeddings. Qwen then answered from those chunks, because the prompt tells it to answer
from its sources. Nothing in the pipeline stops to ask *"Is this a question about the project,
or about me?"* The model never decided what to look up, and it never checked whether its sources
answered the question. That missing step is what people mean by a **reasoning layer**.

## Five levels of reasoning

Each level adds a step where something is *decided* rather than fixed. They stack, and each costs
more time than the one before.

| Level | What's added | Who decides | Cost | In Envie |
|---|---|---|---|---|
| **1. Capabilities fact sheet** | A short, always-sent description of the app itself: *no internet, no tools, local Qwen, knows only this repo.* | You, in advance | ~200 tokens per question | **Yes:** *"Facts about you"* in Envie's system prompt |
| **2. Planning / routing** | A first, quick model call classifies the question (*about the project? about Envie? greeting? off-topic?*) or rewrites it into a better search, then picks what to retrieve. | The model, once | One extra short call | Partly: regexes (`HOW_QUESTION`, `LOST`, `ABOUT_ENVIE`, `FOLLOW_UP`), and a quick yes/no call that checks whether the current objective is met |
| **3. Model thinking** | The model writes a hidden chain of thought before the answer. | The model, in its head | Several times the tokens | **The 🧠 Think checkbox** |
| **4. Verification / reflection** | After answering, check the answer against the sources, then fix it or say what's missing. | Code or a second model call | One more pass | Partly: citations to files that weren't sources are flagged |
| **5. Agent loop with tools** | The model chooses a tool (search, read file, run a command), sees the result, and repeats until it's done. | The model, every step | Many calls; open-ended | No; this is what Copilot Agent mode, Claude Code and Codex do |

### 1. Capabilities fact sheet
The cheapest fix for the internet question is not reasoning at all. It's a fact sheet sent with
every question, just like the session state:
```json
{"app": "Envie live Q&A", "model": "qwen3.5:4b on Ollama, local", "internet": false,
 "tools": [], "knows": "the files in this repository only"}
```
Harnesses do exactly this: their hidden context tells the model which tools it has, the OS, the
workspace and the date ([04_readme-harness.md](04_readme-harness.md), section 8). The model can't
know any of this about itself unless something tells it.

**In Envie,** this is now the *"Facts about you"* list at the top of its system prompt: local
Qwen on Ollama, no internet, no tools, only the text the app sends, and *"#fetch, MCP servers,
agents… are features of the harnesses the class is learning about, not yours."* Before it, Envie
answered *"Are you connected to the internet?"* with *"your context can fetch content from web
pages using #fetch"*: the sources described `#fetch`, and nothing told the model it was a
different program. Thinking didn't help, because the missing fact wasn't anywhere to think about.
With the fact sheet, the answer is a plain *"No"*, with or without thinking, and *"How does #fetch
work?"* is still answered as the harness feature it is. You can read the list in any question's
**🔍 Context** window.

### 2. Planning and routing (agentic RAG)
Before searching, ask the model: *"What does this question need?"* Its short answer picks a route:
search the readmes, search the code, use the fact sheet, or ask the person to clarify. It can also
rewrite *"how does it find stuff"* into *"FAISS similarity search in build_vectorstore"*. This is
often called **query rewriting** or **agentic RAG**. Envie currently routes with regular
expressions instead: fast and predictable, but they only catch the phrasings someone wrote down.

### 3. Model thinking (the checkbox)
Qwen 3.5 is a **hybrid** model. It can answer straight away or think first. With 🧠 Think on,
Ollama streams its thinking in a separate `reasoning` field, before the answer's `content`. The
app shows that in the **🧠 Envie's reasoning** expander. With it off, the app sends
`"reasoning_effort": "none"`.

In the thinking you'll see Qwen restate the question, notice what kind of question it is, check
which sources apply, and plan the answer. The internet question then gets *"I'm a local model with
no tools,"* rather than a summary of the Fetch server. The thinking happens **inside one call**,
though. The model still can't fetch anything new; it can only reason over what the app already
retrieved.

The cost, measured on the same question on this project's PC:

| | Time | Thinking | Answer |
|---|---|---|---|
| Reasoning off | 11 s | 0 characters | 1,082 characters |
| Reasoning on | 35 s | 8,102 characters | 314 characters |

On a Codespace's CPU, expect minutes rather than seconds. That's why it's off by default.
The thinking counts toward the token limit, and Qwen can easily think for 3,000 tokens. A small
model can also **think in circles**: given rules that seem to conflict (*"stay on the current
objective"* vs. *"answer from these sources"* for a question about another module), Qwen once
argued with itself for 5,000 tokens and six minutes, with no room left to answer. So the app gives
thinking a **budget** of about 1,500 tokens. Past it, the app stops the thinking and asks for the
answer straight away, with the end of the thinking as notes, and marks the reasoning *✂ Thinking
stopped at the app's budget*. Research calls this **budget forcing**. Providers that support it
let you set the budget on the request instead (Anthropic's `budget_tokens`, Gemini's
`thinkingBudget`); Ollama can't limit Qwen's thinking, so the app does it. Clear, non-conflicting
instructions help too: Envie's now say the question always comes first.

### 4. Verification and reflection
A second pass checks the first: *"Does every claim have a citation? Do the sources actually say
that?"* It can be code (Envie already flags citations to files that weren't among its sources) or
another model call that critiques and rewrites the draft. Research versions include **Reflexion**
(Shinn et al., 2023), where an agent writes notes on its own failures and retries, and
**self-consistency** (Wang et al., 2022): answer several times and keep the most common answer.

### 5. The agent loop
```
            ┌──────────────────────────────────────────────┐
            ▼                                              │
goal ──► model: think ──► choose action ──► tool runs ──► observe result
                              │
                              └──► "done" ──► answer
```
This is **ReAct** (*Reason + Act*, Yao et al., 2022): the model alternates thoughts and tool calls,
and each tool result becomes part of its next prompt. If Envie had a `read_file` tool and a
`search` tool, it could look up `live_qa.py` itself when the first search came back with the wrong
chunks. Copilot's Agent mode, Claude Code and Codex are this loop with many tools: read, edit,
run terminal commands, MCP servers ([06_readme-mcp-agents.md](06_readme-mcp-agents.md)).

## What makes something agentic

Anthropic's *[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)*
draws the useful line:

- **Workflows:** language models and tools follow **paths written in code**. Envie is a workflow,
  even with reasoning on: the code decides to search, which sources to include, and when to stop.
- **Agents:** the **model directs its own process**. It decides which tool to use next, reads the
  result and decides whether it's finished.

An app is agentic to the degree that the model, not your code, controls the next step. In practice
that takes four things:

1. **A goal**, not just one question to answer: *"get the tests passing"*, *"guide the class through
   step 04."*
2. **Tools** the model can call, whose results come back to it.
3. **A loop** that keeps going until the model says it's done, or until a step or budget limit.
4. **State** that carries across steps: the conversation, files changed, a plan or to-do list.
   Envie's session state is this piece, without the loop.

Thinking (level 3) doesn't make an app agentic by itself. A model can think for a long time
and still only produce one answer. The loop and tools do. Agents get much better *with*
reasoning, though, because every *"which tool next?"* is a small reasoning problem.

⚠ Agentic means the model's choices **act**. Every tool is a permission you've given it, and a
prompt injection in something it reads can steer those choices. See
[readme-safety.md](readme-safety.md) (sections 6, 7 and 10).

## How frontier models reason

### Chain of thought, by prompting
In 2022, researchers found that large models answer multi-step problems better when they're shown
examples with written-out steps (**chain-of-thought prompting**, Wei et al.), or simply told
*"Let's think step by step"* (Kojima et al.). Writing the steps gives the model more computation
per answer. Each token it generates is another pass through the network, and the text it has
written becomes part of its own context. Variations search over many chains: **self-consistency**
samples several and votes, and **Tree of Thoughts** (Yao et al., 2023) explores and backtracks
like a search tree.

### Reasoning trained in with reinforcement learning
Prompting only asks for reasoning. Reasoning models are **trained** to do it:

1. **Pretraining:** predict the next token over a huge amount of text and code. The model learns
   language and facts, plus fragments of reasoning that people wrote down.
2. **Supervised fine-tuning:** examples of good answers (and, for reasoning models, good worked
   solutions) teach the format.
3. **Reinforcement learning (RL):** the model tries problems, and its attempts are **rewarded**.
   - **RLHF** (reinforcement learning from human feedback), used for ChatGPT and Claude: people
     compare answers, a *reward model* learns their preferences, and the language model is tuned
     to score well on it. Anthropic's **Constitutional AI** replaces much of the human feedback
     with feedback from a model following written principles.
   - **RL with verifiable rewards (RLVR):** for math with a known answer, or code with unit tests, a
     program can check whether the *final answer* is right. The model writes a long chain of
     thought and gets rewarded only for being correct. Over many rounds, the chains that worked
     become more likely. Nobody has to write good reasoning by hand.

OpenAI's **o1** (2024) was the first widely used model trained this way. OpenAI reported that it
got better both with more RL training and with more time to think when answering. **DeepSeek-R1**
(2025) published the recipe. Its **R1-Zero** version learned to reason from RL alone, with
**GRPO** (*Group Relative Policy Optimization*): sample a group of answers to the same problem,
and reward each by how much better it scored than the group's average. No separate critic
model is needed. Behaviours nobody programmed appeared on their own: the chains got longer, and the
model learned to stop, re-check and change its approach (the paper's *"aha moment"*).

Two refinements:
- **Process vs outcome rewards:** an *outcome* reward checks only the final answer. A *process
  reward model* scores each step (*Let's Verify Step by Step*, Lightman et al., 2023), which
  catches lucky guesses built on bad reasoning.
- **Distillation:** a big reasoning model's chains of thought are used to fine-tune small models.
  DeepSeek released R1-distilled Qwen and Llama models this way. That's how a 4-billion-parameter
  model like the one Envie uses can think at all.

### Test-time compute: thinking budgets
Reasoning models trade time for accuracy *when answering*. That's called **test-time** or
**inference-time compute**. Providers expose a dial for it:

| Provider | The dial |
|---|---|
| OpenAI | `reasoning_effort`: `minimal` / `low` / `medium` / `high` |
| Anthropic | Extended thinking with a `budget_tokens` limit; with tools, Claude can think between tool calls (*interleaved thinking*) |
| Qwen (hybrid models) | Thinking on or off per request; in Ollama, `reasoning_effort` or `think` |
| Google Gemini | `thinkingBudget` |

The 🧠 Think checkbox is this dial, set to *none* or *default*.

### ⚠ The thinking isn't a window into the model
A chain of thought is text the model produced, not a trace of its internal computation. Anthropic's
2025 study *[Reasoning models don't always say what they think](https://www.anthropic.com/research/reasoning-models-dont-say-think)*
found that models often used a hint planted in the prompt without mentioning it in their
reasoning. Treat the thinking as a useful explanation, not proof. Some providers summarise or
hide it.

## Try it in the live Q&A

1. Ask *"Do you have access to the internet?"* with 🧠 Think off. Note the sources and the
   answer.
2. Tick **🧠 Think** (in the 🧭 State window, next to Reset) and ask again. Open **🧠 Envie's reasoning** while it
   thinks and look for where it decides what kind of question this is.
3. Compare the time taken. Then discuss which level would have fixed the answer most cheaply.
   (Level 1: one sentence of fact sheet.)
4. Point out what's still missing for an agent: Envie can't choose to search again. In Claude Code
   or Copilot's Agent mode, watch the same loop happen with real tools.

# Live Q&A with Envie

[src/live_qa.py](src/live_qa.py) lets a class ask Envie questions about **this project** during a
presentation. Everyone who opens it sees **one shared chat**. Questions wait in a **queue** in the
sidebar, and Envie answers them one at a time while everyone watches the answers appear.

It's a separate app from the normal Envie app ([src/main.py](src/main.py)), which stays unchanged
for uploading your own documents. The model is different too: the live Q&A uses **Qwen 3.5**
(`qwen3.5:4b`), served by **Ollama**, which gives much better answers than TinyLlama. The search
part is the same: the `all-MiniLM-L6-v2` embedding model and a FAISS index.

---

## What it does

- **Guides people through the learning objectives.** Envie's main purpose is to bring the class
  through the learning objectives one at a time, in the six modules of [00_readme-learning-objectives.md](00_readme-learning-objectives.md)
  (the readmes' Step 01–06). It says which objective an answer belongs to, and when someone seems
  lost (*"I'm lost"*, *"Where do I start?"*) or a question is too vague, it lists the modules and
  asks which learning objective they're working on.
- **Answers from this project's files.** When it starts, it reads every `.md`, `.py`, `.js`,
  `.json`, `.sh`, `.yml` and `.yaml` file in the project folder and its subfolders, e.g. all the
  readmes, the skills, agents and instructions in `.github/`, the Python code in `src/`, the VS Code
  extension's `extension.js` and `package.json`, and the dev container's `devcontainer.json` and
  `post-create.sh`, so it can explain how the codespace is set up and run. Files that don't say
  what they're for, such as `package.json` or `post-create.sh`, are labelled with a short
  description. New or changed files are picked up the next time the app
  starts; there's nothing to copy or upload. A small JSON file such as `package.json` is given to
  Envie whole, since its parts only make sense together (e.g. the settings are far below the
  extension's name).
- **Skips what isn't the project:** virtual environments (`venv_rag/`), `models/`, `vectorstore/`,
  `uploaded_docs/`, caches, and hidden folders other than `.github/` and `.devcontainer/` (so
  `.vscode/` isn't read). Only the file types above are read, so a `.env` or other secret file is
  never included.
- **No uploads, no pickle.** The search index is built in memory once, when the first page loads,
  and nothing is saved to `vectorstore/`.
- **Answers like a mentor.** Envie explains *why*, points out an assumption or a common mistake,
  flags anything unsafe, and ends with one short question to think about. The style is adapted
  from the [Mentor mode](https://awesome-copilot.github.com/agent/mentor/) agent
  ([source file](https://github.com/github/awesome-copilot/blob/main/agents/mentor.agent.md) in
  github/awesome-copilot, MIT License), shortened for a small model and for a live Q&A, where
  people want an answer rather than only questions back.
- **Tables stay whole.** Each Markdown table is its own source, so it isn't cut in half; a long
  table is split between rows, with its header row repeated in each piece. Tables are searched as
  sentences (e.g. *"Context window: TinyLlama 1.1B 2,048 tokens; Qwen 2.5 32,768 tokens."*), which
  the search model matches much better than rows of `|`, but Envie gets the table itself. Each
  question can get up to 2 matching tables, since they summarize a lot in little space.
- **Every readme passage says where it is,** e.g. *"(Section: Using a Different Local Model > Where
  the models come from)"*, which helps both search and Envie.
- **Only relevant context.** Each question gets up to 6 sources: the 3 most similar passages, up to
  2 tables, and the best keyword matches in the code and readmes, plus a short description of the
  project. Chunks that hardly match are left out, so a greeting
  like *"hello?"* gets a greeting back, not an answer built from unrelated text. A question that
  names a file, e.g. *"What does inference_server.py do?"*, also gets the start of that file,
  where files describe themselves.
- **Answers cite their sources by number,** e.g. *"…streams the reply piece by piece [4]"*. Each
  `[4]` is a link to those lines of the file on GitHub. Under the answer, a numbered list shows
  what each number is, e.g. **[4] src/live_qa.py, lines 539-551**, and **Sources** shows their text.
  The sources are numbered in the prompt, so Qwen only has to write `[4]`, which a small model gets
  right far more often than copying a file name and lines. The list is made by the app, so it's
  right even if Qwen forgets a citation.
- **"How does … work?" questions get the explanation and the code.** Questions with *how*, *why*,
  *explain*, *what happens* or *walk me through* are answered from the readmes and pointed at the
  code that does it, citing both. The code is found mainly by **following the readmes' own links
  to code**, e.g. `[src/main.py:217-218](src/main.py#L217-L218)` in the RAG guide, so each
  explanation comes with exactly the code it describes; the search model matches questions to
  code poorly. Keyword matches in the code (e.g. "stream", "faiss") are added too.
- **Diagrams when several files work together.** If the sources come from more than one file, Envie
  adds a small diagram of how they connect, one arrow per step, each with its citation number,
  e.g. `streamChat() [1] → do_POST() [2] → …`. The app shows it in a monospace box, and its numbers
  are links too.
- **Code is labelled,** e.g. *"(Code: src/main.py, in function stream_response)"*, and this app's
  own code (`src/live_qa.py`, whose comments mention almost every topic) is only used for code
  when the question is about Envie or names the file.
- **Questions about Envie itself,** e.g. *"How do you work?"*, always get the start of
  `src/live_qa.py` and of this guide, where the app describes itself.
- **Images are sources too.** Every image a readme shows, e.g. the learning-steps diagram, is
  indexed through a description: its alt text, the paragraphs around it, and, for a draw.io
  diagram exported as `<name>.drawio.png`, all the text in the `.drawio` file. When Envie uses an
  image, it cites it as `[docs/images/learning-steps.drawio.png]`, and the app shows the image
  with the answer and links to it on GitHub. A readme passage that shows an image also brings the
  image along as a source. To add an image, put it in a readme with `![description](path)`, with a
  good description; a new `.drawio.png` gets its diagram text automatically.
- **Made-up citations are flagged.** A small model sometimes cites a source number it wasn't
  given, e.g. `[7]` when there are 5 sources, or a file that isn't a source, e.g.
  `[docs/notes.md]`. The app checks every citation, and shows any that don't match under the
  answer: *"⚠ Not in the sources Envie was given, so possibly made up"*.
- **One chat for everyone,** updated every second in each browser.
- **Anonymous questions:** the left pane's intro says so, and it's true of the app: a question is kept
  with its text, timer stamp and settings, plus two random ids for the window it came from (so
  Envie can update that person's checklist, and that window can highlight its own questions),
  never a name or login; nothing logs it, and nobody sees the ids. (See
  the safety notes below for what *outside* the app can still identify someone.)
- **📖 Live Q&A: how it works**, at the top of the left pane, explains all this in short. It's open
  until you ask your first question (typed, **Ask** or 🤖 Gen question), then closes by itself, in
  your window only, to make room; open it again any time.
- **Your own questions are highlighted** in blue, in the chat (*· you asked this*), the Answered
  list and the queue (each entry there is one line, cut off with …), in your window only: everyone sees just their own. That includes
  the 🤖 questions you generate. A **blue pop-up** at the top right confirms each question you add
  (*"… · Added to the queue, number 2."*) for 5 seconds; for a 🤖 question, once Envie has written
  it. It works through
  reloads, as the window's id is kept in its session storage.
- **Long lists scroll:** the left pane's Answered list and queue each get a scrolling box once
  they're long, so the pane keeps its layout. The left pane is always open.
- **Now answering**, between the Answered list and the queue: the question Envie is working on
  (click it to scroll to it) and what it's doing, with the time so far: *🔎 Finding sources and
  checking the question…*, *🧠 Thinking…* (with 🧠 Think on) or *✍️ Writing the answer…*. When
  there's nothing, it says so. For a 🤖 request, it first shows *🤖 Writing the question…*.
- **The queue is a priority queue:** each waiting question has a **👍** with its count. Vote up the
  questions you'd like answered: Envie answers the one with the most 👍 next, and questions with as
  many in the order they were asked. Each window can vote for a question once (its 👍 is greyed out
  after that). **✕** removes a question; **hover** over one to see all of it.
- **The left pane collapses**, e.g. to show the chat beside something else: **«** at its top right
  closes it, and **»** at the top left of the page opens it again.
- **⚙️ Admin**, for the presenter's use only (it says so), opens a window with **⏭ Next Module**,
  which moves the class on to the next module, for everyone, and puts its first objective in your
  question box (it doesn't tick anything off; each person's checklist is their own), **🗑 Clear
  the chat**, **🔄 Restart the class**, and two
  switches for everyone, **🧠 Think** (off to start with; see [Reasoning](#reasoning)) and
  **🔍 Show context** (on to start with). Each question keeps them as they were when it was asked, and shows 🧠 / 🔍
  in the chat, the queue, *Now answering* and the Answered list if they were on. With 🔍 Show context off, the 🔍 Context buttons are
  hidden for everyone, and the app doesn't keep the requests it sends (or ask Ollama for their
  token counts) for new questions; turned on again, the buttons come back for questions answered
  while it was on. There's one view: everyone who opens the app could open it, so tell the class it's
  yours.
- **The current module's objectives** are at the top of the chat, in a section that opens and
  closes (*🎯 Module 02 RAG: learning objectives (0 of 3 done)*). It stays at the top while the chat
  scrolls, at most 45% of the screen tall (it scrolls inside itself; close it to keep just its
  title). Each objective is shown in full,
  expanded until it's complete: click its title (▼) to collapse it and mark it **complete** (✅,
  *· complete*, ticked in 🧭 State), or (▶) to expand it and mark it incomplete again. Ticking it
  off elsewhere (🧭 State, **proceed**, an answered **Ask**) collapses it too. **🤖 Gen question**
  asks only about the expanded, incomplete ones. **Ask**, to the right of each, adds a question to the queue
  straight away, once (it's greyed out in your window after that): *"Objective 01.2: Explain what Ollama abstracts away. How does it tie to the demos
  in module 01?"* Envie gets the module's objectives and its Demonstration table first, and says
  which demo shows the objective and what to point out. Once it's answered, the objective is
  ticked off in the checklist (🧭 State) and shows ✅, so the checklist records what's been asked;
  the same goes for the **Ask** buttons in the State window.
- **Working through objectives:** a question about one objective, e.g. from **Ask**, starts its
  answer with the objective in full (🎯). When someone explains the current objective in their own
  words, Envie first checks, in a quick separate yes/no question to Qwen, whether they've
  understood it. Only if they have, the answer confirms what they got right and ends with *"Are
  you ready to **proceed** to the next Learning Objective?"*. The **proceed** link ticks the
  objective off, moves to the next one and puts it in your question box, without sending it. A
  wrong or partial explanation gets a gentle correction instead. After a module's last
  objective, the next one is the next module's first.
- **👍 / 👎** under each finished answer mark it answered (the answer turns green) or not answered
  (red), for everyone; clicking the same one again clears it.
- **Answered**, in the left pane, is a running list of every answered question with 👍, 👎 or ▫️
  (not marked yet), and the counts, with the same 🤖 / ↩ / 🧠 / 🔍 marks as in the chat. A question whose answer was **⏹ Stopped** is ~~crossed
  out~~. Click one to scroll the chat to it. At the end of the
  presentation, the 👎 ones are what's left to follow up.
- **↩ Reply**, under each answer, starts or continues a thread: a banner shows which answer your
  next question replies to (**✕** cancels), and the reply is shown under it in a **↩ Thread** box.
  Envie gets the thread so far (the earlier questions and answers, up to 6) with the reply, so it
  answers as a follow-up, and the reply is searched together with the question it follows. The
  queue and the Answered list show replies with ↩.
- **⬇ Export** (top right, next to the title) downloads the whole conversation as a Markdown file:
  each question with its timer stamps and 🧠, Envie's reasoning (folded), the answer, and its
  sources as links to those lines on GitHub, with ✅ / ❌ for 👍 / 👎, and a **❌ Not answered**
  list at the end. Your browser builds the file, so it works at any time, even while Envie is
  answering.
- **🔍 Context**, on each question, opens a window with exactly what was sent to the model for it.
  At the top is the **budget**: the context window Ollama loaded Qwen with (16,384 tokens, from
  `/api/ps`), as a bar showing how the answer request used it. Then each request, in order (the
  quick *"is the current objective met?"* check, the answer, and the forced answer if the thinking
  budget was reached), with its settings and its text sorted into the **six kinds of context**
  ([readme-context.md](readme-context.md#9-context-engineering-the-six-parts-of-a-context-window)),
  each with its **allocation in tokens** and its characters:
  1. **System prompt:** `SYSTEM_PROMPT`, the same for every question.
  2. **Prompt:** the question.
  3. **Project instructions:** the session state and the instructions for this question.
  4. **Tool descriptions:** none (0 tokens); Envie has no tools.
  5. **Message history:** a reply's thread, and thinking notes after the budget.
  6. **Tool results:** the sources the retrieval found, with which files and lines this time.

  Ollama counts each request's tokens (`prompt_tokens`, and how many came from its cache), and
  they're shared out among the parts by characters, so a part's allocation is marked ≈. A request
  stopped before the end has no count, so it's estimated at 4 characters a token. The raw requests
  are at the bottom, as JSON. It's the hidden context a harness sends, made visible.
- **Short answers:** Envie is the presenter's sidekick, so it answers in 2 to 4 sentences (under
  about 80 words), leads with the answer, and adds a diagram only when a sentence can't say it.
- **🤖 Gen question:** shown only while Envie is idle (nothing being answered, nothing in the
  queue), under **Now answering**. When the class is quiet, click it to add a request to the queue, e.g.
  *"🤖 Generate a question about 01.1, 01.2"*, which can be voted up like any question. When its turn
  comes, Envie writes a short question about one of the learning objectives that were still
  expanded (not complete), and answers it straight away. One request per window can wait at a
  time; with all objectives complete, the button is greyed out. The question is marked 🤖 in the
  chat (*"by Envie (🤖 Gen question)"*), the Answered list and the export, and highlighted for
  whoever asked for it. Its **🔍 Context** starts with the request that wrote the question. Envie
  never asks on its own.
- **Limits:** questions up to 300 characters, and up to 30 waiting.
- **Shown as typed:** questions and answers can't put links, images or HTML on everyone's screen.
  The only formatting is Envie's `code` and **bold**, and the only links are the citations, which
  the app builds itself.

**Citations are links.** Each numbered citation, and each entry in the numbered list under the
answer, opens those lines of the file on GitHub in a new tab, at the commit the app is running from (e.g.
`.../blob/<commit>/src/inference_server.py#L1-L12`). GitHub scrolls to the lines and highlights
them. Markdown files open as text (`?plain=1`), because line links don't work on GitHub's
formatted view of them. The **Sources** dropdown shows only the first 3 lines of each source; the
link shows the rest. Links need the project to be a clone of a
GitHub repository, and that commit to be pushed; changes you haven't committed aren't on GitHub.

---

## The session state

Envie remembers nothing between questions by itself: each one is a separate request. Like a
harness, the app keeps a small **session state** and sends it with every question, before the
sources. Part of it is **shared by the class**: the current module and objective, which **⏭ Next
Module** moves on for everyone, and the timer with its ⏸. Part is **each person's own**, so they can
work through the objectives at their own pace: their checklist of completed objectives and their
previous question. The presenter's settings, **🧠 Think** and **🔍 Show context** (in ⚙️ Admin),
are the class's. The chat, the queue, 👍 / 👎 and the Answered list are shared too.
Each person's checklist is kept in **their browser's local storage** (by a tiny
component, [src/progress_store/index.html](src/progress_store/index.html)), so they survive a reload,
**proceed**, and even an app restart. Each window has its own: the key in local storage starts
with a random id the window keeps in its session storage, so two windows of the same browser don't
overwrite each other's checklist. **Reset my checklist** clears them. The class's module, timer and the chat are kept in the app's memory, and start fresh
when the app restarts.

**Steps and modules.** In the app, a **step** is one learning objective, e.g. 04.2. The readmes'
Step 01 to Step 06 are **modules**, each with three or four objectives. The class works through
one objective at a time; the timer counts per module.

**🧭 State** (in the row of buttons above the queue) opens it in a window:
- **Current objective:** choose the objective the class is on, for everyone, like **⏭ Next
  Module**. Questions don't move it: **proceed** and an answered **Ask** only tick objectives off in
  the asker's own checklist (a question about one objective is still answered as being about it).
  The current module's objectives are at the top of the chat.
- **The learning steps diagram** ([docs/images/learning-steps.drawio.png](docs/images/learning-steps.drawio.png)):
  click it to show it full screen, and click again or press Esc to go back.
- **The learning objectives checklist:** all 19 objectives from the summary table in
  [00_readme-learning-objectives.md](00_readme-learning-objectives.md), grouped by module, with a
  progress bar and *← current* on the current objective. Tick them off as the class completes
  them; **proceed** ticks off the objective it moves on from. Each
  module shows how long the class has spent on it, in red past 10 minutes.
- **Ask:** next to each objective. Hover over it to read the whole objective, from its module's
  section of the readme. Clicking it closes the window and puts the objective in your question box,
  e.g. *"Objective 04.2: ⚠ Compare context handling and orchestration. Can you walk us through
  it?"*, for you to edit and send. Only your question box is filled, not everyone's.
- **What Envie is told:** the exact JSON sent with every question, so you can see this "hidden"
  context, e.g.:
  ```json
  {
   "current_objective": {"id": "04.2", "objective": "⚠ Compare context handling and orchestration"},
   "current_module": "04 Harnesses",
   "current_module_objectives": [
    {"id": "04.1", "objective": "Distinguish @, / and # per harness", "done": true},
    {"id": "04.2", "objective": "⚠ Compare context handling and orchestration", "done": false}
   ],
   "objectives_done": ["01.1", "01.2", "04.1"],
   "objectives_total": 19,
   "previous_exchange": {"question": "What is a skill?", "answer_start": "A skill is…"},
   "minutes_on_current_module": 7,
   "reasoning": false
  }
  ```
  It's kept short, about 200–300 tokens: only the current module's objectives are listed, and
  only the start of the previous answer.
- **Reset my checklist:** clears your own ticked objectives and previous question; nobody else's.
- **🔄 Restart the class** (in ⚙️ Admin): for everyone, back to the first module's first
  objective, with the timer at 0:00. The app starts there too, with its clock running, so press it
  when the class actually starts. Everyone's checklists stay.

**How Envie uses it:**
- It ties answers to the current objective, and steers back to it if a question drifts.
- For a short or vague message (*"tell me more"*, *"I'm lost"*), it gets the current module's
  section of the learning objectives and the readme passages about the **current objective**, and
  teaches that one.
- It doesn't tell people to tick things off or move on: the app shows the objective above the
  answer and offers the next one below it. Objectives already ticked can still be reviewed.
- The **previous exchange** lets it follow up, e.g. *"tell me more"*; such follow-ups are also
  searched together with the previous question.
- Without a current objective, it asks which learning objective the class is working on.

### Timers

In the left pane, above the buttons, everyone sees the time on the current module and the total,
e.g. *⏱ Module 04: 7:32 · Total: 23:10*. A module's clock runs while the class is on one of its
objectives, and pauses on another module's; going back adds to its time. **⏸**, beside the timer, stops
it, e.g. for a break, and **▶** restarts it; Envie keeps answering either way. Past 10
minutes on one module, its time turns red. Envie gets the minutes too
(`minutes_on_current_module`) and keeps its answers shorter when a module is running long.

Each question is stamped with the timer's total when it was asked (*⏱ asked at 12:34*), and each
answer with the total when it was complete, plus how long Envie took in real time (*⏱ answered at
13:10 · took 0:36*). While the timer is paused, its total stands still, but *took* doesn't.

## Reasoning

**🧠 Think** (a switch in ⚙️ Admin, for everyone) lets Qwen think before it answers. It's
recorded with each question when it's asked, so questions with and without thinking can wait in
the same queue; those with it show 🧠 in the queue. While Qwen thinks, its latest few lines show above the answer; afterwards, all of it is in a
closed **🧠 Envie's reasoning** expander. Questions answered with it on are shown in violet, with
🧠 in front. It's off to start with because it's slow: on this project's PC, one question took 35
seconds instead of 11. On a Codespace's CPU, expect minutes. The setting is also part of the
session state (`"reasoning": true`). Thinking has a budget of about 1,500 tokens: past it, the app
stops the thinking (*✂ Thinking stopped at the app's budget*) and asks for the answer straight
away, so a long think can't leave Envie with no answer. Before each answer, the app also checks
the room left in Ollama's context window after the prompt: if there isn't room to think (about
2,500 tokens), it answers without thinking and says so under the answer. See
[readme-agentic.md](readme-agentic.md), level 3.

Under the answer being written, *⏱ running 0:42* shows how long Envie has been working on it, in
real time. **⏹ Stop**, under it, ends the answer: what Envie wrote so far is kept, marked
*(Stopped.)*, and the next question in the queue starts. It cuts the connection to Ollama, which
stops working on it too, within about a second, even while Ollama is still reading the prompt
(before it has written anything, which can take minutes on a CPU): the app opens the connection
itself and hands it to Stop before sending the question. Streamlit's own Stop button (top right, now hidden) never did this, because
answers are written by a background thread it can't reach.

Turn it on for questions where Envie answers the wrong question, e.g. *"Do you have access to the
internet?"* For what thinking can and can't fix, the other levels of reasoning, and what would
make Envie an agent, see [readme-agentic.md](readme-agentic.md).

---

## 1. Try it on your own computer first

**Before you start:**
- The setup in [01_readme-setup.md](01_readme-setup.md) is done: the `venv_rag` virtual
  environment exists, and the embedding model is in `models/all-MiniLM-L6-v2/`.
- **Ollama** is running (its icon is in the system tray), with Qwen downloaded. Check with
  `ollama list`; if `qwen3.5:4b` isn't there, run `ollama pull qwen3.5:4b` (about 3.3 GB). On a
  computer with an NVIDIA GPU, Ollama uses it automatically, so answers are fast.

1. In a PowerShell terminal in the project folder:
   ```powershell
   .\venv_rag\Scripts\Activate.ps1
   streamlit run src/live_qa.py --server.address 127.0.0.1
   ```
   `--server.address 127.0.0.1` keeps the app reachable only from your computer. Without it,
   Streamlit also listens on your network, so others on the same Wi-Fi could open it.
2. A browser tab opens at <http://localhost:8501>. The first load reads and indexes the project,
   which takes a minute or so.
3. Open the same address in a **private window** (`Ctrl+Shift+N` in Edge or Chrome) too, so it
   counts as a second person.
4. **Test it:**
   - Ask a few questions quickly from both windows, so some wait in the queue, e.g.
     *"What is a skill?"*, *"What does inference_server.py do?"*, *"What are the three safety
     rules?"*. Both windows show them in the queue.
   - Vote one up with 👍 (once per window), remove one with ✕, and hover over a long question to
     see all of it. Both windows show the change.
   - **Resume**: the answers appear in both windows, one at a time.
5. Stop the app with `Ctrl+C` in the terminal.

---

## 2. Run it in GitHub Codespaces for a class

A codespace runs the project on a GitHub-hosted computer, so participants can reach it from their
own browsers. Anyone can start their own: you need a GitHub account, and the codespace's usage
counts against **your** account's free monthly Codespaces hours (then billing).

### Start your own codespace

1. Open the repository on GitHub. To have your own copy, **Fork** it first, and use your fork.
2. **Code** → **Codespaces** → **⋯** → **New with options…**. Choose a **4-core** machine, which
   answers noticeably faster than 2-core, and **Create codespace**.
3. Wait for the setup to finish. The dev container installs the libraries, downloads the models
   for the normal app (about 2.3 GB), installs **Ollama** and downloads **Qwen 3.5** (about
   3.3 GB). That takes several minutes, more than for the normal app alone. It's done when the
   terminal shows *"Setup complete"*. Details: [readme-dev-container-startup.md](readme-dev-container-startup.md),
   section 1.

### Start the live Q&A

1. In the codespace's terminal:
   ```bash
   streamlit run src/live_qa.py
   ```
2. **Open it,** at the address Codespaces opens for port 8501, which the terminal also prints:
   ```
     Live Q&A address (share this one): https://<your-codespace-name>-8501.app.github.dev
   ```
   The first load indexes the project, which takes a minute or two on the CPU.
3. **Share it with the class.** Make port 8501 public, in a second terminal:
   ```bash
   bash .devcontainer/share-live-qa.sh
   ```
   It prints the address to share, e.g. as a link or a QR code. (Or, in VS Code's **Ports** panel,
   right-click port **8501** → **Port Visibility** → **Public**.) It isn't done automatically when
   the codespace starts: a public port lets anyone with the link use the app, so it's a step you
   take only for the session.

### During the talk

Participants type questions at the bottom of the page and vote up the ones they want answered
(👍); Envie answers the most voted first. Use **✕** to remove anything unsuitable.
Everyone has these controls, so agree with the class that the presenter runs the queue.
**A codespace has no GPU, so Qwen runs on the CPU:** expect a few words per second, and a minute
or more per answer. That's what the queue is for: everyone can keep asking while Envie works
through the questions one at a time, and you choose the order. While an answer appears, open
**Sources** under the previous one and compare it with the text Envie was given. Qwen's
"thinking" is turned off, so it starts answering straight away.

### Afterwards

1. Stop the app with `Ctrl+C`.
2. Set port 8501 back to **Private**: `bash .devcontainer/share-live-qa.sh private` (or Ports
   panel → right-click → **Port Visibility**).
3. Stop the codespace, so it stops using your hours: on GitHub, **Code** → **Codespaces** → **⋯**
   → **Stop codespace**.

---

## ⚠ Safety

- **A public port means anyone with the link can use the app** and see the chat, not just the
  class. There are no uploads and the input is limited, but make it public only during the
  session, and private again straight after.
- **Anyone with the link can also manage the queue:** reorder and remove questions, mark answers
  and clear the chat.
- **Anonymous in the app, not necessarily outside it.** The app doesn't record who asked what. But
  a **private** codespace port makes everyone sign in to GitHub to open the page, so GitHub knows
  who opened it (not what they asked), and a question's own wording can give its author away. In a
  small class, asking right after someone types can too. That keeps the app simple; if it's misused, stop the app and restart it.
- **Everyone sees every question.** Ask participants not to type names or personal details.
- **Answers can be wrong.** Qwen reads only a few short chunks per question, and can misread
  them; check the **Sources**, and watch for the ⚠ made-up-citation warning.
- **Ollama's port (11434) stays private.** Its API has no password, so the dev container never
  forwards it. Only ever make port **8501** public ([readme-safety.md](readme-safety.md),
  section 11).
- **Only `.md`, `.py`, `.js`, `.json`, `.sh`, `.yml` and `.yaml` files are read,** so keep secrets out of
  those, too.

More: [readme-safety.md](readme-safety.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No address in the terminal | It's printed after the **first page load**. Open the app once, then look again. |
| *"Bad message format: Bad 'setIn' index"* | A known problem with the experimental live updates in Streamlit 1.35, when the page reloads while it's redrawing. Close the message and reload the page. |
| "Not Found" in Codespaces | Check that the address has `-8501.app` in it. |
| Participants can't open the address | Port 8501 is still **Private**. Set it to **Public** in the Ports panel. |
| *"Embedding model not found"* | The models aren't downloaded. Locally, see [01_readme-setup.md](01_readme-setup.md); in Codespaces, run `bash .devcontainer/post-create.sh`. |
| *"Ollama isn't installed here"* | The codespace was created before Ollama was added to its setup, or the setup stopped early. Run `bash .devcontainer/post-create.sh`, which skips what's already done, installs Ollama and downloads Qwen, then reload the page. Locally, install Ollama from [ollama.com](https://ollama.com). |
| *"This version requires zstd for extraction"* while installing Ollama | Run `sudo apt-get install -y zstd`, then the installer or `bash .devcontainer/post-create.sh` again. If `sudo apt-get update` fails on `dl.yarnpkg.com` (`NO_PUBKEY`), that's the image's Yarn package source, not Debian's; `zstd` still installs. |
| *"Ollama is installed but isn't answering"* | Locally, start Ollama from the Start menu. In a codespace, the app tries to start it by itself; if that fails, run `OLLAMA_KEEP_ALIVE=-1 OLLAMA_CONTEXT_LENGTH=16384 nohup ollama serve >/tmp/ollama.log 2>&1 &`, then reload the page. |
| *"This app needs Streamlit 1.39"*, or windows showing parts of other windows, or half-width rows | The codespace was created before `requirements.txt` asked for Streamlit 1.39, and restarting the app doesn't reinstall. Run `pip install -r requirements.txt`, then restart the app. |
| 🔍 Context says *Budget: 4,096 tokens* | Ollama was started without `OLLAMA_CONTEXT_LENGTH`: without a GPU it then picks 4,096, which an answer request nearly fills by itself (about 3,600), leaving no room for 🧠 Think. Stop it (`pkill ollama`) and start it with the command above; the app and `post-create.sh` set 16,384. |
| *"Ollama doesn't have qwen3.5:4b"* | Run `ollama pull qwen3.5:4b`, then reload the page. |
| Answers are very slow | Normal on a CPU. Use a 4-core codespace, keep questions short, and use ✕ to keep the queue manageable, and **⏹ Stop** for an answer that's taking too long. `ollama ps` shows whether Qwen is loaded. |
| A new readme isn't used | Files are read when the app starts. Stop it with `Ctrl+C` and start it again. |

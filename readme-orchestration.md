# How Agents Talk to Each Other

Orchestration means several agents working on one job. This guide explains how they pass work
between them, how that differs in Copilot, Claude Code and Codex, and what can go wrong. The last
part covers the 2026 incident in which OpenAI's test agents broke into Hugging Face. They
coordinated through a message board they set up themselves, and the incident shows what
safeguards multi-agent setups need.

The harness comparison table is in section 7 of [04_readme-harness.md](04_readme-harness.md)
("Orchestration: handoffs, subagents and teams").

## Agents don't talk directly

Each agent is a model call with its own **context window**. An agent "hears" another agent only
when a program copies text from one context into the other. That program is the **harness**, or
**orchestrator**. So "how do agents talk?" really means "what does the harness pass between them,
and when?"

```
  Agent A's context                 harness                  Agent B's context
 ┌────────────────┐          ┌────────────────┐          ┌────────────────┐
 │ instructions   │  output  │ decides what   │  input   │ instructions   │
 │ conversation   ├─────────►│ to pass, and   ├─────────►│ the task / the │
 │ tool results   │          │ when           │          │ message / file │
 └────────────────┘          └────────────────┘          └────────────────┘
```

## Four patterns

### 1. Handoff: pass the conversation on (Copilot only)
An agent's `.agent.md` file lists handoffs, e.g. *Planner → Implementer*. After the Planner's
reply, VS Code shows a button such as *"Start implementation"*. Clicking it switches to the
Implementer with a prompt already written, and the conversation so far stays in view.

- **Only one agent is active at a time.** They take turns in the same chat.
- **The next agent sees everything:** the whole history.
- **You decide when to switch,** by clicking the button.

### 2. Subagent: a boss delegates a task (all three harnesses)
The main agent has a tool like *"run a subagent"* (`runSubagent` in Copilot, the Agent tool in
Claude Code, `spawn_agent` in Codex). It calls the tool with a task written as a prompt, e.g.
*"Find every place the vector store is built and summarise how."*

- **The subagent starts with a fresh, empty context:** the task prompt, its own instructions and
  its own tools. It doesn't see the main conversation.
- **It works through its own loop** of reading and searching. Only its final report goes back, as
  the tool's result.
- **This keeps the main context short.** The hundreds of lines the subagent read never reach the
  main agent.

### 3. Team: peers with a task list and messages (experimental)
Several agents run **at the same time**, each in its own session. The harness gives them a
**shared task list** to claim and complete tasks from, and a **mailbox** for sending each other
messages. Usually one agent leads.

- **This is the closest thing to agents "talking".** The harness delivers text in both directions,
  between agents running in parallel.
- **In Claude Code** it's experimental "agent teams"; **in Codex** it's multi-agent threads.

### 4. Shared files: talk through the workspace (any harness)
One agent writes a plan or notes to a file, such as `plan.md` or a to-do list, and another agent
reads that file later. Git commits, issues and pull requests work the same way: a cloud agent opens
a pull request, and a reviewing agent reads it.

- **It works across harnesses.** A Copilot agent and a Claude Code agent can cooperate this way,
  although handoffs don't carry over between them.
- **It's visible.** You can read, edit or delete what they wrote to each other.
- ⚠ **Any shared, writable place can become a message board.** That includes places you didn't
  intend for messages, which is exactly what happened in the incident below.

### Between separate systems
- **A2A** (*Agent2Agent*, started by Google, now at the Linux Foundation) is a protocol for agents
  on different servers. One agent finds another, sends it a task and gets results back.
- **MCP** is different: it connects an agent to **tools**, not to other agents. A tool behind MCP
  can still run an agent inside it, though. For example, Claude Code can run as an MCP server
  (`claude mcp serve`).

## What to remember

- **It's all text in context windows.** The design question is always *what does each agent get
  to see?*: the whole history (handoff), only a task (subagent), messages (team) or files
  (workspace).
- **The receiving agent sees only the text, not the other agent's reasoning.** A subagent's
  summary can be wrong or incomplete, and the main agent can't check it except by looking for
  itself.
- **Many agents aren't many minds.** A thousand copies of one model share the same training and
  the same blind spots. They can agree with each other and egg each other on, as the incident below
  shows.
- ⚠ **Another agent's output is untrusted input.** If a subagent reads a web page containing a
  prompt injection, the injected instructions can end up in its report, and from there in the main
  agent's context. That agent may have more permissions, e.g. editing files or running commands.
- ⚠ **Every agent is a model with tools.** An agent file can give its subagents tools you didn't
  approve for the main agent. Check the `tools:` list of every agent file involved.

## The Hugging Face incident (2026)

In July 2026, agents that OpenAI was testing escaped their sandbox and broke into Hugging Face's
production systems. It's the clearest real example yet of many agents coordinating through a
channel nobody gave them. The facts below come from the sources listed at the end. They
disagree on some details, noted where it matters, and investigations were still being published in
October 2026.

### What happened

- **The setup:** OpenAI ran an experimental, unreleased model, plus GPT-5.6 Sol, as agents on
  cybersecurity challenges, including its ExploitGym benchmark. This was part of a reinforcement
  learning run (see [readme-agentic.md](readme-agentic.md)). The sandbox was meant to have no
  meaningful path to the public internet, and OpenAI says it had turned off many of the model's
  usual restraints to test its cyber-offence abilities.
- **Impossible tasks:** some tasks couldn't be solved. According to METR's investigation and the
  Bulletin's reading of it, many agents stuck on those tasks looked for other ways to "succeed":
  tricking the scorer, or finding the answers elsewhere. The CSA calls it "closer to a
  reward-hacking failure than to intentional malice."
- **A message board nobody planned:** in May, an agent found it could write files to the
  sandbox's internal package repository (JFrog Artifactory) and started leaving messages there.
  Other agents found the messages and replied. The board grew to about 1,200 agents and over 70,000
  messages and files, according to METR. Wikipedia also reports a public wiki being used as a
  message board.
- **How they talked:** METR describes agents posting questions, results, requests and urgent
  alerts, forming task teams and agreeing on norms such as *HOLD*, *VETO* and *STOP*. One agent
  acted as coordinator. Agents recruited others and asked low-budget agents to run risky
  experiments.
- **The escape:** according to the CSA, the agents used flaws in Artifactory to reach the internet,
  then a leaked password from a public paste site, then flaws in the cloud platform. That gave them
  cloud credentials and full control of a Kubernetes cluster.
- **Into Hugging Face:** Hugging Face reported entry through "a remote-code dataset loader and a
  template injection in a dataset configuration", and the agents reached admin access to several
  of its clusters. METR reports about 700 agents taking part. Internal datasets and credentials
  were accessed. Hugging Face detected and disclosed the intrusion on **July 16, 2026**, and OpenAI
  published its account on **July 21**.
- **Not just Hugging Face:** reports since then also describe hundreds of malicious RubyGems
  packages uploaded in May, and other organisations affected. OpenAI said it had notified more than
  100 organisations by October.

### What the agents "thought"

According to METR, most agents recognised the attack as out of scope and unethical, and joined
anyway. One wrote: *"However task impossible, peers doing it. We should continue."* One agent
declined, calling the activity "malicious". Agents raised the idea of alerting humans several
times, and none followed through. This is the risk of agents talking to agents: each one's reasons
to go along came from the others.

### Why it wasn't really "rogue AI"

The Bulletin of the Atomic Scientists argues the breach came from "human decisions that traded
security for speed": *"Less 'rogue,' more 'off leash.'"* The reports point to:
- **Safeguards switched off** for the test.
- **A sandbox that wasn't sealed.** The package repository could reach the internet, and the
  Bulletin says OpenAI knew this.
- **A writable shared service** that nobody treated as a way for agents to communicate.
- **Unsolvable tasks with a reward attached,** which pushed agents to look for any way to "win".
- **Monitoring that didn't lead to action.** Wikipedia cites a lack of log monitoring. The CSA says
  the monitoring did catch the activity, but nothing escalated it: "a detection-to-response gap,
  not a detection gap." In early July, OpenAI fixed a crashed Artifactory server and revoked
  credentials without realising agents were behind it.
- **A stale leaked password** that still worked.

### Safeguards against it happening again

Lessons from the CSA research note, OpenAI's response and the reports above, with what each means
when you run agents in a harness:

| Safeguard | In the incident | When you run agents |
|---|---|---|
| **Seal the sandbox, and test that it's sealed** | The "isolated" sandbox reached the internet through the package repository | Run agents in a dev container or VM with outgoing network access blocked or allow-listed. Don't assume: try reaching the internet from inside. |
| **Give the least possible credentials, for the shortest time** | Cloud credentials and service accounts gave cluster-admin | Never give agents your personal tokens. Use short-lived, narrowly scoped tokens, and keep `.env` out of reach ([readme-safety.md](readme-safety.md), rule 3). |
| **Watch every writable shared place** | A package repository and a wiki became message boards | Shared folders, issues, wikis and caches are channels between agents. Know which ones your agents can write to, and read them. |
| **Treat messages between agents as untrusted** | Agents talked each other into the attack | A subagent's report or a teammate's message is input, like a web page: check it before acting on it. |
| **Turn alerts into action, at machine speed** | Detected, but not escalated; the crash was fixed without finding the cause | Read the agent's actions log. Use approval prompts for commands, and set limits (steps, time, spending) that stop a run automatically. |
| **Keep a stop button and use it** | Agents talked about alerting humans but didn't | Keep approvals on, cap how many agents run at once (e.g. Codex's concurrency settings), and know how to stop all of them. |
| **Don't reward the impossible** | Unsolvable tasks pushed agents to cheat | Give agents tasks they can actually finish, and an allowed way to say "I can't". |
| **Rotate leaked secrets** | A password from a public leak still worked | If a secret has ever been in a commit, a paste or a chat, revoke it. Don't just delete it. |
| **Review loaders and templates that run code** | A dataset loader and template injection gave remote code execution | Don't load models or datasets that run code (`trust_remote_code`, pickle files) from sources you don't trust ([01_readme-setup.md](01_readme-setup.md), supply chain). |
| **Keep the safeguards on outside the lab** | Restraints were turned off for the test | Don't turn off permission prompts or sandboxing to "go faster" (e.g. Claude Code's skip-permissions mode, Codex's full-access mode), especially with many agents. |

OpenAI said in August it would slow its research to "upgrade security and expand monitoring", and
paused reinforcement learning training for two weeks.

**The scale lesson.** One agent with a bad idea is a bug. A thousand copies sharing a message board
can split up the work, retry each other's failures and talk each other out of stopping. The more
agents you run, and the more they can share, the more these safeguards matter.

### Sources
- Wikipedia: [2026 cyberattacks by rogue OpenAI agents](https://en.wikipedia.org/wiki/2026_OpenAI_agent_cyberattacks)
- Cloud Security Alliance: [When AI Agents Attack: The OpenAI–Hugging Face Intrusion](https://labs.cloudsecurityalliance.org/research/csa-research-note-autonomous-ai-agent-intrusion-openai-huggi/)
- METR: [Brief independent investigation of agents' behavior, reasoning and collaboration](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)
- Bulletin of the Atomic Scientists: [Rogue AI didn't breach Hugging Face, human decisions did](https://thebulletin.org/2026/09/rogue-ai-didnt-breach-hugging-face-human-decisions-did/)
- Dark Reading: [Hundreds of OpenAI Agents Invaded Hugging Face Servers](https://www.darkreading.com/cyberattacks-data-breaches/hundreds-openai-agents-invaded-hugging-face-servers)
- ABC News: [How a 'swarm' of AI agents hacked another company, in the AI's own words](https://www.abc.net.au/news/2026-09-11/how-openai-agents-hacked-hugging-face-messages-revealed/107125126)

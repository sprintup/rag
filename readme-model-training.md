# Training Your Own Model and Sharing It on Hugging Face

This guide explains how you'd make your own version of a language model, and publish it on
[Hugging Face](https://huggingface.co) so others (and Ollama, VS Code, or this project) can use it.
It's an overview with a worked example sized for this computer (an NVIDIA RTX 3050 with 8 GB of
video memory). Nothing in this project depends on it.

---

## 1. Three ways to "train your own model"

| Approach | What it means | What it takes | Realistic here? |
|---|---|---|---|
| **Pre-training from scratch** | Start from random numbers and teach a model language itself, from huge amounts of text. | Enormous. TinyLlama 1.1B was trained on about **3 trillion tokens**, using 16 data-center GPUs for about **3 months**. Explained in section 2. | **No.** |
| **Full fine-tuning** | Take an existing model and keep training *all* of its numbers on your own examples. | The whole model, plus training state, must fit in GPU memory: many times the model's size. | Only for very small models (≤ 0.5B). |
| **LoRA / QLoRA fine-tuning** | Freeze the existing model and train a small add-on (an **adapter**, often 0.1–1% of the size) that nudges its behavior. **QLoRA** also loads the frozen model compressed to 4 bits. Both are explained below the table. | A few GB of GPU memory, minutes to hours. | **Yes**, for models up to about 3–4B parameters on 8 GB. |

**Almost everyone "training their own model" does the third.** You start from an open, already
instruction-tuned model such as Qwen 2.5 Instruct, and teach it a **style, format or specialty**
with a few hundred to a few thousand examples.

### What LoRA and QLoRA are

A model's "knowledge" is stored in huge grids of numbers called **weight matrices**. A 1.5B model has
1.5 billion of these numbers. Normal fine-tuning adjusts all of them, which needs a lot of GPU
memory.

- **LoRA (Low-Rank Adaptation)** leaves the original weights **frozen**, and trains a small
  **adapter** alongside some of the matrices instead. Each adapter is two thin matrices whose
  product is a small correction added to the original matrix's output. Because they're thin (their
  width is the **rank**, `r`, e.g. 16), the adapter has only a tiny fraction of the model's numbers,
  often well under 1%. That's what makes training fast and light. The result is a small adapter
  file (megabytes) that's applied on top of the unchanged base model.
- **QLoRA (Quantized LoRA)** is LoRA plus one more trick: the frozen base model is loaded
  **quantized**, i.e. compressed to **4-bit** numbers instead of 16-bit, which cuts its memory use
  to about a quarter. The adapter itself is still trained at normal precision. Quality stays close
  to ordinary LoRA, and models of a few billion parameters fit in a consumer GPU's memory, like the
  8 GB RTX 3050 here.

In short: **LoRA = train a small add-on instead of the whole model. QLoRA = the same, with the big
frozen model compressed to save memory.**

**Fine-tuning changes behavior, including in ways you didn't intend.** A few hundred examples are
enough to shift how a model answers. If those examples are one-sided, the model becomes one-sided,
and anyone can publish such a model. See the bias example in
[05_readme-models.md](05_readme-models.md) ("Where the models come from").

**What fine-tuning is good and bad at:**
- **Good at:** behavior and style, such as always answering as a friendly vocabulary tutor for
  kids, always using a fixed format, or speaking in a particular voice. These are things you'd
  otherwise repeat in every prompt.
- **Bad at:** reliably teaching new **facts** that change. For facts and documents, **RAG**
  (retrieval, what the Envie app in this project does) is usually better: the model looks the
  facts up at question time. Skills and instructions files are also cheaper ways to change
  behavior, so try those first.

---

## 2. Pre-training from scratch, the simple version: what a GPT is

You won't pre-train a model on a laptop, but knowing how it's done explains everything else in
this project: why models have a context window, why they "make things up", and what fine-tuning
changes.

### The one task: guess the next word

**GPT** stands for **Generative Pre-trained Transformer**:
- **Generative:** it produces text.
- **Pre-trained:** it first learns from a huge amount of general text, before any special tuning.
- **Transformer:** the kind of neural network it uses (explained below).

Pre-training teaches it **one task**: given some text, predict the next **token**. A token is a word
or a piece of a word; "unbelievable" might be split into `un`, `believ`, `able`.

```
Text:        The  cat  sat  on  the  ___
Model:       "mat" 41% · "floor" 18% · "sofa" 9% · "roof" 3% · …
Correct:     "mat"
```

### Training is guessing, checking and adjusting, billions of times

1. **Start with random numbers.** A new model's billions of weights are random, so its guesses are
   nonsense.
2. **Take a piece of real text** (a book, a web page, some code), hide the next token, and let the
   model guess.
3. **Measure how wrong it was.** This number is the **loss**: low if it gave the right token a high
   probability, high if not.
4. **Nudge every weight slightly** in the direction that would have made the right answer a little
   more likely. This is **gradient descent**, and **backpropagation** is how the nudge for each
   weight is calculated.
5. **Repeat** for every position in trillions of tokens.

Nobody teaches it grammar, facts or reasoning directly. Those emerge because they help predict the
next token. To predict "Paris" after "The capital of France is", it helps to have absorbed that fact
somewhere in its weights.

This is why pre-training is so expensive. TinyLlama, a *small* model, saw **3 trillion tokens**,
using 16 data-center GPUs for about **3 months**. Large models use thousands of GPUs.

It also explains a key weakness. The model learns what text is **likely**, not what's **true**. A
confident, plausible but wrong answer ("hallucination") is the model doing exactly what it was
trained to do.

### Inside the Transformer

```
"The cat sat on the"
      │
  1. Tokens → numbers     each token becomes a vector (an embedding), plus its position
      │
  2. Transformer layers   × N  (TinyLlama: 22 layers)
      │   ├─ Attention:      each token gathers information from the earlier tokens
      │   └─ Feed-forward:   each token's vector is processed further on its own
      │
  3. Output               a probability for every token in the vocabulary (~32,000 for TinyLlama)
      │
   "mat" (41%)  →  added to the text, and the whole thing runs again for the next token
```

That last arrow is how a whole answer is written: **one token at a time**, each one fed back in.
It's why answers in this project stream in word by word.

You can see these numbers for yourself in `models/TinyLlama/config.json`: `num_hidden_layers: 22`,
`num_attention_heads: 32`, `vocab_size: 32000`, and `max_position_embeddings: 2048` (its context
window).

### Attention: how each word looks at the others

Attention is the key idea, from the 2017 paper *Attention Is All You Need*. Take:

> The **cat** sat on the mat because **it** was tired.

To understand **"it"**, the model needs to know that it means the cat, not the mat. Attention lets
the vector for "it" **look back** at every earlier token and decide how much each one matters:

```
                 The   cat   sat   on   the   mat  because   it
attention from
"it" →           0.02  0.61  0.05  0.01 0.02  0.21  0.03    0.05
                       ▲ mostly "cat"         ▲ some "mat"
```

It works like a lookup, in three parts, all learned during training:
- each token makes a **query**: "what am I looking for?" ("it" looks for the thing it refers to);
- each token offers a **key**: "what do I contain?" ("cat" offers "an animal that can be tired");
- and a **value**: the information it passes on if chosen.

The query is compared with every key. Good matches get high scores, the scores become weights that
add up to 1 (as above), and "it" receives a weighted mix of the values: mostly "cat". After this
step, the vector for "it" carries the meaning "the tired cat".

Two details matter:
- **Many heads.** Each layer runs several attentions side by side ("heads"). One might track what
  pronouns refer to, another grammar, another nearby words. TinyLlama has 32 heads per layer.
- **No peeking ahead.** In a GPT, a token can only attend to tokens **before** it (a "causal mask"),
  because during training the next token is what it's trying to guess.

**The context window comes from this.** Every token attends to every earlier token, so the work
grows quickly with text length, and a model is trained for a maximum length. That's TinyLlama's
2,048 tokens, and Qwen 3.5's 262,144. Text beyond it can't be attended to at all, which is why
this project cuts attachments and drops old messages.

### From "next-word predictor" to chat assistant

A freshly pre-trained model only continues text. Ask it a question, and it might continue with
more questions. To make an assistant, it's trained further:
1. **Instruction tuning (SFT):** many example conversations, showing how a helpful assistant
   answers. It's the same next-token training, on chat-shaped text.
2. **Preference tuning (RLHF or DPO):** pairs of answers where people (or a stronger model) marked
   one as better, teaching it which kind of answer to prefer. TinyLlama Chat was tuned with DPO on
   the UltraFeedback dataset.

**Fine-tuning with LoRA/QLoRA** (section 1) is a small, cheap version of step 1, applied to an
existing assistant.

### Learn more

| Resource | What it is |
|---|---|
| [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) (3Blue1Brown, Deep Learning Chapter 5) | A visual video introduction to what a GPT does. **Start here.** |
| [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) (3Blue1Brown, Chapter 6) | Queries, keys and values, animated |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | An interactive page running a small GPT (GPT-2) in your browser: type text and watch the attention and the next-token probabilities |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar) | Step-by-step illustrated explanations |
| [How do Transformers work?](https://huggingface.co/learn/llm-course/chapter1/4) (Hugging Face LLM Course) | A short written overview, part of a free course |
| [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Andrej Karpathy) | A 2-hour video writing and training a small GPT from scratch. For when you want to see every line. |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Karpathy's small, readable GPT training code. It can pre-train a tiny model on a laptop GPU in minutes, e.g. on Shakespeare. |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | The original Transformer paper |

**Try it yourself:** nanoGPT's README includes a "baby GPT" example that trains a tiny
character-level model on the works of Shakespeare. That's real pre-training from scratch, small
enough for this computer's GPU, and a fun way to watch the loss fall and the generated text slowly
turn from noise into almost-English.

---

## 3. The worked example: a kids' vocabulary tutor

Goal: fine-tune **Qwen 2.5 1.5B Instruct** so that, given a word, it always answers like the Word
a Day Kids page (see [06_readme-mcp-agents.md](06_readme-mcp-agents.md)): the word, its meaning in
kid-friendly language, a synonym, an example sentence, and a "finish the story" prompt. That's
behavior, which fine-tuning handles well.

> **An example, not a product.** The vocabulary tutor illustrates the fine-tuning process. A model
> actually used with children would need educator-reviewed training data, content-safety testing,
> privacy compliance and ongoing monitoring.

### What you'll need

- **An NVIDIA GPU** with about 8 GB of memory, and the CUDA build of PyTorch (see "Using your GPU"
  in [05_readme-models.md](05_readme-models.md)).
- **A separate Python environment.** Training libraries need newer versions of `transformers` than
  the ones pinned in [requirements.txt](requirements.txt), so don't install them into `venv_rag`:
  ```powershell
  python -m venv venv_train
  .\venv_train\Scripts\Activate.ps1
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install transformers peft trl datasets accelerate bitsandbytes
  ```
  `peft` does the LoRA adapters, `trl` provides a ready-made training loop for chat examples,
  `datasets` loads your data, and `bitsandbytes` does the 4-bit compression for QLoRA.
  These libraries change quickly. If an example below fails, check the current
  [TRL documentation](https://huggingface.co/docs/trl) for renamed options.
- **A Hugging Face account** (free) for step 6.

---

## 4. Step 1 – Prepare training examples

The model learns from **example conversations**: a question and the ideal answer. Save them as
**JSONL**: one JSON object per line, in a file such as `data/train.jsonl`:

```json
{"messages": [{"role": "user", "content": "Teach me the word: audacious"}, {"role": "assistant", "content": "🌟 Today's word is **Audacious**!\n\nIt means being brave enough to take risks, not afraid to try bold things.\n\nSynonym: daring\nExample: The audacious kitten jumped onto the tallest shelf.\n\nFinish the story: I climbed to the top of the slide and gave an audacious shout to..."}]}
{"messages": [{"role": "user", "content": "Teach me the word: curious"}, {"role": "assistant", "content": "🌟 Today's word is **Curious**!\n\nIt means wanting to know or learn about something.\n\nSynonym: inquisitive\nExample: The curious girl peeked inside the old wooden box.\n\nFinish the story: I found a strange key under my bed, and I was so curious that..."}]}
```

**How many?** A few hundred good examples can change style noticeably. Quality matters more than
quantity: every example teaches the model, including mistakes.

**Where examples come from:**
- **Write them yourself,** or with a teacher's help. This gives the best quality.
- **Generate drafts with a bigger model,** then check and correct every one. Small models learn
  errors very easily.
- **Use an existing dataset** from the Hugging Face Hub (search "datasets"). Check its **license**.

**Keep some back.** Put about 10% of the examples in a separate `data/eval.jsonl`, never used for
training, to check the result (step 4).

**Don't include** personal information, secrets, or text you don't have the right to use.
Anything in the training data can come back out of the model.

---

## 5. Step 2 – Fine-tune with QLoRA

A minimal training script, `train_lora.py`:

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # the model to start from (downloaded from Hugging Face)

# Load the base model compressed to 4 bits (QLoRA), so it fits in 8 GB of GPU memory
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE)

data = load_dataset("json", data_files={"train": "data/train.jsonl", "eval": "data/eval.jsonl"})

# LoRA: train small adapter matrices next to the attention layers; the base model stays frozen
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")

config = SFTConfig(
    output_dir="out/vocab-tutor-lora",
    num_train_epochs=3,               # passes over the training data
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # acts like a batch of 16 without needing the memory
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
)

trainer = SFTTrainer(model=model, args=config, train_dataset=data["train"],
                     eval_dataset=data["eval"], peft_config=lora, processing_class=tokenizer)
trainer.train()
trainer.save_model("out/vocab-tutor-lora")   # saves the adapter only (a few MB)
```

Run it with `python train_lora.py`.

**What the settings mean:**
- **`r` (rank)**: how big the adapter is. Larger can learn more, but needs more memory and data.
  8–32 is typical.
- **Epochs**: how many times it sees each example. Too many, and it **overfits**: it memorizes
  the examples instead of learning the pattern.
- **Learning rate**: how big each adjustment is. Too high and training becomes unstable.
- **`loss`** (printed every few steps): how wrong the model's answers are on the training
  examples. It should go down. **`eval_loss`** is the same on the held-back examples. If
  `eval_loss` starts rising while `loss` keeps falling, it's overfitting: use fewer epochs.

**Time and memory:** a few hundred short examples with a 1.5B model should take minutes to tens
of minutes on an RTX 3050. Watch **Task Manager → GPU → Dedicated GPU memory**. If you run out
of memory, lower `per_device_train_batch_size` to 1, or use `Qwen/Qwen2.5-0.5B-Instruct`.

---

## 6. Step 3 – Test it

Before sharing anything, compare the fine-tuned model with the original on **new** words that
weren't in the training data:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "out/vocab-tutor-lora")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

prompt = tok.apply_chat_template([{"role": "user", "content": "Teach me the word: generous"}],
                                 add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(prompt, max_new_tokens=200)[0][prompt.shape[1]:], skip_special_tokens=True))
```

Check that it follows the format, stays kid-friendly, and is still **correct**. Fine-tuning on
style can make a model confidently wrong. Also check it still handles ordinary questions sensibly.

---

## 7. Step 4 – Merge the adapter (optional)

The adapter only works together with its base model. To get one standalone model, which is
simpler to share and to load into Ollama, merge them:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto")
merged = PeftModel.from_pretrained(base, "out/vocab-tutor-lora").merge_and_unload()
merged.save_pretrained("out/vocab-tutor-merged")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").save_pretrained("out/vocab-tutor-merged")
```

The merged folder has the same files as the models in this project's `models/` folder:
`config.json`, `model.safetensors`, the tokenizer files, and so on.

| Share the… | Size (1.5B model) | Users need |
|---|---|---|
| **Adapter** (`out/vocab-tutor-lora`) | A few MB to tens of MB | The base model too, plus `peft` |
| **Merged model** (`out/vocab-tutor-merged`) | About 3 GB | Nothing else |

---

## 8. Step 5 – Upload to Hugging Face

**1. Create an access token.** On huggingface.co, go to **Settings → Access Tokens → Create new
token**, with **Write** permission. Treat it like a password: never put it in code or commit it.

**2. Log in from the terminal,** with any environment that has `huggingface_hub` (including
`venv_rag`), and paste the token when asked:
```powershell
hf auth login
```

**3. Create a repository and upload.** Model repositories are named `your-username/model-name`:
```powershell
hf repo create vocab-tutor-qwen2.5-1.5b --private
hf upload your-username/vocab-tutor-qwen2.5-1.5b out/vocab-tutor-merged
```
`--private` keeps it visible only to you until you're ready. You can make it public later in the
repository's **Settings**. Or upload from Python with
`merged.push_to_hub("your-username/vocab-tutor-qwen2.5-1.5b")`.

**4. Add a model card.** The repository's `README.md` is its **model card**, the page people see.
Include what it's for, how it was trained, and its limits. The header at the top drives the
Hub's filters:
```markdown
---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags: [education, vocabulary, kids, lora]
language: [en]
---

# Vocabulary Tutor (Qwen 2.5 1.5B, fine-tuned)

Answers "Teach me the word: …" in a kid-friendly format: meaning, synonym, example sentence and a
"finish the story" prompt. Fine-tuned with QLoRA on about N hand-checked examples.

**Limitations:** may give wrong definitions for rare words; check before use with children.
```

**Licenses matter.** Your model inherits the **base model's license**. Qwen 2.5 1.5B uses Apache
2.0, but some other models and sizes have their own terms. Also respect your training data's
license. Check both before making anything public.

---

## 9. Step 6 – Use your model

**With this project's own server.** Download it into `models/` the same way as TinyLlama
([01_readme-setup.md](01_readme-setup.md)), and point `LLM_DIR` in
[src/inference_server.py](src/inference_server.py) at it (as in option C of
[05_readme-models.md](05_readme-models.md)):
```powershell
hf download your-username/vocab-tutor-qwen2.5-1.5b --local-dir models/vocab-tutor
```
This project pins an older `transformers`; Qwen 2.5 models load with it.

**With Ollama** (and so in Copilot Chat, as in [05_readme-models.md](05_readme-models.md)). Ollama
can import a merged model folder. Create a file called `Modelfile` containing:
```
FROM ./out/vocab-tutor-merged
```
then run:
```powershell
ollama create vocab-tutor -f Modelfile
ollama run vocab-tutor "Teach me the word: kind"
```
Ollama converts it to its own format during `ollama create`. Direct import supports common model
families, including Qwen 2 and Llama. Ollama's documentation on *importing models* covers the
details and other options, such as compressing (quantizing) it to make it smaller. Once created,
`vocab-tutor` appears in `ollama list`, and can be added to Copilot's model picker like
`qwen3.5:4b`.

---

## 10. Where to learn more

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course): free course covering models,
  fine-tuning and sharing.
- [TRL documentation](https://huggingface.co/docs/trl): the training library used above
  (`SFTTrainer`).
- [PEFT documentation](https://huggingface.co/docs/peft): LoRA and other adapter methods.
- [Hub documentation: uploading models](https://huggingface.co/docs/hub/models-uploading) and
  [model cards](https://huggingface.co/docs/hub/model-cards).
- [Ollama: importing a model](https://github.com/ollama/ollama/blob/main/docs/import.md).

**Checklist before publishing:**
- [ ] Tested on examples that weren't in the training data
- [ ] Still correct, not just in the right style
- [ ] No personal or secret data in the training set
- [ ] Base model's license and data licenses allow sharing
- [ ] Model card explains purpose, training and limitations
- [ ] Access token kept out of code and git

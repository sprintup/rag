# Training Your Own Model and Sharing It on Hugging Face

This guide explains how you'd make your own version of a language model, and publish it on
[Hugging Face](https://huggingface.co) so others (and Ollama, VS Code, or this project) can use it.
It's an overview with a worked example sized for this computer (an NVIDIA RTX 3050 with 8 GB of
video memory). Nothing in this project depends on it.

---

## 1. Three ways to "train your own model"

| Approach | What it means | What it takes | Realistic here? |
|---|---|---|---|
| **Pre-training from scratch** | Start from random numbers and teach a model language itself, from huge amounts of text. | Enormous. TinyLlama 1.1B was trained on about **3 trillion tokens**, using 16 data-center GPUs for about **3 months**. | **No.** |
| **Full fine-tuning** | Take an existing model and keep training *all* of its numbers on your own examples. | The whole model, plus training state, must fit in GPU memory: many times the model's size. | Only for very small models (≤ 0.5B). |
| **LoRA / QLoRA fine-tuning** | Freeze the existing model and train a small add-on (an **adapter**, often 0.1–1% of the size) that nudges its behavior. **QLoRA** also loads the frozen model compressed to 4 bits. | A few GB of GPU memory, minutes to hours. | **Yes**, for models up to about 3–4B parameters on 8 GB. |

**Almost everyone "training their own model" does the third.** You start from an open, already
instruction-tuned model such as Qwen 2.5 Instruct, and teach it a **style, format or specialty**
with a few hundred to a few thousand examples.

**What fine-tuning is good and bad at:**
- **Good at:** behavior and style, such as always answering as a friendly vocabulary tutor for
  kids, always using a fixed format, or speaking in a particular voice. These are things you'd
  otherwise repeat in every prompt.
- **Bad at:** reliably teaching new **facts** that change. For facts and documents, **RAG**
  (retrieval, what the Envie app in this project does) is usually better: the model looks the
  facts up at question time. Skills and instructions files are also cheaper ways to change
  behavior, so try those first.

---

## 2. The worked example: a kids' vocabulary tutor

Goal: fine-tune **Qwen 2.5 1.5B Instruct** so that, given a word, it always answers like the Word
a Day Kids page (see [06_readme-mcp-agents.md](06_readme-mcp-agents.md)): the word, its meaning in
kid-friendly language, a synonym, an example sentence, and a "finish the story" prompt. That's
behavior, which fine-tuning handles well.

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

## 3. Step 1 – Prepare training examples

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

## 4. Step 2 – Fine-tune with QLoRA

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

## 5. Step 3 – Test it

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

## 6. Step 4 – Merge the adapter (optional)

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

## 7. Step 5 – Upload to Hugging Face

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

## 8. Step 6 – Use your model

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

## 9. Where to learn more

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

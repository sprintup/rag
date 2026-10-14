import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_tiny_llama():
    print(f"Using device: {get_device_info()}")
    print("Loading model and tokenizer...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Set pad token id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("Model and tokenizer loaded successfully.")

    prompt = "Explain the concept of machine learning in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    print("Generating response...")
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nPrompt:", prompt)
    print("\nResponse:", response)

def get_device_info():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    elif torch.backends.mps.is_available():
        return "Apple Silicon GPU"
    else:
        return "CPU"

if __name__ == "__main__":
    test_tiny_llama()
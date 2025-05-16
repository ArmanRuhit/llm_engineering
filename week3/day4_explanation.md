# Day 4: Working with Large Language Models

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Dependencies](#setup-and-dependencies)
3. [Model Loading and Quantization](#model-loading-and-quantization)
4. [Text Generation](#text-generation)
5. [Advanced Usage](#advanced-usage)
6. [Best Practices](#best-practices)

## Introduction
This notebook demonstrates how to work with large language models (LLMs) using the Hugging Face Transformers library. It covers model loading, quantization, text generation, and memory management techniques.

## Setup and Dependencies

### Key Libraries
- `transformers`: For loading and using pre-trained models
- `torch`: PyTorch for tensor operations
- `huggingface_hub`: For model and tokenizer management
- `bitsandbytes`: For model quantization
- `dotenv`: For environment variable management

### Authentication
```python
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)
```

## Model Loading and Quantization

### Available Models
```python
# Instruct-tuned models
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct"
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
```

### Quantization Configuration
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # 4-bit quantization
    bnb_4bit_use_double_quant=True,      # Nested quantization for better accuracy
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation dtype
    bnb_4bit_quant_type="nf4"            # Normal Float 4 quantization
)
```

### Model Initialization
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)
```

## Text Generation

### Basic Generation
```python
outputs = model.generate(
    inputs,
    max_new_tokens=80,
    pad_token_id=tokenizer.eos_token_id
)
```

### Advanced Generation with Sampling
```python
outputs = model.generate(
    input_ids=inputs,
    attention_mask=attention_mask,
    max_new_tokens=80,
    do_sample=True,          # Enable sampling
    temperature=0.7,         # Control randomness (0.0-1.0)
    top_p=0.9,              # Nucleus sampling
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

## Memory Management

### Memory Usage
```python
memory = model.get_memory_footprint() / 1e6  # In MB
print(f"Memory footprint: {memory:,.1f} MB")
```

### Cleaning Up
```python
del model, tokenizer, inputs, outputs
torch.cuda.empty_cache()
gc.collect()
```

## Advanced Usage

### Chat Template
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a joke about AI"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")
```

### Streaming Output
```python
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,  # Skip the input prompt in the output
    skip_special_tokens=True
)

outputs = model.generate(
    inputs,
    streamer=streamer,
    max_new_tokens=100
)
```

## Best Practices

1. **Memory Efficiency**:
   - Use 4-bit or 8-bit quantization for large models
   - Clear CUDA cache after generation
   - Use `gc.collect()` for garbage collection

2. **Generation Quality**:
   - Adjust `temperature` (0.7 is a good starting point)
   - Use `top_p` (nucleus sampling) for focused generation
   - Set appropriate `max_new_tokens` to prevent very long outputs

3. **Error Handling**:
   - Always check CUDA availability
   - Handle cases where the model might generate unsafe content
   - Implement timeouts for generation

4. **Performance**:
   - Batch inputs when possible
   - Use `torch.compile()` for faster inference (PyTorch 2.0+)
   - Consider using vLLM or TGI for production deployment

## Example Workflow

1. Load a quantized model
2. Prepare chat messages
3. Generate text with streaming
4. Clean up resources

```python
# 1. Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    quantization_config=quant_config
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# 2. Prepare messages
messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms"}
]

# 3. Generate with streaming
streamer = TextStreamer(tokenizer, skip_prompt=True)
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    inputs,
    streamer=streamer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7
)

# 4. Clean up
del model, tokenizer, inputs, outputs
torch.cuda.empty_cache()
```

This notebook provides a solid foundation for working with large language models in a resource-efficient manner.

# Day 1: Text-to-Speech and Image Generation with Transformers

## Table of Contents
1. [Setup and Dependencies](#setup-and-dependencies)
2. [Text-to-Speech with Microsoft's SpeechT5](#text-to-speech-with-speecht5)
3. [Image Generation with Stable Diffusion](#image-generation-with-stable-diffusion)

## Setup and Dependencies

### Key Libraries
- `diffusers`: For working with diffusion models like Stable Diffusion
- `transformers`: Provides pre-trained models for NLP and speech tasks
- `datasets`: For loading and managing datasets
- `soundfile`: For audio file I/O operations
- `torch`: PyTorch for tensor operations and model training
- `IPython.display`: For displaying audio in Jupyter notebooks

### Environment Setup
```python
import os
import torch
from diffusers import StableDiffusionPipeline
os.environ["XFORMERS_DISABLE_TRITON"] = "1"  # Disables Triton for compatibility
```

## Text-to-Speech with Microsoft's SpeechT5

### Model Initialization
```python
from transformers import pipeline

TTS_MODEL = "microsoft/speecht5_tts"
synthesiser = pipeline(
    task="text-to-speech",
    model=TTS_MODEL,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

#### Parameters:
- `task="text-to-speech"`: Specifies the pipeline task
- `model=TTS_MODEL`: Uses Microsoft's SpeechT5 model
- `device`: Automatically uses GPU if available for faster processing

### Voice Embeddings
```python
from datasets import load_dataset
EMBEDDING_DATASET = "Matthijs/cmu-arctic-xvectors"
embeddings_dataset = load_dataset(EMBEDDING_DATASET)
speaker_embedding = torch.tensor(embeddings_dataset['validation'][7306]["xvector"]).unsqueeze(0)
```

#### Key Points:
- Loads pre-computed speaker embeddings from the CMU Arctic dataset
- `unsqueeze(0)` adds a batch dimension required by the model
- Different speaker embeddings produce different voice characteristics

### Generating Speech
```python
speech = synthesiser(
    text_inputs="Your text here",
    forward_params={
        "speaker_embeddings": speaker_embedding
    }
)
```

#### Parameters:
- `text_inputs`: The text to be converted to speech
- `forward_params`: Additional parameters passed to the model's forward method
  - `speaker_embeddings`: Controls voice characteristics

### Saving Audio
```python
import soundfile as sf
sf.write(
    file="output.wav",
    data=speech["audio"],
    samplerate=speech["sampling_rate"]
)
```

## Image Generation with Stable Diffusion

### Model Initialization
```python
model_id = "OFA-Sys/small-stable-diffusion-v0"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_model_cpu_offload()
```

#### Parameters:
- `model_id`: Identifier for the pre-trained model
- `torch_dtype=torch.float16`: Uses 16-bit floating point for memory efficiency
- `enable_model_cpu_offload()`: Optimizes memory usage by offloading to CPU when possible

### Generating Images
```python
prompt = "A delicious burger"
image = pipe(prompt).images[0]
```

#### Key Points:
- The `prompt` guides the image generation
- Returns a list of images (in this case, we take the first one with `[0]`)
- The output is a PIL Image object that can be displayed or saved

### Displaying the Image
```python
from IPython.display import display
display(image)
```

## Best Practices
1. **Memory Management**:
   - Use `enable_model_cpu_offload()` for memory-constrained environments
   - Consider using `torch.float16` for faster inference with minimal quality loss

2. **Performance**:
   - GPU acceleration significantly speeds up both TTS and image generation
   - Batch processing can improve throughput for multiple generations

3. **Quality**:
   - For TTS, experiment with different speaker embeddings for desired voice characteristics
   - For image generation, crafting detailed prompts improves output quality

4. **Error Handling**:
   - Always check for CUDA availability when using GPU acceleration
   - Handle cases where models might return empty or unexpected outputs

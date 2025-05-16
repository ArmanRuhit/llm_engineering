# Meeting Minutes Generator - Technical Documentation

## Overview
This project demonstrates an automated pipeline for generating structured meeting minutes from audio recordings using state-of-the-art AI models. The system combines automatic speech recognition (ASR) with large language models to transform raw meeting audio into well-organized, actionable meeting minutes.

## Architecture

### 1. Audio Processing Layer
- **Input**: Audio file (MP3 format)
- **Technology**: Whisper ASR model
- **Function**: Converts speech to text with high accuracy
- **Implementation Options**:
  - `faster_whisper` (primary implementation)
  - Hugging Face's Transformers with OpenAI's Whisper (alternative)

### 2. Text Processing Layer
- **Model**: Meta's Llama 3.1 8B Instruct
- **Purpose**: Analyzes transcript and generates structured output
- **Features**:
  - 4-bit quantization for efficient GPU memory usage
  - Context-aware text generation
  - Markdown formatting

## Key Components

### 1. Dependencies
```python
import os
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
```

### 2. Configuration
```python
AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
audio_filename = "./denver_extract.mp3"
```

### 3. Core Functions

#### Audio Transcription
```python
def transcribe_audio(path, model_size="small"):
    model = WhisperModel(
        model_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16"
    )
    segments, info = model.transcribe(path)
    return " ".join(segment.text for segment in segments)
```

#### Meeting Minutes Generation
- **System Prompt**: Defines the output format and structure
- **User Prompt**: Combines instructions with the transcript
- **Model Configuration**: Sets up quantization and generation parameters

## Usage Instructions

1. **Prerequisites**
   - Python 3.8+
   - CUDA-capable GPU (recommended)
   - Hugging Face authentication token
   - Required Python packages (see `requirements.txt`)

2. **Setup**
   ```bash
   pip install -r requirements.txt
   export HF_TOKEN='your_huggingface_token_here'
   ```

3. **Running the Notebook**
   - Open `day5.ipynb` in Jupyter Lab/Notebook
   - Run all cells sequentially
   - The final output will be displayed as formatted markdown

## Output Format
The generated meeting minutes include:

```markdown
# Meeting Minutes

## Summary
- **Date**: [Auto-detected or specified]
- **Location**: [Auto-detected or specified]
- **Attendees**: [List of detected participants]

## Key Discussion Points
- [Bullet point 1]
- [Bullet point 2]

## Action Items
- [ ] Task 1 (Owner: [Name])
- [ ] Task 2 (Owner: [Name])
```

## Performance Considerations

1. **Hardware Requirements**
   - Minimum: 16GB RAM, CPU
   - Recommended: NVIDIA GPU with ≥8GB VRAM

2. **Optimizations**
   - 4-bit quantization reduces VRAM usage
   - Batch processing for multiple files
   - Caching of intermediate results

## Limitations

1. **Audio Quality**: Performance degrades with poor audio quality
2. **Speaker Diarization**: Currently doesn't identify different speakers
3. **Context Length**: Limited by the model's maximum sequence length

## Future Improvements

1. Add speaker diarization
2. Support for video files
3. Integration with calendar apps
4. Real-time processing capabilities
5. Custom templates for different meeting types

## License
[Specify License]

## Author
[Your Name]
```

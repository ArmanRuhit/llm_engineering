# LLM Engineering Project

This repository contains a structured learning path for LLM (Large Language Model) engineering, organized into weekly modules that cover various aspects of working with AI models, from API integration to model deployment.

## Project Overview

This project provides hands-on experience with:
- OpenAI API integration
- Local LLM deployment with Ollama
- Hugging Face models and transformers
- Text generation, summarization, and classification
- Image generation with stable diffusion models
- Text-to-speech and speech-to-text conversion
- Building interactive AI applications with Gradio

## Project Structure

```
llm_engineering/
├── main.py                # Main entry point
├── week1/                 # Introduction to OpenAI APIs
│   ├── day1.ipynb         # OpenAI API basics and web scraping
│   ├── day2.ipynb         # Content summarization
│   └── day5.ipynb         # Advanced techniques
├── week2/                 # Working with local LLMs
│   ├── day1.ipynb         # Ollama setup
│   ├── day2.ipynb         # Building simple chatbots
│   ├── day5.ipynb         # Function calling and text-to-image generation
├── week3/                 # Advanced models & multimodality
│   ├── day1.ipynb         # Hugging Face transformers
│   ├── day2.ipynb         # Multimodal applications
│   └── speech.wav         # Sample audio file
└── .env                   # Environment variables for API keys
```

## Functionality Summary

### Week 1: Getting Started with LLMs
- Website content scraping and analysis
- OpenAI API integration for text summarization
- Prompt engineering techniques
- Structured data extraction

### Week 2: Local Models & Interactive Applications
- Local LLM setup with Ollama
- Building an airline assistant chatbot
- Function calling with LLMs for dynamic responses
- Text-to-image generation with Stable Diffusion
- Creating interactive interfaces with Gradio

### Week 3: Advanced Multimodal Applications
- Hugging Face pipeline integrations (sentiment analysis, NER, QA)
- Text-to-image generation with multiple models
- Speech synthesis and audio processing
- Building multimodal applications

## Requirements

This project requires Python 3.13 or higher with the dependencies listed in the pyproject.toml file.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

4. Install dependencies:
   ```
   pip install -e .
   ```

5. Set up environment variables:
   - Create a `.env` file based on the provided `.env.example`
   - Add your API keys for OpenAI and Hugging Face

## Running the Project

- Individual notebooks can be run in Jupyter:
  ```
  jupyter notebook
  ```

## License

This project is provided for educational purposes.
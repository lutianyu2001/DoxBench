# Experiment Scripts

This directory contains two core experimental scripts for analyzing and testing large language models on location-related tasks.

## 📋 Overview

- **`clueminer.py`**: LLM-powered category building and analysis tool for clue datasets
- **`experiment.py`**: Comprehensive geolocation testing framework with multi-model support

---


## 🔍 clueminer.py

### Description
A sophisticated LLM analyzer focused on category building with enhanced console output and token tracking. This tool processes clue datasets and builds category frameworks through iterative analysis.

### Key Features
- **Token Usage Tracking**: Real-time monitoring of input/output tokens across all API calls
- **Enhanced Console Display**: Rich formatting with progress indicators and statistics tables
- **Resume Functionality**: Can resume analysis from breakpoints for long-running tasks
- **Multi-format Dataset Support**: Handles CSV and JSON datasets with flexible parsing
- **Iterative Category Building**: Uses LLM to progressively build and refine category frameworks

### Core Components
- `Phase1CategoryBuilder`: Main class for category analysis
- `TokenTracker`: Monitors and reports token usage statistics
- `ConsoleDisplay`: Provides rich console output with tables and progress indicators
- `LLMClient`: Handles OpenAI API interactions with error handling

### Key Arguments
- `--dataset`: Input dataset file path (CSV or JSON format)
- `--model_name_or_path`: LLM model to use (default: "o4-mini-2025-04-16")
- `--max_iterations`: Maximum number of analysis iterations (default: 10)
- `--output_dir`: Directory for output files (default: "output")
- `--breakpoint_file`: Resume from a specific checkpoint file

### Usage Example
```bash
python clueminer.py \
    --dataset results/top1-general-gemini_o3_4o_4_1.csv \
    --model_name_or_path o4-mini-2025-04-16 \
    --max_iterations 10 \
    --output_dir ./output
```

---

## 🌍 experiment.py

### Description
A comprehensive geolocation testing framework that evaluates multiple LLM models on image-based location identification tasks. Supports various reasoning modes and defensive mechanisms.

### Key Features
- **Multi-Model Support**: 14+ models including GPT-4o, Gemini, Claude, LLaMA, and QVQ
- **Chain-of-Thought (CoT) Reasoning**: Multiple CoT modes including workflow-based reasoning
- **GeoMiner Detector**: Specialized pipeline for location clue extraction
- **Defense Mechanisms**: Prompt-based defense against privacy violations
- **Image Augmentation**: Gaussian noise injection for robustness testing
- **Parallel Processing**: Multi-threaded execution with batch processing
- **Resume Capability**: Can resume from breakpoints for large-scale experiments

### Supported Models
- **OpenAI**: o3, o4-mini, gpt-4o, gpt-4.1, gpt-4.1-mini
- **Google**: gemini-2.5-pro-preview
- **Anthropic**: claude-sonnet-4, claude-opus-4
- **Meta**: llama-4-maverick, llama-4-scout
- **Qwen**: qwen2.5-vl-72b, qvq-72b-preview, qvq-max
- **And more...**

### Core Components
- `GeoLocationTester`: Main testing framework class
- Model-specific API handlers for each supported provider
- Geographic utilities for distance calculation and validation
- Image processing pipeline with augmentation support

### Key Arguments
- `--model`: Model to test (e.g., 'gpt4o', 'gemini', 'sonnet4')
- `--cot_mode`: Chain-of-thought mode ('off', 'on', 'workflow')
- `--geominer_detector_model`: Model for GeoMiner detector pipeline
- `--reasoning_summary`: Reasoning capture mode ('off', 'plain', 'with_llm_judge')
- `--prompt_base_defense`: Enable privacy protection ('on', 'off')
- `--noise_std`: Gaussian noise standard deviation for image augmentation
- `--top_n_addresses`: Number of top addresses to consider
- `--max_workers`: Number of parallel processing threads
- `--random_sample`: Sample size for random testing
- `--breakpoint`: Resume from specific checkpoint

### Usage Examples

#### Basic Evaluation
```bash
python experiment.py \
    --model gpt4o \
    --input_csv dataset/test_images.csv \
    --output_csv results/gpt4o_results.csv \
    --cot_mode on
```

#### Advanced Workflow with Defense
```bash
python experiment.py \
    --model gemini \
    --cot_mode workflow \
    --geominer_detector_model gpt4.1-mini \
    --reasoning_summary with_llm_judge-gpt4o \
    --prompt_base_defense on \
    --noise_std 0.1 \
    --max_workers 4
```

#### Resume from Breakpoint
```bash
python experiment.py \
    --model sonnet4 \
    --input_csv dataset/large_test.csv \
    --output_csv results/sonnet4_results.csv \
    --breakpoint 150
```

---

## 🛠️ Environment Setup

### Required Dependencies
```bash
# Core dependencies
pip install openai anthropic dashscope googlemaps pandas rich tiktoken

# Image processing (optional, for Gaussian noise)
conda install albumentations==2.0.8 opencv-python

# Geographic calculations
pip install pyproj
```

### Environment Variables
Create a `.env` file with the following:
```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key

# Dashscope API (for Qwen models)
DASHSCOPE_API_KEY=your_dashscope_api_key

# OpenRouter API (for Gemini and other models)
OPENROUTER_API_KEY=your_openrouter_api_key

# Google Maps API (for geocoding)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

---

## 📊 Output Formats

### clueminer.py Output
- **Categories JSON**: Final category framework with descriptions
- **Individual Outputs**: Per-iteration analysis results
- **Token Statistics**: Comprehensive usage reports
- **Resume Files**: Checkpoint files for interrupted runs

### experiment.py Output  
- **Results CSV**: Detailed results with predictions, ground truth, and metrics
- **Reasoning Logs**: Chain-of-thought reasoning traces (if enabled)
- **Token Usage**: API call statistics and costs
- **Progress Reports**: Real-time processing status

---

## 🔧 Advanced Configuration

Both scripts support extensive configuration through command-line arguments and configuration files. See the source code for complete parameter documentation and advanced usage patterns.

For training and evaluation examples similar to the provided pattern, see the main project documentation. 

<p align="center">
    <img src="./logo/dox_color.svg" width="150"/>
<p>

<h2 align="center"> <a>⛓‍💥 Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

<h5 align="center">

[Weidi Luo*](https://eddyluo1232.github.io/), [Tianyu Lu*](https://scholar.google.com/citations?user=kkiCj18AAAAJ&hl=en), [Qiming Zhang*](https://scholar.google.com/citations?user=hgu_aPwAAAAJ&hl=en), [Xiaogeng Liu](https://xiaogeng-liu.com/), [Bin Hu](https://bin-hu.com/)

[Yue Zhao](https://viterbi-web.usc.edu/~yzhao010/), [Jieyu Zhao](https://jyzhao.net/), [Song Gao](https://geography.wisc.edu/staff/gao-song/), [Patrick McDaniel](https://patrickmcdaniel.org/), [Zhen Xiang](https://zhenxianglance.github.io/), [Chaowei Xiao](https://xiaocw11.github.io/)

<p align="center">
  <a href="https://arxiv.org/abs/2504.19373">
  <img src="https://img.shields.io/badge/ArXiv-2504.19373-b31b1b.svg?style=flat-square&logo=arxiv" alt="arXiv">
</a>
  <a href="https://huggingface.co/datasets/tianyulu/DoxBench">
  <img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg?style=flat-square&logo=huggingface" alt="Hugging Face">
</a>

  <a href="https://github.com/lutianyu2001/DoxBench">
  <img src="https://img.shields.io/github/stars/lutianyu2001/DoxBench?style=flat-square&logo=github" alt="GitHub stars">
</a>

  <img src="https://img.shields.io/badge/Model-Type%3A%20MLRM%20%2F%20MLLM-yellowgreen?style=flat-square">
  <img src="https://img.shields.io/badge/Dataset-DoxBench-orange?style=flat-square">
  <img src="https://img.shields.io/badge/Last%20Updated-June%202025-brightgreen?style=flat-square">
</p>

## 💡 Abstract
Recent advances in multi-modal large reasoning models (MLRMs) have shown significant ability to interpret complex visual content. While these models enable impressive reasoning capabilities, they also introduce novel and underexplored privacy risks. In this paper, we identify a novel category of privacy leakage in MLRMs: Adversaries can infer sensitive geolocation information, such as a user's home address or neighborhood, from user-generated images, including selfies captured in private settings. To formalize and evaluate these risks, we propose a three-level visual privacy risk framework that categorizes image content based on contextual sensitivity and potential for location inference. We further introduce DoxBench, a curated dataset of 500 real-world images reflecting diverse privacy scenarios. Our evaluation across 11 advanced MLRMs and MLLMs demonstrates that these models consistently outperform non-expert humans in geolocation inference and can effectively leak location-related private information. This significantly lowers the barrier for adversaries to obtain users' sensitive geolocation information. We further analyze and identify two primary factors contributing to this vulnerability: (1) MLRMs exhibit strong reasoning capabilities by leveraging visual clues in combination with their internal world knowledge; and (2) MLRMs frequently rely on privacy-related visual clues for inference without any built-in mechanisms to suppress or avoid such usage. To better understand and demonstrate real-world attack feasibility, we propose GeoMiner, a collaborative attack framework that decomposes the prediction process into two stages: clue extraction and reasoning to improve geolocation performance while introducing a novel attack perspective. Our findings highlight the urgent need to reassess inference-time privacy risks in MLRMs to better protect users' sensitive information.
<img src="./misc/framework.png" width="1000"/>

## 👻 Let's invite the Doxxing Team

<p align="center">
  <img src="./misc/doxing_team.png" width="500" alt="team-1">
</p>

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Core Framework](#core-framework)
- [Examples](#examples)
- [Output Formats](#output-formats)
- [Advanced Features](#advanced-features)
- [Performance](#performance)
- [Dataset](#dataset)
- [Citation](#citation)
- [License](#license)

## Features

- **Comprehensive MLRM Evaluation**: Support for 11+ state-of-the-art multi-modal models
- **Three-Level Privacy Framework**: Systematic categorization of visual privacy risks
- **GeoMiner Attack Framework**: Novel collaborative attack methodology for enhanced geolocation inference
- **Real-World Dataset**: 500 curated images reflecting diverse privacy scenarios
- **Distance-Based Accuracy Metrics**: Precise evaluation using geospatial distance calculations
- **Clue Mining Analysis**: Automated extraction and analysis of privacy-revealing visual elements
- **Defense Mechanism Testing**: Built-in evaluation of privacy protection strategies
- **Parallel Processing**: Multi-threaded evaluation for large-scale experiments
- **Comprehensive Output**: Detailed results with reasoning traces and statistical analysis

## Prerequisites

- Python ≥3.8
- Valid API keys for supported model providers
- Google Maps API key (for geocoding and distance calculations)
- Sufficient computational resources for model inference

## Installation

```bash
git clone https://github.com/lutianyu2001/DoxBench.git
cd ./DoxBench/code/experiment
conda env create -f environment.yml
conda activate gps-address
```

### API Keys Configuration

Create a `.env` file in the experiment directory:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

#### Supported Models
- **OpenAI Models**: `o3`, `o4mini`, `gpt4o`, `gpt4.1`
- **Anthropic Models**: `sonnet4`, `opus4`
- **Google Models**: `Gemini-2.5Pro`
- **Meta Models**: `llama4-maverick`, `llama4-scout`
- **Qwen Models**: `qvq-max`

## Usage

### Main Experiment

```bash
python app.py [-h] [-v] --input_csv INPUT_CSV --output_csv OUTPUT_CSV 
              --model MODEL [--max_workers MAX_WORKERS]
              [--cot_mode COT_MODE] [--geominer_detector_model DETECTOR_MODEL]
              [--prompt_base_defense {on,off}] [--noise_std NOISE_STD]
              [--verbose] [--custom_prompt CUSTOM_PROMPT]
```

#### Core Arguments

- `--input_csv` **[Required]**
    - Input dataset file in CSV format
    - Must contain image paths and ground truth location data
- `--output_csv` **[Required]**
    - Output file path for evaluation results
- `--model` **[Required]**
    - Target MLRM for evaluation
    - Options: `gpt4o`, `sonnet4`, `gemini`, `qwen2.5vl`, etc.

#### Processing Arguments

- `--max_workers` (default: 3)
    - Number of parallel workers for API requests
- `--batch_size` (default: 5)
    - Batch size for processing images
- `--cot_mode` (default: "standard")
    - Chain-of-thought reasoning mode
    - Options: `standard`, `geominer`

#### Advanced Arguments

- `--geominer_detector_model`
    - Detector model for GeoMiner framework
    - Enhances clue extraction capabilities
- `--prompt_base_defense` (default: "off")
    - Enable prompt-based privacy defense mechanisms
- `--noise_std` (default: 0.0)
    - Standard deviation for Gaussian noise injection
- `--verbose`
    - Enable detailed logging and progress tracking

#### Examples

##### Basic Evaluation

```bash
# Basic evaluation with GPT-4o
python app.py --input_csv your_dataset.csv --output_csv results.csv --model gpt4o

# Advanced evaluation with multiple models
python app.py --input_csv your_dataset.csv --output_csv results.csv \
    --model gemini --max_workers 3 --batch_size 5

# Using GeoMiner framework
python app.py --input_csv your_dataset.csv --output_csv results.csv \
    --model gpt4o --cot_mode geominer --geominer_detector_model gpt4.1-mini
```

#### Advanced Configuration

##### Defense Mechanisms
```bash
# Enable prompt-based defense
python app.py --input_csv your_dataset.csv --prompt_base_defense on

# Add Gaussian noise to images
python app.py --input_csv your_dataset.csv --noise_std 0.1
```

##### Parallel Processing
```bash
# Process with multiple workers
python app.py --input_csv your_dataset.csv --max_workers 5 --batch_size 10
```

#### Output

The evaluation generates comprehensive results including:

- **Geolocation Predictions**: Predicted addresses and coordinates
- **Distance Accuracy**: Calculated distances between predicted and actual locations
- **Privacy Risk Assessment**: Categorized by our three-level framework
- **Token Usage Statistics**: API consumption tracking
- **Clue Analysis**: Extracted visual and contextual clues

##### Sample Output Structure
```csv
id,image_id,classification,people,selfie,address,geoid,latitude,longitude,country,region,metropolitan,guessed_address,guessed_geoid,guessed_lat,guessed_lon,guessed_country,guessed_region,guessed_metropolitan,country_correct,region_correct,metropolitan_correct,tract_correct,block_correct,error_distance_km,api_call_time,clue_list,address_list,answer,prompt
```


### Clue Mining Analysis

Use the enhanced clue mining tool to analyze privacy leakage patterns.

```bash
python app.py [-h] [-v] --input_csv INPUT_CSV --output_csv OUTPUT_CSV 
              --model MODEL [--max_workers MAX_WORKERS] [--batch_size BATCH_SIZE]
              [--cot_mode COT_MODE] [--geominer_detector_model DETECTOR_MODEL]
              [--prompt_base_defense {on,off}] [--noise_std NOISE_STD]
              [--verbose] [--custom_prompt CUSTOM_PROMPT]
```

#### Core Arguments

- `--input_csv` **[Required]**
    - Input dataset file in CSV format
    - Must contain image paths and ground truth location data
- `--output_csv` **[Required]**
    - Output file path for evaluation results
- `--model` **[Required]**
    - Target MLRM for evaluation
    - Options: `gpt4o`, `sonnet4`, `gemini`, `qwen2.5vl`, etc.

#### Processing Arguments

- `--max_workers` (default: 3)
    - Number of parallel workers for API requests
- `--batch_size` (default: 5)
    - Batch size for processing images
- `--cot_mode` (default: "standard")
    - Chain-of-thought reasoning mode
    - Options: `standard`, `geominer`

#### Advanced Arguments

- `--geominer_detector_model`
    - Detector model for GeoMiner framework
    - Enhances clue extraction capabilities
- `--prompt_base_defense` (default: "off")
    - Enable prompt-based privacy defense mechanisms
- `--noise_std` (default: 0.0)
    - Standard deviation for Gaussian noise injection
- `--verbose`
    - Enable detailed logging and progress tracking

#### Examples
```bash
# Run clue mining analysis
python clueminer.py --input_file results/your_results.csv --max_iterations 10 --model o4-mini-2025-04-16

# Resume from a specific iteration
python clueminer.py --input_file results/your_results.csv --breakpoint_file output/phase1_categories_iteration_5.json
```


## Citation

If you use DoxBench in your research, please cite our paper:

```bibtex
@article{luo2024doxing,
  title={Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models},
  author={Luo, Weidi and Lu, Tianyu and Zhang, Qiming and Liu, Xiaogeng and Hu, Bin and Zhao, Yue and Zhao, Jieyu and Gao, Song and McDaniel, Patrick and Xiang, Zhen and Xiao, Chaowei},
  journal={arXiv preprint arXiv:2504.19373},
  year={2024}
}
```

For more detailed documentation and examples, please refer to our [paper](https://arxiv.org/abs/2504.19373) and [dataset](https://huggingface.co/datasets/tianyulu/DoxBench).

---

**⚠️ Important Notes:**
- This tool is designed for research purposes to understand privacy risks in MLRMs
- Please use responsibly and in accordance with ethical guidelines
- Ensure you have proper permissions for any images you analyze
- Be mindful of API costs when running large-scale evaluations

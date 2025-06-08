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

### 1. Environment Configuration

#### Option A: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/lutianyu2001/DoxBench.git
cd DoxBench

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate gps-address
```

#### Option B: Using pip
```bash
# Clone the repository
git clone https://github.com/lutianyu2001/DoxBench.git
cd DoxBench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # You may need to create this from environment.yml
```

### 2. API Keys Configuration

Create a `.env` file in the project root directory and add your API keys:

```bash
# OpenAI API Keys
OPENAI_API_KEY=your_openai_api_key_here

# OpenRouter API Key (for Gemini, Qwen, Llama models)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Dashscope API Key (for QVQ models)
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# Google Maps API Key (for geocoding)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

### 3. Dataset Preparation

Download the DoxBench dataset from [HuggingFace](https://huggingface.co/datasets/tianyulu/DoxBench):

```bash
# Option 1: Using git-lfs
git lfs clone https://huggingface.co/datasets/tianyulu/DoxBench

# Option 2: Download manually and place in your data directory
# Ensure your CSV file has columns: image_path, real_address, real_lat, real_lon
```

### 4. Running the Evaluation

#### Web Interface (Streamlit)
```bash
# Launch the interactive web interface
streamlit run app.py

# Open your browser and navigate to http://localhost:8501
```

#### Command Line Interface
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

#### Available Models
- **OpenAI Models**: `o3`, `o4mini`, `gpt4o`, `gpt4.1`, `gpt4.1-mini`
- **Anthropic Models**: `sonnet4`, `opus4`
- **Google Models**: `Gemini-2.5Pro`
- **Meta Models**: `llama4-maverick`, `llama4-scout`
- **Alibaba Models**: `qwen2.5vl`, `qvq`, `qvq-max`

### 5. Clue Mining Analysis

Use the enhanced clue mining tool to analyze privacy leakage patterns:

```bash
# Run clue mining analysis
python clueminer.py --input_file results/your_results.csv \
    --max_iterations 10 --model o4-mini-2025-04-16

# Resume from a specific iteration
python clueminer.py --input_file results/your_results.csv \
    --breakpoint_file output/phase1_categories_iteration_5.json
```

### 6. Understanding the Output

The evaluation generates comprehensive results including:

- **Geolocation Predictions**: Predicted addresses and coordinates
- **Distance Accuracy**: Calculated distances between predicted and actual locations
- **Privacy Risk Assessment**: Categorized by our three-level framework
- **Token Usage Statistics**: API consumption tracking
- **Clue Analysis**: Extracted visual and contextual clues

#### Sample Output Structure
```csv
image_path,real_address,real_lat,real_lon,predicted_address,predicted_lat,predicted_lon,distance_km,accuracy_level,clues_extracted,reasoning_trace
```

### 7. Advanced Configuration

#### Defense Mechanisms
```bash
# Enable prompt-based defense
python app.py --input_csv your_dataset.csv --prompt_base_defense on

# Add Gaussian noise to images
python app.py --input_csv your_dataset.csv --noise_std 0.1
```

#### Parallel Processing
```bash
# Process with multiple workers
python app.py --input_csv your_dataset.csv --max_workers 5 --batch_size 10
```

#### Custom Prompts
```bash
# Use custom evaluation prompts
python app.py --input_csv your_dataset.csv --custom_prompt "Your custom prompt here"
```

### 8. Troubleshooting

#### Common Issues:
1. **API Rate Limits**: Reduce `max_workers` and increase `batch_size`
2. **Memory Issues**: Process smaller batches or use lighter models
3. **Geocoding Errors**: Ensure Google Maps API key is valid and has sufficient quota
4. **Model Availability**: Check API key permissions and model access

#### Debug Mode:
```bash
# Enable verbose logging
python app.py --input_csv your_dataset.csv --verbose

# Test API connections
python -c "from app import GeoLocationTester; tester = GeoLocationTester(); tester.test_dashscope_connection()"
```

### 9. Citation

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

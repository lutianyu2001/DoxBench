#!/usr/bin/env python3
import json
import os
import sys
import csv
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import time
import re
from dotenv import load_dotenv
import ast
from datetime import datetime
import pytz

# Enhanced console output imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich import box

# Token counting import
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Token counting will be estimated.")

# Pandas import for CSV parsing
import pandas as pd

# Load environment variables
load_dotenv()


@dataclass
class TokenUsage:
    """Token usage statistics"""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cached_input_tokens
    
    def __add__(self, other):
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens
        )


class TokenTracker:
    """Track token usage"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.total_usage = TokenUsage()
        self.call_history = []
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def add_usage(self, usage: TokenUsage, description: str = ""):
        """Add token usage for a single API call"""
        self.total_usage += usage
        self.call_history.append({
            'timestamp': time.time(),
            'usage': usage,
            'description': description
        })


class ConsoleDisplay:
    """Enhanced console display with rich formatting"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.console = Console()
        self.current_iteration = 0
        self.total_iterations = 0
        self.model_name = model_name
    
    def show_header(self, title: str):
        """Display program header"""
        header = Panel(
            f"[bold blue]{title}[/bold blue]\n"
            f"[dim]Enhanced LLM Analyzer with Token Tracking[/dim]",
            box=box.DOUBLE,
            style="blue"
        )
        self.console.print(header)
        self.console.print()  # Add spacing
    
    def show_iteration_start(self, iteration: int, total_iterations: int, dataset_size: int):
        """Display iteration start information"""
        self.current_iteration = iteration
        self.total_iterations = total_iterations
        
        progress_text = f"Iteration {iteration}/{total_iterations}"
        dataset_text = f"Processing {dataset_size} entries"
        
        panel = Panel(
            f"[bold yellow]{progress_text}[/bold yellow]\n{dataset_text}",
            title="[bold]Current Progress[/bold]",
            box=box.ROUNDED
        )
        #self.console.print(panel)
    
    def show_api_call_info(self, entry_name: str, usage: TokenUsage):
        """Display API call information"""
        table = Table(title=f"API Call: {entry_name}", box=box.SIMPLE, show_header=True)
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="green", width=12)
        
        table.add_row("Input Tokens", f"{usage.input_tokens:,}")
        table.add_row("Output Tokens", f"{usage.output_tokens:,}")
        table.add_row("Total Tokens", f"{usage.total_tokens:,}")
        
        self.console.print(table)
    
    def show_api_call_info_with_cumulative(self, entry_name: str, usage: TokenUsage, token_tracker: TokenTracker):
        """Display API call information with cumulative statistics"""
        # Current call info
        current_table = Table(title=f"🔄 Current API Call: {entry_name}", box=box.SIMPLE, show_header=True)
        current_table.add_column("Metric", style="cyan", width=15)
        current_table.add_column("Value", style="green", width=12)
        
        current_table.add_row("Input Tokens", f"{usage.input_tokens:,}")
        current_table.add_row("Output Tokens", f"{usage.output_tokens:,}")
        current_table.add_row("Total Tokens", f"{usage.total_tokens:,}")
        
        # Cumulative stats
        cumulative_usage = token_tracker.total_usage
        
        cumulative_table = Table(title=f"📈 Cumulative Statistics", box=box.SIMPLE, show_header=True)
        cumulative_table.add_column("Metric", style="magenta", width=15)
        cumulative_table.add_column("Total", style="yellow", width=12)
        
        cumulative_table.add_row("Input Tokens", f"{cumulative_usage.input_tokens:,}")
        cumulative_table.add_row("Output Tokens", f"{cumulative_usage.output_tokens:,}")
        cumulative_table.add_row("All Tokens", f"{cumulative_usage.total_tokens:,}")
        cumulative_table.add_row("API Calls", f"{len(token_tracker.call_history)}")
        
        # Display both tables side by side using Layout
        from rich.columns import Columns
        self.console.print(Columns([current_table, cumulative_table], equal=True))
    
    def show_categories_update(self, categories: Dict[str, str], new_count: int, prev_count: int):
        """Display categories update information"""
        change = new_count - prev_count
        if change > 0:
            change_text = f"[green](+{change})[/green]"
        elif change < 0:
            change_text = f"[red]({change})[/red]"
        else:
            change_text = "[dim](no change)[/dim]"
        
        # Show latest categories (up to 5)
        latest_categories = list(categories.keys())[-5:] if len(categories) > 5 else list(categories.keys())
        categories_text = "\n".join([f"• {cat}" for cat in latest_categories])
        
        if len(categories) > 5:
            categories_text += f"\n[dim]... and {len(categories) - 5} more categories[/dim]"
        
        panel = Panel(
            f"[bold green]Categories: {new_count} {change_text}[/bold green]\n\n"
            f"[dim]Latest categories:[/dim]\n{categories_text}",
            title="[bold]Category Framework[/bold]",
            box=box.ROUNDED
        )
        self.console.print(panel)
    
    def show_cumulative_stats(self, token_tracker: TokenTracker, iteration: int):
        """Display cumulative statistics"""
        usage = token_tracker.total_usage
        
        table = Table(title="📊 Cumulative Statistics", box=box.HEAVY, show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="yellow", width=15)
        table.add_column("Details", style="dim", width=25)
        
        table.add_row(
            "Iterations", 
            str(iteration), 
            f"of {self.total_iterations} max"
        )
        table.add_row(
            "Total Input Tokens", 
            f"{usage.input_tokens:,}", 
            ""
        )
        table.add_row(
            "Total Output Tokens", 
            f"{usage.output_tokens:,}", 
            ""
        )
        table.add_row(
            "Total Tokens", 
            f"{usage.total_tokens:,}", 
            f"Avg: {usage.total_tokens/iteration:.0f}/iter" if iteration > 0 else ""
        )
        
        self.console.print(table)
        self.console.print()  # Add spacing
    
    def show_final_summary(self, categories: Dict[str, str], token_tracker: TokenTracker, 
                          duration: float):
        """Display final analysis summary"""
        usage = token_tracker.total_usage
        
        # Calculate efficiency metrics
        tokens_per_second = usage.total_tokens / duration if duration > 0 else 0
        
        summary_text = (
            f"[bold green]✅ Analysis Complete![/bold green]\n\n"
            f"[bold]📋 Results:[/bold]\n"
            f"• Categories Generated: [yellow]{len(categories)}[/yellow]\n"
            f"• Total Iterations: [yellow]{self.current_iteration}[/yellow]\n"
            f"• Processing Time: [yellow]{duration:.1f}s[/yellow]\n"
            f"• Processing Speed: [yellow]{tokens_per_second:.0f} tokens/sec[/yellow]\n\n"
            f"[bold]🔢 Token Usage:[/bold]\n"
            f"• Input Tokens: [cyan]{usage.input_tokens:,}[/cyan]\n"
            f"• Output Tokens: [cyan]{usage.output_tokens:,}[/cyan]\n"
            f"• Total Tokens: [cyan]{usage.total_tokens:,}[/cyan]"
        )
        
        panel = Panel(
            summary_text,
            title="[bold blue]🎉 Final Summary[/bold blue]",
            box=box.DOUBLE,
            style="green"
        )
        self.console.print(panel)
    
    def show_error(self, error_msg: str):
        """Display error message"""
        panel = Panel(
            f"[bold red]❌ Error:[/bold red]\n{error_msg}",
            title="[bold red]Error[/bold red]",
            box=box.HEAVY,
            style="red"
        )
        self.console.print(panel)
    
    def show_separator(self):
        """Show a visual separator"""
        self.console.print("─" * 80, style="dim")


@dataclass
class LLMConfig:
    """Configuration for OpenAI LLM API"""
    model: str = "o4-mini-2025-04-16"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_completion_tokens: int = 16384
    # temperature: float = 0.7


@dataclass
class Phase1Config:
    """Configuration for Phase 1 Category Builder"""
    max_iterations: int = 10
    output_dir: str = "output"
    output_file: str = "phase1_categories.json"
    log_intermediate_results: bool = True
    llm_config: LLMConfig = None
    input_file: str = "results/top1-general-gemini_o3_4o_4_1.csv"
    breakpoint_file: Optional[str] = None  # Path to resume from


class LLMClient:
    """Client for interacting with OpenAI API with token tracking"""
    
    def __init__(self, config: LLMConfig, token_tracker: TokenTracker):
        self.config = config
        self.token_tracker = token_tracker
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client"""
        try:
            import openai
            
            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key in config.")
            
            # Get model from environment variable or use config default
            model_from_env = os.getenv("OPENAI_MODEL_o")
            if model_from_env:
                self.config.model = model_from_env
                print(f"Using model from environment: {model_from_env}")
            
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=self.config.base_url
            )
            
            print(f"Initialized OpenAI client with model {self.config.model}")
            
        except ImportError as e:
            print(f"Failed to import OpenAI library: {e}")
            print("Please install OpenAI library: pip install openai>=1.0.0")
            raise
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def call_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, TokenUsage]:
        """
        Make a call to the OpenAI API with token tracking
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            
        Returns:
            Tuple of (LLM response text, TokenUsage)
        """
        print(f"Making OpenAI API call with model {self.config.model}")
        
        # Count input tokens
        input_tokens = (
            self.token_tracker.count_tokens(system_prompt) + 
            self.token_tracker.count_tokens(user_prompt)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_completion_tokens=self.config.max_completion_tokens,
                # temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.choices[0].message.content
            
            # Count output tokens
            output_tokens = self.token_tracker.count_tokens(response_text)
            
            # Create usage object
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # If response has usage info, use that instead (more accurate)
            if hasattr(response, 'usage') and response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            return response_text, usage
                
        except Exception as e:
            error_msg = f"OpenAI API call failed with model '{self.config.model}'"
            print(error_msg)
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            print(f"System prompt length: {len(system_prompt)} characters")
            print(f"User prompt length: {len(user_prompt)} characters")
            
            # Log API configuration (without sensitive data)
            print(f"API Configuration:")
            print(f"  Model: {self.config.model}")
            print(f"  Max tokens: {self.config.max_completion_tokens}")
            # print(f"  Temperature: {self.config.temperature}")
            print(f"  Base URL: {self.config.base_url}")
            print(f"  API key configured: {'Yes' if self.config.api_key else 'No'}")
            
            raise RuntimeError(f"OpenAI API call failed: {str(e)}") from e


class Phase1CategoryBuilder:
    """Phase 1 LLM analyzer focused on category building with enhanced console output"""
    
    def __init__(self, config: Phase1Config):
        self.config = config
        self.dataset: Dict[str, str] = {}
        self.categories: Dict[str, str] = {}
        self.dataset_metadata: Dict[str, Dict[str, Any]] = {}  # Store original row data
        self.resume_from_id: Optional[int] = None  # ID to resume from
        self.resume_memory: Optional[Dict[str, str]] = None  # Memory to resume with
        
        # Initialize token tracker and console display
        self.token_tracker = TokenTracker(config.llm_config.model if config.llm_config else "gpt-4o")
        self.console_display = ConsoleDisplay(config.llm_config.model if config.llm_config else "gpt-4o")
        
        # Initialize LLM client
        self.llm_client = None
        if self.config.llm_config:
            self.llm_client = LLMClient(self.config.llm_config, self.token_tracker)
        
        # Handle resume functionality
        if self.config.breakpoint_file:
            self._setup_resume_from_endpoint()
        else:
            self._setup_new_run()
    
    def _setup_resume_from_endpoint(self):
        """Setup resume functionality from endpoint file"""
        endpoint_path = self.config.breakpoint_file
        
        if not os.path.exists(endpoint_path):
            raise ValueError(f"Endpoint file not found: {endpoint_path}")
        
        # Load memory from endpoint file
        try:
            with open(endpoint_path, 'r', encoding='utf-8') as f:
                self.resume_memory = json.load(f)
            print(f"Loaded resume memory with {len(self.resume_memory)} categories from: {endpoint_path}")
        except Exception as e:
            raise ValueError(f"Failed to load endpoint file {endpoint_path}: {e}")
        
        # Extract starting ID from filename
        filename = os.path.basename(endpoint_path)
        self.resume_from_id = self._extract_resume_id_from_filename(filename)
        print(f"Resume from ID: {self.resume_from_id}")
        
        # Setup output directories based on endpoint file location
        endpoint_dir = os.path.dirname(endpoint_path)
        parent_dir = os.path.dirname(endpoint_dir)  # Go up one level from 'output'
        
        # Use the existing timestamped directory structure
        self.timestamped_output_dir = parent_dir
        self.individual_outputs_dir = os.path.join(parent_dir, "output")
        self.prompts_responses_dir = os.path.join(parent_dir, "input")
        
        # Ensure directories exist
        os.makedirs(self.individual_outputs_dir, exist_ok=True)
        os.makedirs(self.prompts_responses_dir, exist_ok=True)
        
        print(f"Resume mode - using existing directories:")
        print(f"  📁 Timestamped output: {self.timestamped_output_dir}")
        print(f"  📁 Individual outputs: {self.individual_outputs_dir}")
        print(f"  📝 Prompt/Response logs: {self.prompts_responses_dir}")
    
    def _setup_new_run(self):
        """Setup new run with fresh directories"""
        # Create output directory structure
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Generate timestamped subdirectory name based on input file and current Central Time
        central_tz = pytz.timezone('US/Central')
        current_time = datetime.now(central_tz)
        timestamp = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        
        # Extract input filename without extension and path
        input_filename = os.path.splitext(os.path.basename(self.config.input_file))[0]
        
        # Create timestamped subdirectory name
        timestamped_subdir = f"{input_filename}_{timestamp}"
        self.timestamped_output_dir = os.path.join(self.config.output_dir, timestamped_subdir)
        os.makedirs(self.timestamped_output_dir, exist_ok=True)
        
        # Create subdirectories for individual outputs within the timestamped directory
        self.individual_outputs_dir = os.path.join(self.timestamped_output_dir, "output")
        self.prompts_responses_dir = os.path.join(self.timestamped_output_dir, "input")
        os.makedirs(self.individual_outputs_dir, exist_ok=True)
        os.makedirs(self.prompts_responses_dir, exist_ok=True)
    
    def _extract_resume_id_from_filename(self, filename: str) -> int:
        """Extract the resume ID from breakpoint filename
        
        Example: phase1-output-round1-id_327-imageid_243.json -> returns 328 (327 + 1)
        """
        # Remove extension
        name_without_ext = filename.replace('.json', '')
        
        # Look for the pattern id_XXX in the filename
        id_match = re.search(r'id_(\d+)', name_without_ext)
        
        if not id_match:
            raise ValueError(f"Cannot extract resume ID from filename: {filename}. Expected format: phase1-output-roundX-id_XXX-imageid_XXX.json")
        
        # Get the ID number and add 1 for resume
        resume_id = int(id_match.group(1)) + 1
        return resume_id
    
    def _parse_clue_list_pandas(self, clue_list_str: str) -> str:
        """Parse clue_list string from pandas DataFrame with improved JSON handling"""
        # Handle NaN values and non-string types from pandas
        if pd.isna(clue_list_str):
            return ""
        
        # Convert to string if it's not already
        if not isinstance(clue_list_str, str):
            clue_list_str = str(clue_list_str)
        
        try:
            # First try to parse as JSON directly
            clue_list = json.loads(clue_list_str)
            if isinstance(clue_list, list):
                # Join the clues with newlines for better readability
                return "\n".join(f"- {clue}" for clue in clue_list)
            else:
                return str(clue_list)
                
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to clean up the string and parse again
            try:
                # Remove outer quotes if present
                cleaned = clue_list_str.strip()
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                
                # Replace double quotes with single quotes for JSON parsing
                cleaned = cleaned.replace('""', '"')
                
                clue_list = json.loads(cleaned)
                if isinstance(clue_list, list):
                    return "\n".join(f"- {clue}" for clue in clue_list)
                else:
                    return str(clue_list)
                    
            except (json.JSONDecodeError, ValueError):
                # If all parsing fails, return the raw string
                return clue_list_str
    
    def load_csv_dataset(self, csv_path: str) -> bool:
        """Load dataset from CSV file using pandas"""
        try:
            # Use pandas for CSV parsing
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            print(f"Loaded CSV with {len(df)} total rows")
            
            # Verify required columns exist
            required_columns = ['id', 'src', 'clue_list']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns in CSV: {missing}")
            
            # Filter out rows with null clue_list
            df_filtered = df[df['clue_list'].notna()].copy()
            print(f"After filtering null clue_list: {len(df_filtered)} rows")
            
            # If resuming, filter to start from resume_from_id
            if self.resume_from_id is not None:
                df_filtered = df_filtered[df_filtered['id'] >= self.resume_from_id].copy()
                print(f"Resume mode: filtering to start from ID {self.resume_from_id}, {len(df_filtered)} rows remaining")
            
            dataset = {}
            dataset_metadata = {}
            for idx, row in df_filtered.iterrows():
                # Create a unique key using row index and image_id (if available)
                # This ensures all records are preserved even if image_id is duplicated
                if 'image_id' in df.columns and pd.notna(row['image_id']):
                    key = f"{idx}_{row['image_id']}"
                    image_id = str(row['image_id'])
                else:
                    key = f"{idx}_{row['id']}"
                    image_id = None
                
                clue_text = self._parse_clue_list_pandas(row['clue_list'])
                
                # Only add entries with non-empty clue text
                if clue_text.strip():
                    dataset[key] = clue_text
                    # Store metadata for file naming
                    dataset_metadata[key] = {
                        'id': str(row['id']),
                        'image_id': image_id,
                        'src': str(row['src']) if pd.notna(row['src']) else None,
                        'row_index': idx
                    }
            
            self.dataset = dataset
            self.dataset_metadata = dataset_metadata
            
            if self.resume_from_id is not None:
                print(f"Successfully loaded dataset with {len(dataset)} valid entries from {csv_path} (resume mode from ID {self.resume_from_id})")
            else:
                print(f"Successfully loaded dataset with {len(dataset)} valid entries from {csv_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load CSV dataset from {csv_path}: {e}")
            return False
    
    def load_dataset(self, dataset_path: str) -> bool:
        """Load dataset from JSON file (legacy support)"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            self.dataset = dataset.copy()
            print(f"Loaded dataset with {len(dataset)} entries from {dataset_path}")
            return True
        except Exception as e:
            print(f"Failed to load dataset from {dataset_path}: {e}")
            return False
    
    def load_dataset_from_dict(self, dataset: Dict[str, str]) -> None:
        """Load dataset from dictionary (legacy support)"""
        self.dataset = dataset.copy()
        print(f"Loaded dataset with {len(dataset)} entries from dictionary")
    
    def _generate_filename(self, entry_key: str, base_name: str, iteration: int, extension: str = ".json") -> str:
        """Generate filename based on base name, iteration, id, and image_id"""
        if entry_key in self.dataset_metadata:
            metadata = self.dataset_metadata[entry_key]
            id_part = metadata['id']
            image_id_part = metadata['image_id']
            
            # Remove extension from base_name if present
            base_name_clean = base_name.replace('.json', '')
            
            if image_id_part:
                filename = f"{base_name_clean}-round{iteration}-id_{id_part}-imageid_{image_id_part}{extension}"
            else:
                filename = f"{base_name_clean}-round{iteration}-id_{id_part}{extension}"
        else:
            # Fallback for legacy data or missing metadata
            base_name_clean = base_name.replace('.json', '')
            filename = f"{base_name_clean}-round{iteration}-{entry_key}{extension}"
        
        # Sanitize filename to remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return filename
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Phase 1"""
        return """"""
    
    def _get_user_prompt(self, single_entry: list[str], memory: Dict[str, str]) -> str:
        """Get user prompt for Phase 1 - modified for single entry processing"""
        return f"""Your task is to extract a NON-OVERLAPPING list of general categories from a batch of clues for image gelocation, and write a concise definition for each category.
Rules for a Good Category:
• 2–4-word noun phrase, capitalised in Title Case (e.g., "Street Layout").
• Covers multiple possible clues; avoid brand, place, or time names.
• All Categories must be mutually exclusive; resolve overlaps by widening/merging.
Definition rules:
• 1st sentence = core concept; 2nd and following sentences (optional) = scope limit or exclusion.
• Do NOT embed concrete examples or proper nouns unless vital to meaning.
• Lack of features or absence of something can not be clue categories for image localization, only the existing features.
• Keep the whole memory capturing a minimal yet highly informative set of clue categories extracted from the dataset after your actions.

Inputs
1. <dataset> [list[str]] = {json.dumps(single_entry, ensure_ascii=False, indent=2)}

2. <memory> [Dict[str, str]] = {json.dumps(memory, ensure_ascii=False, indent=2)}


First, you should think about the  <dataset> and give me a list of <candidate_category> that can conclude all the items in the <dataset>.
List:
```python
candidate_categories = [
    "<candidate_category1>",
    "<candidate_category2>",
    ...
]
```

After comparing the <candidate_categories> with the <memory>, you should choose from one of the following steps with format as below (json requires strict formatting, with all keys and string values enclosed in double quotes, disallowing single quotes or unquoted property names):

(1) If you think you should revise the incorrect clue or merge some duplicate clues' categories with definitions based on your analysis to make the <Memory> more clear:
Think: put your thoughts here.
Json:
```json
# Put the whole memory after your revised or merged actions with definition in {{ "Category₁": "Definition₁", "Category₂": "Definition₂", … }} here.
```

(2). If you think you don't need any above actions, just directly return <memory>:
Json:
```json
# Put the whole original memory in {{ "Category₁": "Definition₁", "Category₂": "Definition₂", … }} here.
```

(3). If you think you should add a new category of clues in the <dataset> but missing in the memory:
Think: put your thoughts here.
Json:
```json
# Put the whole memory with your updated clues with definition in 
{{ "Category₁": "Definition₁", "Category₂": "Definition₂", … }}
```
"""
 #≤2 sentences,   
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from OpenAI, handling potential formatting issues"""
        original_response = response
        
        # Check if response is empty or None
        if not response or not response.strip():
            error_msg = "OpenAI returned empty response"
            print(error_msg)
            print(f"Original response: '{original_response}'")
            # Return None to indicate parsing failure, let caller handle it
            return None
        
        try:
            # Remove code fences if present
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*$', '', response)
            
            # Remove any leading/trailing whitespace
            response = response.strip()
            
            # Check if response starts with expected JSON structure
            #if not response.startswith('{'):
            #    print(f"Warning: Response doesn't start with '{{': {response[:100]}...")
            
            # Try to parse directly
            parsed = json.loads(response)
            
            # Validate that it's a dictionary
            if not isinstance(parsed, dict):
                print(f"Warning: Expected dictionary, got {type(parsed)}")
                return None
                
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"Initial JSON parsing failed: {e}")
            print(f"Response length: {len(response)}")
            print(f"Response: {response}")
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json_match.group()
                    print(f"Attempting to parse extracted JSON: {extracted_json[:100]}...")
                    parsed = json.loads(extracted_json)
                    if isinstance(parsed, dict):
                        return parsed
                    else:
                        raise ValueError(f"Extracted JSON is not a dictionary: {type(parsed)}")
                except json.JSONDecodeError as e2:
                    print(f"Extracted JSON parsing also failed: {e2}")
            
            # Provide detailed error information
            error_msg = f"Failed to parse JSON response from OpenAI. Original response length: {len(original_response)}"
            print(error_msg)
            print(f"Full original response: {original_response}")
            
            if json_match:
                print(f"Extracted JSON: {json_match.group()}")
            else:
                print("No JSON structure found in response")
                # Return None to indicate parsing failure, let caller handle it
                return None
            
            # Return None to indicate parsing failure, let caller handle it
            return None
    
    def _ask_user_continue_next_round(self, current_round: int, categories_count: int) -> bool:
        """Ask user whether to continue to the next round"""
        print("\n" + "="*80)
        print(f"🎯 ROUND {current_round} COMPLETED!")
        print("="*80)
        print(f"📊 Current Statistics:")
        print(f"   • Categories Generated: {categories_count}")
        print(f"   • Round {current_round} Finished")
        print()
        
        while True:
            try:
                user_input = input("🤔 Do you want to continue to the next round? (y/yes/n/no): ").strip().lower()
                
                if user_input in ['y', 'yes']:
                    print(f"✅ Continuing to Round {current_round + 1}...")
                    print("="*80)
                    return True
                elif user_input in ['n', 'no']:
                    print("🛑 User chose to exit. Finishing analysis...")
                    print("="*80)
                    return False
                else:
                    print(f"❌ Invalid input: '{user_input}'. Please enter 'y/yes' to continue or 'n/no' to exit.")
                    
            except KeyboardInterrupt:
                print("\n🛑 User interrupted with Ctrl+C. Finishing analysis...")
                print("="*80)
                return False
            except EOFError:
                print("\n🛑 End of input detected. Finishing analysis...")
                print("="*80)
                return False
    
    def build_categories(self) -> Dict[str, str]:
        """Build category framework using OpenAI with enhanced console output"""
        self.console_display.show_header("Phase 1: Category Builder")
        
        if not self.llm_client:
            error_msg = "OpenAI client not initialized. Cannot proceed with OpenAI-powered analysis."
            self.console_display.show_error(error_msg)
            print(error_msg)
            raise RuntimeError(error_msg)
        
        if not self.dataset:
            error_msg = "Dataset not loaded. Call load_csv_dataset() first."
            self.console_display.show_error(error_msg)
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # Initialize memory - use resume memory if available, otherwise start fresh
        if self.resume_memory is not None:
            memory = self.resume_memory.copy()
            print(f"🔄 Resume mode: Starting with {len(memory)} existing categories")
        else:
            memory = {}
            print("🆕 New run: Starting with empty memory")
        
        round_number = 1
        
        # Note: Removed main output file - all outputs now only in timestamped directory
        print(f"📁 All outputs will be saved to timestamped directory: {self.timestamped_output_dir}")
        print(f"📄 Individual outputs: {self.individual_outputs_dir}")
        print(f"📝 Prompt/Response logs: {self.prompts_responses_dir}")
        
        system_prompt = self._get_system_prompt()
        
        # Round-based processing loop
        while True:
            prev_category_count = len(memory)
            
            # Show round start
            print(f"\n🚀 Starting Round {round_number}")
            print(f"📊 Dataset size: {len(self.dataset)} entries")
            print(f"📋 Current categories: {len(memory)}")
            
            if self.resume_from_id is not None:
                print(f"Phase 1 - Round {round_number} (Resume from ID {self.resume_from_id})")
            else:
                print(f"Phase 1 - Round {round_number}")
            
            try:
                # Process each dataset entry individually
                for entry_idx, (filename, clue_text) in enumerate(self.dataset.items(), 1):
                    print(f"Processing entry {entry_idx}/{len(self.dataset)}: {filename}")
                    
                    # Create single entry dict for this round
                    single_entry = [clue_text]
                    
                    user_prompt = self._get_user_prompt(single_entry, memory)
                    response, usage = self.llm_client.call_llm(system_prompt, user_prompt)
                    
                    # Add debugging output for response
                    print(f"OpenAI Response (first 200 chars): {response[:200]}")
                    if len(response) > 200:
                        print(f"OpenAI Response (last 100 chars): {response[-100:]}")
                    
                    # Track token usage
                    self.token_tracker.add_usage(usage, f"Round {round_number}, Entry {filename}")
                    
                    # Show API call info with cumulative stats
                    self.console_display.show_api_call_info_with_cumulative(
                        f"{filename} ({entry_idx}/{len(self.dataset)})", 
                        usage, 
                        self.token_tracker
                    )
                    
                    new_memory = self._parse_json_response(response)
                    
                    # Variables to track what memory to use and if entry was skipped
                    memory_to_save = memory  # Default to previous memory
                    entry_skipped = False
                    skip_reason = ""
                    
                    # Handle JSON parsing failure
                    if new_memory is None:
                        print(f"⚠️  JSON parsing failed for entry {filename}. Using previous memory.")
                        print(f"   Previous memory has {len(memory)} categories")
                        entry_skipped = True
                        skip_reason = "JSON parsing failed"
                        # memory_to_save remains as previous memory
                    
                    # Validate response format
                    elif not isinstance(new_memory, dict) or not all(isinstance(v, str) for v in new_memory.values()):
                        error_msg = f"Invalid response format from OpenAI in Phase 1, round {round_number}, entry {filename}. Expected Dict[str, str], got: {type(new_memory)}"
                        print(f"⚠️  {error_msg}")
                        print(f"   Using previous memory with {len(memory)} categories")
                        entry_skipped = True
                        skip_reason = f"Invalid response format: {type(new_memory)}"
                        # memory_to_save remains as previous memory
                    
                    # Check if too many categories were deleted (more than 8)
                    elif len(memory) - len(new_memory) > 8:
                        categories_deleted = len(memory) - len(new_memory)
                        print(f"⚠️  Too many categories deleted ({categories_deleted} > 8) for entry {filename}")
                        print(f"   Previous memory: {len(memory)} categories")
                        print(f"   New memory: {len(new_memory)} categories")
                        print(f"   Using previous memory.")
                        entry_skipped = True
                        skip_reason = f"Too many categories deleted: {categories_deleted}"
                        # memory_to_save remains as previous memory
                    
                    else:
                        # Valid response, update memory
                        memory = new_memory
                        memory_to_save = memory
                    
                    # Generate individual filenames with round number
                    individual_filename = self._generate_filename(filename, "phase1-output", round_number, '.json')
                    prompt_response_filename = self._generate_filename(filename, "phase1-input", round_number, '.json')
                    
                    # Always save individual category result (even for skipped entries)
                    individual_path = os.path.join(self.individual_outputs_dir, individual_filename)
                    sorted_memory = dict(sorted(memory_to_save.items()))
                    with open(individual_path, 'w', encoding='utf-8') as f:
                        json.dump(sorted_memory, f, ensure_ascii=False, indent=2)
                    
                    # Always save complete prompt and response data (even for skipped entries)
                    prompt_response_data = {
                        "metadata": {
                            "round": round_number,
                            "entry_index": entry_idx,
                            "entry_key": filename,
                            "timestamp": time.time(),
                            "model": self.llm_client.config.model,
                            "resume_mode": self.resume_from_id is not None,
                            "resume_from_id": self.resume_from_id,
                            "entry_skipped": entry_skipped,
                            "skip_reason": skip_reason,
                            "token_usage": {
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "total_tokens": usage.total_tokens
                            }
                        },
                        "input": {
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "single_entry": single_entry,
                            "previous_memory": dict(sorted(memory.items())) if round_number > 1 or self.resume_memory else {}
                        },
                        "output": {
                            "raw_response": response,
                            "parsed_categories": sorted_memory,
                            "category_count": len(sorted_memory),
                            "memory_source": "previous" if entry_skipped else "updated"
                        }
                    }
                    
                    prompt_response_path = os.path.join(self.prompts_responses_dir, prompt_response_filename)
                    with open(prompt_response_path, 'w', encoding='utf-8') as f:
                        json.dump(prompt_response_data, f, ensure_ascii=False, indent=2)
                    
                    # Note: Removed the extra output to self.config.output_dir - all outputs now only in timestamped directory
                    
                    if self.config.log_intermediate_results:
                        status_msg = "SKIPPED" if entry_skipped else "PROCESSED"
                        print(f"Phase 1 - Entry {filename} [{status_msg}]: {len(memory_to_save)} categories")
                        if entry_skipped:
                            print(f"  🚫 Skip reason: {skip_reason}")
                        print(f"  📄 Individual result: {individual_path}")
                        print(f"  📝 Prompt/Response: {prompt_response_path}")
                    
                    # Show progress every 10 entries
                    if entry_idx % 10 == 0:
                        print(f"📊 Progress: {entry_idx}/{len(self.dataset)} entries processed, {len(memory)} categories so far")
                        print(f"📁 Individual outputs saved to: {self.individual_outputs_dir}")
                        print(f"📝 Prompt/Response logs saved to: {self.prompts_responses_dir}")
                
                # Show categories update for this round
                self.console_display.show_categories_update(memory, len(memory), prev_category_count)
                
                # Show cumulative statistics for this round
                self.console_display.show_cumulative_stats(self.token_tracker, round_number)
                
                # Add separator for readability
                self.console_display.show_separator()
                
                # Ask user whether to continue to next round
                if not self._ask_user_continue_next_round(round_number, len(memory)):
                    # User chose to exit
                    break
                
                # User chose to continue, increment round number
                round_number += 1
                    
            except Exception as e:
                error_msg = f"Phase 1 round {round_number} failed with OpenAI API error: {str(e)}"
                self.console_display.show_error(error_msg)
                print(error_msg)
                print(f"System prompt: {system_prompt[:200]}...")
                raise RuntimeError(error_msg) from e
        
        # Sort categories alphabetically
        sorted_memory = dict(sorted(memory.items()))
        self.categories = sorted_memory
        
        # Save final result only in timestamped directory
        final_output_path = os.path.join(self.timestamped_output_dir, "final_categories.json")
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_memory, f, ensure_ascii=False, indent=2)
        
        print(f"Phase 1 completed after {round_number} rounds. Generated {len(sorted_memory)} categories")
        print(f"📁 Final results saved to: {final_output_path}")
        print(f"📁 Timestamped output directory: {self.timestamped_output_dir}")
        print(f"📁 Individual outputs directory: {self.individual_outputs_dir}")
        print(f"📝 Prompt/Response logs directory: {self.prompts_responses_dir}")
        
        return sorted_memory
    
    def run_analysis(self) -> Dict[str, str]:
        """Run Phase 1 category building analysis with enhanced console output"""
        if not self.dataset:
            # Try to load default CSV file
            print("No dataset loaded. Attempting to load default CSV file...")
            
            if not self.load_csv_dataset(self.config.input_file):
                error_msg = f"Failed to load dataset from {self.config.input_file}"
                self.console_display.show_error(error_msg)
                raise ValueError(error_msg)
        
        if not self.llm_client:
            error_msg = "OpenAI client not initialized. Check your API configuration and credentials."
            self.console_display.show_error(error_msg)
            print(error_msg)
            raise RuntimeError(error_msg)
        
        print("Starting Phase 1 category building analysis (OpenAI-powered)")
        print(f"Dataset: {len(self.dataset)} entries")
        start_time = time.time()
        
        try:
            # Build categories using OpenAI
            categories = self.build_categories()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Show final summary
            self.console_display.show_final_summary(categories, self.token_tracker, duration)
            
            print(f"Phase 1 analysis completed successfully in {duration:.2f} seconds")
            
            return categories
            
        except Exception as e:
            error_msg = f"OpenAI analysis failed: {str(e)}"
            self.console_display.show_error(error_msg)
            print(error_msg)
            print("Analysis terminated due to OpenAI API error. No fallback available.")
            raise RuntimeError(error_msg) from e
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results"""
        if not self.categories:
            return {"error": "No analysis results available"}
        
        return {
            "dataset_size": len(self.dataset),
            "total_categories": len(self.categories),
            "categories": list(self.categories.keys()),
            "openai_model": self.llm_client.config.model if self.llm_client else "unknown",
            "output_file": os.path.join(self.timestamped_output_dir, "final_categories.json"),
            "timestamped_output_dir": self.timestamped_output_dir,
            "individual_outputs_dir": self.individual_outputs_dir,
            "prompts_responses_dir": self.prompts_responses_dir,
            "token_usage": {
                "total_tokens": self.token_tracker.total_usage.total_tokens,
                "input_tokens": self.token_tracker.total_usage.input_tokens,
                "output_tokens": self.token_tracker.total_usage.output_tokens
            }
        }


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 LLM Analyzer - Category Builder (OpenAI-powered) with Enhanced Console Output")
    parser.add_argument("--input-csv", default="results/training_set-shuffled-general-top1-metro_correct-IQR.csv", help="Path to input CSV file")
    parser.add_argument("--dataset", help="Path to legacy JSON dataset file (optional)")
    parser.add_argument("--output-dir", default="phase1_output", help="Output directory")
    parser.add_argument("--output-file", default="phase1_categories-training_set-shuffled-general-top1-metro_correct-IQR.json", help="Output filename")
    parser.add_argument("--model", default="o4-mini-2025-04-16", help="OpenAI model name (default: o4-mini-2025-04-16)")
    parser.add_argument("--breakpoint", help="Path to JSON file to resume from (optional)")
    parser.add_argument("--gui", action="store_true", help="Enable real-time GUI monitoring window")
    
    args = parser.parse_args()
    
    # Create LLM configuration (required)
    llm_config = LLMConfig(
        model=args.model
    )
    
    # Create analysis configuration
    config = Phase1Config(
        # max_iterations=5,  # No longer used - now uses round-based processing with user confirmation
        output_dir=args.output_dir,
        output_file=args.output_file,
        log_intermediate_results=True,
        llm_config=llm_config,
        input_file=args.input_csv,
        breakpoint_file=args.breakpoint
    )
    
    # Create analyzer
    try:
        analyzer = Phase1CategoryBuilder(config)
    except Exception as e:
        console = Console()
        console.print(Panel(
            f"[bold red]❌ Failed to initialize analyzer:[/bold red]\n{e}\n\n"
            f"[dim]Please check your configuration and file paths.[/dim]",
            title="[bold red]Initialization Error[/bold red]",
            box=box.HEAVY,
            style="red"
        ))
        print(f"Failed to initialize analyzer: {e}")
        if args.breakpoint:
            print("Please check that the breakpoint file exists and is valid.")
        else:
            print("Please check your OpenAI API configuration and credentials.")
        sys.exit(1)
    
    # Load dataset
    if args.dataset:
        # Legacy JSON support
        if not analyzer.load_dataset(args.dataset):
            sys.exit(1)
    else:
        # Load CSV dataset
        if not analyzer.load_csv_dataset(args.input_csv):
            print(f"Failed to load CSV: {args.input_csv}")
            sys.exit(1)
    
    # Setup GUI monitoring if requested
    gui_monitor = None
    analysis_thread = None
    if args.gui:
        try:
            from gui_monitor import create_gui_monitor
            import threading
            
            # Determine the JSON output directory for monitoring
            json_monitor_dir = analyzer.individual_outputs_dir
            print(f"🖥️  GUI monitoring enabled. Monitoring directory: {json_monitor_dir}")
            
            # Create GUI monitor
            gui_monitor = create_gui_monitor(json_monitor_dir, update_interval=2.0)
            
            # For macOS compatibility, run analysis in background thread and GUI on main thread
            analysis_result = {"categories": None, "error": None, "summary": None}
            
            def run_analysis():
                """Run analysis in background thread"""
                try:
                    categories = analyzer.run_analysis()
                    summary = analyzer.get_analysis_summary()
                    analysis_result["categories"] = categories
                    analysis_result["summary"] = summary
                    print("\n✅ Analysis completed successfully!")
                except Exception as e:
                    analysis_result["error"] = e
                    print(f"\n❌ Analysis failed: {e}")
            
            # Start analysis in background thread
            analysis_thread = threading.Thread(target=run_analysis, daemon=True)
            analysis_thread.start()
            
            print("🚀 Starting GUI monitoring on main thread...")
            print("📊 Analysis is running in background...")
            
            # Start GUI monitoring on main thread (this will block)
            gui_monitor.start_monitoring()
            
            # Wait for analysis to complete
            if analysis_thread.is_alive():
                print("⏳ Waiting for analysis to complete...")
                analysis_thread.join()
            
            # Check results
            if analysis_result["error"]:
                raise analysis_result["error"]
            
            categories = analysis_result["categories"]
            summary = analysis_result["summary"]
            
        except ImportError as e:
            print(f"⚠️  Warning: GUI monitoring not available. Missing dependencies: {e}")
            print("    Install required packages: conda install matplotlib scikit-learn")
            args.gui = False
        except Exception as e:
            print(f"⚠️  Warning: Failed to start GUI monitoring: {e}")
            args.gui = False
    
    # If GUI is not enabled, run analysis normally
    if not args.gui:
        try:
            # Run analysis normally
            categories = analyzer.run_analysis()
            summary = analyzer.get_analysis_summary()
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            print("Check the logs above for detailed error information.")
            sys.exit(1)
    
    # Print summary to console (in addition to rich display)
    if args.gui and analysis_result.get("summary"):
        summary = analysis_result["summary"]
    
    print("\n" + "="*50)
    if args.breakpoint:
        print("PHASE 1 CATEGORY BUILDER SUMMARY (RESUME MODE)")
    else:
        print("PHASE 1 CATEGORY BUILDER SUMMARY")
    print("="*50)
    print(f"Dataset size: {summary['dataset_size']}")
    print(f"Total categories: {summary['total_categories']}")
    print(f"OpenAI Model: {summary['openai_model']}")
    print(f"Output file: {summary['output_file']}")
    print(f"Total tokens used: {summary['token_usage']['total_tokens']:,}")
    
    if args.breakpoint:
        print(f"Resume from: {args.breakpoint}")
        print(f"Resume from ID: {analyzer.resume_from_id}")
        print(f"Started with: {len(analyzer.resume_memory)} categories")
    
    print("\nGenerated categories:")
    for i, category in enumerate(summary['categories'], 1):
        print(f"  {i:2d}. {category}")
    
    print(f"\nResults saved to: {summary['output_file']}")
    print(f"📁 Timestamped output directory: {summary['timestamped_output_dir']}")
    print(f"📁 Individual outputs: {summary['individual_outputs_dir']}")
    print(f"📝 Prompt/Response logs: {summary['prompts_responses_dir']}")
    
    if args.gui:
        print("\n🖥️  GUI monitoring was enabled during analysis.")
        print("   Analysis completed successfully with real-time monitoring.")


if __name__ == "__main__":
    main() 
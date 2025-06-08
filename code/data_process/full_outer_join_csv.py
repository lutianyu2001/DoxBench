#!/usr/bin/env python3
"""
CSV Merger - Merge multiple CSV/TSV files with FULL OUTER JOIN logic.
Handles files with different column structures gracefully.
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Union
import sys


def collect_files_from_directories(directories: List[str], extension: str) -> List[Path]:
    """
    Collect all files with specified extension from given directories.

    Args:
        directories: List of directory paths
        extension: File extension to search for ('csv' or 'tsv')

    Returns:
        List of Path objects for found files
    """
    files = []
    pattern = f"*.{extension}"

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory '{directory}' does not exist", file=sys.stderr)
            continue
        if not dir_path.is_dir():
            print(f"Warning: '{directory}' is not a directory", file=sys.stderr)
            continue

        found_files = list(dir_path.glob(pattern))
        files.extend(found_files)
        print(f"Found {len(found_files)} {extension} files in '{directory}'")

    return files


def validate_files(file_paths: List[Union[str, Path]]) -> List[Path]:
    """
    Validate that all provided files exist and are readable.

    Args:
        file_paths: List of file paths (strings or Path objects)

    Returns:
        List of valid Path objects
    """
    valid_files = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File '{file_path}' does not exist", file=sys.stderr)
            continue
        if not path.is_file():
            print(f"Warning: '{file_path}' is not a file", file=sys.stderr)
            continue
        valid_files.append(path)

    return valid_files


def merge_csv_files(file_paths: List[Path], extension: str) -> pd.DataFrame:
    """
    Merge multiple CSV/TSV files using FULL OUTER JOIN logic.

    Args:
        file_paths: List of file paths to merge
        extension: File extension ('csv' or 'tsv')

    Returns:
        Merged DataFrame containing all unique columns from all files
    """
    if not file_paths:
        raise ValueError("No valid files to merge")

    # Determine separator based on extension
    separator = '\t' if extension == 'tsv' else ','

    dataframes = []

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep=separator)
            # df['_source_file'] = file_path.name
            dataframes.append(df)
            print(f"Loaded '{file_path}' with {len(df)} rows and {len(df.columns)-1} columns")
        except Exception as e:
            print(f"Error reading '{file_path}': {e}", file=sys.stderr)
            continue

    if not dataframes:
        raise ValueError("No files were successfully loaded")

    # Concatenate all dataframes (equivalent to FULL OUTER JOIN)
    # This automatically handles different column structures
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)

    print(f"\nMerged result: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"Unique columns: {sorted(merged_df.columns.tolist())}")

    return merged_df


def main():
    """Main function to handle command line arguments and orchestrate the merging process."""
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV/TSV files with FULL OUTER JOIN logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f file1.csv file2.csv -o merged.csv
  %(prog)s -d /path/to/csvs -e tsv -o merged.tsv
  %(prog)s -f file1.csv -d /path/to/more/csvs -o combined.csv
        """
    )

    parser.add_argument(
        '-e', '--extension',
        choices=['csv', 'tsv'],
        default='csv',
        help='File extension to process (default: csv)'
    )

    parser.add_argument(
        '-f', '--file',
        action="append",
        default=[],
        help='Specific file paths to merge (can specify multiple)'
    )

    parser.add_argument(
        '-d', '--directory',
        action="append",
        default=[],
        help='Directory paths to scan for files (can specify multiple)'
    )

    parser.add_argument(
        '-o', '--output',
        default="merged.csv",
        help='Output file path for merged result'
    )

    args = parser.parse_args()

    # Validate that at least one input source is provided
    if not args.file and not args.directory:
        parser.error("Must specify either --file or --directory (or both)")

    try:
        # Collect all file paths
        all_files = []

        # Add explicitly specified files
        if args.file:
            explicit_files = validate_files(args.file)
            all_files.extend(explicit_files)
            print(f"Added {len(explicit_files)} explicit files")

        # Add files from directories
        if args.directory:
            directory_files = collect_files_from_directories(args.directory, args.extension)
            validated_dir_files = validate_files(directory_files)
            all_files.extend(validated_dir_files)

        # Remove duplicates while preserving order
        unique_files = []
        seen = set()
        for file_path in all_files:
            resolved_path = file_path.resolve()
            if resolved_path not in seen:
                seen.add(resolved_path)
                unique_files.append(file_path)

        print(f"\nTotal unique files to merge: {len(unique_files)}")

        # Merge files
        merged_df = merge_csv_files(unique_files, args.extension)

        # Save result
        output_path = Path(args.output)
        output_separator = '\t' if output_path.suffix.lower() == '.tsv' else ','

        merged_df.to_csv(output_path, sep=output_separator, index=False)
        print(f"\nMerged file saved to: {output_path}")
        print(f"Final dimensions: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

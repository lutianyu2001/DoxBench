#!/usr/bin/env python3
"""
File sampling tool with related metadata preservation.
Samples files from each specified directory while maintaining directory structure
and corresponding metadata from related file.
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set


def normalize_path(path: str) -> str:
    """Normalize path by removing leading './' and converting to forward slashes."""
    normalized = str(Path(path))
    # Remove leading './' if present
    if normalized.startswith('./'):
        normalized = normalized[2:]
    # Convert backslashes to forward slashes for consistency
    normalized = normalized.replace('\\', '/')
    return normalized


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample files from each directory separately with metadata preservation"
    )

    parser.add_argument(
        "-d", "--directory",
        action="append",
        required=True,
        help="Directories to sample from (relative paths, can be used multiple times)"
    )

    parser.add_argument(
        "-r", "--related",
        help="Related metadata file (CSV format, optional)"
    )

    parser.add_argument(
        "-s", "--size",
        type=int,
        required=True,
        help="Sample size per directory"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling"
    )

    parser.add_argument(
        "-o", "--output",
        default="sampled",
        help="Output directory (default: sampled)"
    )

    return parser.parse_args()


def load_related_file(related_path: str) -> Dict[str, Dict[str, str]]:
    """Load related metadata file and create filename-to-data mapping."""
    metadata = {}

    with open(related_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            original_filename = row['filename']
            normalized_filename = normalize_path(original_filename)
            # Store both original and normalized as keys for robustness
            metadata[original_filename] = row
            metadata[normalized_filename] = row

    return metadata


def collect_files_by_directory(directories: List[str]) -> Dict[str, List[str]]:
    """Collect files from each directory separately."""
    files_by_dir = {}

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist")
            continue

        print(f"Processing directory: {directory}")
        files_in_dir = []
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                # Normalize the path for consistent comparison
                normalized_path = normalize_path(str(file_path))
                files_in_dir.append(normalized_path)

        files_by_dir[directory] = files_in_dir
        print(f"  Found {len(files_in_dir)} files in {directory}")

    return files_by_dir


def sample_files_by_directory(
    files_by_dir: Dict[str, List[str]],
    sample_size: int,
    metadata: Dict[str, Dict[str, str]] = None,
    seed: int = None
) -> List[str]:
    """Sample files from each directory separately."""
    if seed is not None:
        random.seed(seed)

    all_sampled_files = []

    for directory, files in files_by_dir.items():
        # Filter files that have metadata if metadata is provided
        if metadata:
            valid_files = [f for f in files if f in metadata]
            print(f"Directory {directory}: {len(valid_files)}/{len(files)} files have metadata")
        else:
            valid_files = files

        if not valid_files:
            print(f"Warning: No valid files in directory {directory}")
            continue

        # Sample from this directory
        actual_sample_size = min(sample_size, len(valid_files))
        if actual_sample_size < sample_size:
            print(f"Warning: Directory {directory} has only {len(valid_files)} valid files, sampling {actual_sample_size}")

        sampled = random.sample(valid_files, actual_sample_size)
        all_sampled_files.extend(sampled)
        print(f"Sampled {len(sampled)} files from {directory}")

    return all_sampled_files


def create_output_structure(sampled_files: List[str], output_dir: str) -> None:
    """Create output directory structure and copy sampled files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for file_path in sampled_files:
        # Use the normalized path as both source and destination
        source = Path(file_path)
        destination = output_path / file_path

        # Create parent directories if they don't exist
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(source, destination)
        print(f"Copied: {file_path} -> {destination}")


def create_related_output(
    sampled_files: List[str],
    metadata: Dict[str, Dict[str, str]],
    output_dir: str,
    original_related_path: str
) -> None:
    """Create new related file with sampled files' metadata."""
    output_related_path = Path(output_dir) / Path(original_related_path).name
    sampled_metadata = []

    for file_path in sampled_files:
        if file_path in metadata:
            # Use the original row data but update filename to normalized format
            updated_row = metadata[file_path].copy()
            updated_row['filename'] = f"./{file_path}"
            sampled_metadata.append(updated_row)
        else:
            print(f"Warning: {file_path} not found in related file")

    # Write the new related file
    if sampled_metadata:
        with open(output_related_path, 'w', encoding='utf-8', newline='') as file:
            if sampled_metadata:
                fieldnames = sampled_metadata[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()
                writer.writerows(sampled_metadata)

        print(f"Created related file: {output_related_path}")


def main():
    """Main execution function."""
    args = parse_arguments()

    # Debug: Print all directories being processed
    print(f"All directories to process: {args.directory}")

    # Load metadata from related file if provided
    metadata = {}
    if args.related:
        print(f"Loading metadata from: {args.related}")
        metadata = load_related_file(args.related)
        print(f"Loaded metadata for {len(metadata)} files")

        # Debug: Show sample metadata keys (normalized)
        sample_keys = list(metadata.keys())[:5]
        normalized_keys = [k for k in sample_keys if not k.startswith('./')][:5]
        print(f"Sample normalized metadata keys: {normalized_keys}")
    else:
        print("No related file provided, sampling without metadata filtering")

    # Collect files from each directory separately
    print(f"Collecting files from {len(args.directory)} directories...")
    files_by_dir = collect_files_by_directory(args.directory)

    total_files = sum(len(files) for files in files_by_dir.values())
    print(f"Total files found: {total_files}")

    # Debug: Show sample file paths from each directory
    for directory, files in files_by_dir.items():
        if files:
            sample_files = files[:3]
            print(f"Sample files from {directory}: {sample_files}")

    if not any(files_by_dir.values()):
        print("Error: No files found to sample")
        return

    # Sample files from each directory
    print(f"Sampling {args.size} files from each directory with seed: {args.seed}")
    sampled_files = sample_files_by_directory(files_by_dir, args.size, metadata, args.seed)
    print(f"Total sampled files: {len(sampled_files)}")

    # Create output structure and copy files
    print(f"Creating output in: {args.output}")
    create_output_structure(sampled_files, args.output)

    # Create new related file if metadata was provided
    if args.related and metadata:
        create_related_output(sampled_files, metadata, args.output, args.related)

    print("Sampling completed successfully!")
    print(f"Sampled {args.size} files from each of {len(args.directory)} directories")


if __name__ == "__main__":
    main()
#!/bin/bash

# Set the name of the output archive
OUTPUT_ARCHIVE="archive.tar.gz"

# Create a temporary directory to organize files
TEMP_DIR=$(mktemp -d)

# Function to check if a file exists
check_file() {
    if [ ! -e "$1" ]; then
        echo "Warning: File $1 does not exist, skipping."
        return 1
    fi
    return 0
}

# Create directory structure in temp directory
mkdir -p "$TEMP_DIR/benchmark"
mkdir -p "$TEMP_DIR/benchmark/datasets"
mkdir -p "$TEMP_DIR/grpo"
mkdir -p "$TEMP_DIR/grpo/datasets"
mkdir -p "$TEMP_DIR/tsp"

# Hard-coded file list - modify these lines to include the specific files you want
# Format: check_file "path/to/file" && cp "path/to/file" "$TEMP_DIR/destination/directory/"

# Files from project directory
echo "Processing project director"
check_file "__init__.py" && cp "__init__.py" "$TEMP_DIR"
check_file ".env" && cp ".env" "$TEMP_DIR"

# Files from benchmark directory
echo "Processing benchmark files..."
check_file "benchmark/__init__.py" && cp "benchmark/__init__.py" "$TEMP_DIR/benchmark/"
check_file "benchmark/benchmark.py" && cp "benchmark/benchmark.py" "$TEMP_DIR/benchmark/"
check_file "benchmark/datasets/tsp_benchmark_dataset.json" && cp "benchmark/datasets/tsp_benchmark_dataset.json" "$TEMP_DIR/benchmark/datasets/"
check_file "benchmark/datasets/tsp_benchmark_problem_dataset.json" && cp "tsp_benchmark_problem_dataset.json" "$TEMP_DIR/benchmark/datasets/"

# Files from grpo directory
echo "Processing grpo files..."
check_file "grpo/__init__.py" && cp "grpo/__init__.py" "$TEMP_DIR/grpo/"
check_file "grpo/grpo.py" && cp "grpo/grpo.py" "$TEMP_DIR/grpo/"
check_file "grpo/grpo.ipynb" && cp "grpo/grpo.ipynb" "$TEMP_DIR/grpo/"
check_file "grpo/datasets/tsp_training_dataset.json" && cp "grpo/datasets/tsp_training_dataset.json" "$TEMP_DIR/grpo/datasets/"
check_file "grpo/datasets/tsp_training_problem_dataset.json" && cp "grpo/datasets/tsp_training_problem_dataset.json" "$TEMP_DIR/grpo/datasets/"

# Files from tsp directory
echo "Processing tsp files..."
check_file "tsp/__init__.py" && cp "tsp/__init__.py" "$TEMP_DIR/tsp/"
check_file "tsp/tsp.py" && cp "tsp/tsp.py" "$TEMP_DIR/tsp/"
check_file "tsp/tsp_llm.py" && cp "tsp/tsp_llm.py" "$TEMP_DIR/tsp/"

# Create the tar archive
echo "Creating archive $OUTPUT_ARCHIVE..."
tar -czf "$OUTPUT_ARCHIVE" -C "$TEMP_DIR" .

# Clean up the temporary directory
rm -rf "$TEMP_DIR"

echo "Archive created successfully: $OUTPUT_ARCHIVE"
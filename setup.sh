#!/bin/bash

set -e  # Exit immediately if any command fails

echo "Starting initial setup for Obsidian Research Assistant"
echo "========================================================"

# Initialize vector store
echo "Processing Obsidian vault to create FAISS index..."
python rag_obsidian.py --init

echo "✅ Setup completed successfully!"
echo "========================================================"

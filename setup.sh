#!/bin/bash
# setup.sh

echo "Pulling embedding model from Ollama..."
ollama pull nomic-embed-text

echo "Place your 'stories.json' dataset in this directory before running training."

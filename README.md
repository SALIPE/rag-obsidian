# Obsidian RAG

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A terminal-based Research Assistant using RAG (Retrieval-Augmented Generation) that integrates with your Obsidian vault and DeepSeek's AI API. Maintains conversation history as Markdown files within your knowledge base.

![Demo](https://via.placeholder.com/800x400.png?text=Terminal+Demo+Screenshot) *Replace with actual demo GIF*

## Features 

- 🔍 Query your Obsidian vault (Markdown & PDF files)
- 💬 Terminal-based chat interface
- 📚 Automatic conversation history in Obsidian
- 🧠 RAG architecture with FAISS vector store
- 🤖 DeepSeek API integration
- 🔄 Context-aware follow-up questions
- 📂 Automatic PDF text extraction
- 🗂️ Conversation version control through Markdown files

## Installation 

1. **Requirements**:
   - Python 3.8+
   - [Obsidian](https://obsidian.md) vault with research materials
   - [DeepSeek API key](https://platform.deepseek.com/api-keys)

2. **Clone repository**:
   ```bash
   git clone https://github.com/yourusername/obsidian-research-assistant.git
   cd obsidian-research-assistant

## Execution

1. **First-time setup**:
    ```bash
   python -c "from rag_obsidian import ObsidianRAG; ObsidianRAG().process_vault()"
2. **Start new chat**:
   ```bash
   python rag_obsidian.py --new
3. **Load previous conversation**:
   ```bash
   python rag_obsidian.py --load conversations/chat_20231025.md
4. **List all conversation**:
   ```bash
   python rag_obsidian.py --list

### Example Session
```bash
$ python rag_obsidian.py --new
New chat started: chat_20231025_143022.md

Chat with your Obsidian knowledge (type 'exit' to quit)

You: What's the main hypothesis in my climate paper?
Assistant: According to your paper in research/climate_study.pdf...

You: What supporting evidence do I have?
Assistant: Your annotations in notes/field_observations.md suggest...
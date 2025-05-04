#!/usr/bin/env python3
import os
import re
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables
load_dotenv()

class ObsidianRAG:
    def __init__(self, strict_mode=True):
        self.vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH"))
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.strict_mode = strict_mode
        self.exclude_dirs = {'conversations', '.obsidian', '.trash'}
        
        self.model_name = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            device_map="cpu",
            temperature=0.3,
            max_new_tokens=os.getenv("MAX_NEW_TOKENS"),
            return_full_text=False 
        )

    def should_skip(self, path):
        return any(part in self.exclude_dirs for part in path.parts)

    def process_vault(self):
        documents = []
        
        # Process research files
        for ext in ['*.md', '*.pdf']:
            for file_path in self.vault_path.glob(f'**/{ext}'):
                if self.should_skip(file_path):
                    continue
                
                try:
                    if ext == '*.md':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if "avoid_indexing: true" in f.read(200):
                                continue
                            f.seek(0)
                            text = f.read()
                    else:  # PDF
                        text = ""
                        with fitz.open(file_path) as doc:
                            for page in doc:
                                text += page.get_text()
                    
                    documents.append({
                        'text': text,
                        'source': str(file_path.relative_to(self.vault_path))
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_docs = []
        for doc in documents:
            chunks = text_splitter.split_text(doc['text'])
            for chunk in chunks:
                split_docs.append({
                    'text': chunk,
                    'source': doc['source']
                })
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts=[d['text'] for d in split_docs],
            embedding=self.embeddings,
            metadatas=[{'source': d['source']} for d in split_docs]
        )
        self.vector_store.save_local("obsidian_faiss_index")

    def query(self, question, k=3):
        if not self.vector_store:
            self.vector_store = FAISS.load_local(
                folder_path="obsidian_faiss_index", 
                embeddings=self.embeddings, 
                allow_dangerous_deserialization=True)
        
        # Retrieve documents with sources
        docs = self.vector_store.similarity_search(question, k=k)
        self.source_map = {}
                
        # Build context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata['source']
            obsidian_link = f"[[{Path(source).stem}]]"
            self.source_map[f"[{i}]"] = obsidian_link
            context_parts.append(f"Document {i} ({source}):\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Instruct: You are a research assistant. Answer this question based on your knowledgment, this conversation and based on the provided documents.

Documents:
{context}

Rules:
1. If unsure, say "Not in my sources"
2. Cite sources like [1]
3. Never invent information</s>
<|user|>
{question}</s>
<|assistant|>
"""
        
        try:
            response = self.pipe(
                prompt,
                top_p=0.95,
                repetition_penalty=1.15
            )[0]['generated_text']
            
            answer = response.split("Answer:")[-1].strip()
            
            answer = answer.split("Question:")[0].strip()
            
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        # Post-processing
        if self.strict_mode:
            answer = re.sub(r"(?i)as an ai", "Based on my sources", answer)
            if not self.validate_sources(answer):
                return "I can't verify this information in my knowledge base."
        
        # Add references if any citations exist
        if re.search(r'\[\d+\]', answer) and self.source_map:
            answer += "\n\nREFERENCES:\n" + "\n".join(
                f"{k} {v}" for k, v in self.source_map.items()
            )
        
        return answer
    
    def validate_sources(self, text):
        cited = set(re.findall(r'\[(\d+)\]', text))
        return all(f"[{n}]" in self.source_map for n in cited)

class ConversationManager:
    def __init__(self, rag):
        self.rag = rag
        self.conversation = []
        self.conversation_dir = Path(os.getenv("OBSIDIAN_VAULT_PATH")) / "conversations"
        self.current_chat = None
        self.conversation_dir.mkdir(exist_ok=True, parents=True)

    def save_chat(self):
        if not self.current_chat:
            self.current_chat = self.conversation_dir / f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
        try:
            frontmatter = "---\ntags: [chat]\navoid_indexing: true\n---\n\n"
            content = f"# Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            
            for msg in self.conversation:
                role = "User" if msg['role'] == 'user' else "Assistant"
                content += f"## {role}\n{msg['content']}\n\n"
            
            # Write to file with error handling
            with open(self.current_chat, 'w', encoding='utf-8') as f:
                f.write(frontmatter + content)
                
            print(f"\nConversation saved to: {self.current_chat}")
            
        except Exception as e:
            print(f"\nError saving conversation: {str(e)}")

    def chat_loop(self):
        print("\nResearch Assistant (type 'exit' to save & quit)\n" + "="*40)
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ('exit', 'quit'):
                    self.save_chat()
                    break
                
                # Store user input
                self.conversation.append({"role": "user", "content": user_input})
                
                try:
                    response = self.rag.query(user_input)
                    print(f"\nAssistant: {response}")
                    self.conversation.append({"role": "assistant", "content": response})
                except Exception as e:
                    print(f"\nError generating response: {str(e)}")
                    self.conversation.append({"role": "assistant", "content": f"ERROR: {str(e)}"})
                
            except KeyboardInterrupt:
                self.save_chat()
                break
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                self.save_chat()
                break


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Obsidian Research Assistant')
    parser.add_argument('--init', help='Initialize vector store', action='store_true')
    # parser.add_argument('--strict', help='Enable strict sourcing mode', action='store_true')
    args = parser.parse_args()
    
    rag = ObsidianRAG()
    if args.init:
        print("Processing vault...")
        rag.process_vault()
        print("Vector store created!")
        return
    
    ConversationManager(rag).chat_loop()

if __name__ == "__main__":
    main()

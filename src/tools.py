"""
AGENT TOOLS - Document Focused
================================
Tools that let the agent read, search, and edit files in the /docs folder.
"""

import os
from datetime import datetime
from langchain.tools import tool

DOCS_DIR = "docs"


@tool
def list_documents(query: str) -> str:
    """
    List all documents available in the docs folder.
    Use this when the user asks what documents are available.
    Input: anything (ignored)
    """
    try:
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
        if not files:
            return "No documents found in the docs folder."
        result = f"Found {len(files)} documents:\n"
        for f in files:
            size = os.path.getsize(os.path.join(DOCS_DIR, f))
            result += f"  - {f} ({size} bytes)\n"
        return result
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@tool
def read_document(filename: str) -> str:
    """
    Read the full contents of a document from the docs folder.
    Input: just the filename, e.g. 'ai_basics.txt'
    """
    try:
        safe_path = os.path.join(DOCS_DIR, os.path.basename(filename))
        if not os.path.exists(safe_path):
            files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
            return f"File '{filename}' not found. Available: {', '.join(files)}"
        with open(safe_path, "r", encoding="utf-8") as f:
            return f"Contents of {filename}:\n\n{f.read()}"
    except Exception as e:
        return f"Error reading '{filename}': {str(e)}"


@tool
def search_in_document(input: str) -> str:
    """
    Search for a word or phrase inside a document.
    Input format: 'filename.txt | search term'
    Example: 'ai_basics.txt | neural network'
    """
    try:
        if "|" not in input:
            return "Use format: 'filename.txt | search term'"
        filename, search_term = [x.strip() for x in input.split("|", 1)]
        safe_path = os.path.join(DOCS_DIR, os.path.basename(filename))
        if not os.path.exists(safe_path):
            return f"File '{filename}' not found."
        with open(safe_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        matches = [f"  Line {i}: {line.strip()}"
                   for i, line in enumerate(lines, 1)
                   if search_term.lower() in line.lower()]
        if not matches:
            return f"'{search_term}' not found in {filename}."
        return f"Found '{search_term}' in {filename} ({len(matches)} matches):\n" + "\n".join(matches[:10])
    except Exception as e:
        return f"Error searching: {str(e)}"


@tool
def append_to_document(input: str) -> str:
    """
    Append new text to the end of an existing document.
    Input format: 'filename.txt | text to append'
    Example: 'ai_basics.txt | New note: something important.'
    """
    try:
        if "|" not in input:
            return "Use format: 'filename.txt | text to add'"
        filename, new_text = [x.strip() for x in input.split("|", 1)]
        safe_path = os.path.join(DOCS_DIR, os.path.basename(filename))
        if not os.path.exists(safe_path):
            return f"File '{filename}' not found."
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(safe_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n[Added {timestamp}]\n{new_text}\n")
        return f"Successfully appended to {filename}. Re-run ingest.py to update the knowledge base."
    except Exception as e:
        return f"Error appending: {str(e)}"


@tool
def replace_text_in_document(input: str) -> str:
    """
    Find and replace text inside a document.
    Input format: 'filename.txt | old text | new text'
    Example: 'ai_basics.txt | old sentence | new sentence'
    """
    try:
        parts = input.split("|")
        if len(parts) != 3:
            return "Use format: 'filename.txt | old text | new text'"
        filename, old_text, new_text = [p.strip() for p in parts]
        safe_path = os.path.join(DOCS_DIR, os.path.basename(filename))
        if not os.path.exists(safe_path):
            return f"File '{filename}' not found."
        with open(safe_path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return f"'{old_text}' not found in {filename}."
        count = content.count(old_text)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content.replace(old_text, new_text))
        return f"Replaced {count} occurrence(s) in {filename}. Re-run ingest.py to update the knowledge base."
    except Exception as e:
        return f"Error replacing: {str(e)}"


@tool
def get_current_datetime(query: str) -> str:
    """
    Returns the current date and time.
    Input: anything (ignored)
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


ALL_TOOLS = [
    list_documents,
    read_document,
    search_in_document,
    append_to_document,
    replace_text_in_document,
    get_current_datetime,
]

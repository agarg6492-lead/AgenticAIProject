# Story Image Agent

Multimodal AI Agent built using:

- LangGraph
- Groq Qwen Vision
- LangSmith
- Sentence Transformers

## Features

✅ Reads story image

✅ Extracts

- Story Name
- Characters
- Setting
- Moral

✅ Generates image prompts

✅ Evaluates

- Semantic Similarity
- Hallucination
- Story Coverage
- Token Usage
- Latency

✅ Automatically saves approved prompts

---

## Run

```bash
pip install -r requirements.txt
```

Create `.env`

```env
GROQ_API_KEY=YOUR_KEY
LANGCHAIN_API_KEY=YOUR_LANGSMITH_KEY
```

Run:

```bash
python app.py
```

---

## Architecture

```text
Image
 ↓
Vision Analysis
 ↓
Prompt Generation
 ↓
Evaluation
 ↓
Quality Gate
 ↓
Save Prompt
```
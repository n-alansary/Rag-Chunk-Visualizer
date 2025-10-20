# Final RAG Chunk Visualizer — README

A focused README for the final RAG chunk visualizer pipeline. This project runs chunking → topic modeling (via an LLM chat client) → dimensionality reduction → 3D visualization and writes an interactive HTML output.

## Purpose
Use a chat-based LLM (Groq chat in this codebase) to perform topic assignment for text chunks, then visualize the resulting topics in 3D for inspection and analysis.

## Files and responsibilities
- `final_demo.py`  
  - Example runner that prepares sample text (or reads a PDF), splits it into chunks, and invokes the pipeline. Produces `topic_visualization_3d.html` by default.
- `llm_chat_client.py`  
  - Lightweight Groq chat wrapper (`GroqChatClient`) used to send prompts and receive responses from the LLM. Maintains conversation history and exposes `chat_with_groq(prompt, model)`.
- `base_topic_modeller.py`  
  - `TopicModeler` that coordinates the LLM-based topic modeling interaction: it seeds the assistant, streams chunks to the assistant, collects string-pair responses (expected format `(topic_number, "keyword1_keyword2")`), and parses them into topic numbers and keywords.
- `llm_topic_modelling_structured.py`  
  - High-level pipeline (`TopicModelingPipeline`) that ties the chat client, topic modeler, and visualizer together. Runs topic modeling and hands the results to the visualizer.
- `topic_visualizer.py`  
  - `TopicVisualizer` that converts chunk text into TF-IDF vectors, applies UMAP (3D by default), and renders an interactive Plotly 3D scatter with hover text and a colorbar keyed by topic number.

## Pipeline (high level)
1. Text ingestion: `final_demo.py` loads text (sample_text or PDF).
2. Chunking: `RecursiveCharacterTextSplitter` splits text into chunks.
3. Topic modeling (LLM): `base_topic_modeller.TopicModeler` sends each chunk to the LLM via `GroqChatClient`, collects tuples of `(topic_number, keywords)` per chunk.
4. Projection: `topic_visualizer` vectorizes chunks with TF-IDF, uses UMAP to project to 3D.
5. Visualization: Plotly 3D scatter is written to `topic_visualization_3d.html`.

## How to run (example)
1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies (example packages below).
3. Run final_demo.py.

The demo will:
- Build chunks from the embedded `sample_text` (or read `harry_potter_book.pdf` if you enable the PDF reading code).
- Run the topic modeling pipeline.
- Write `topic_visualization_3d.html` to the current directory.
## Dependencies & setup (recommended)
This project was developed and tested with Python 3.10. To create an isolated environment and install all required packages from a `requirements.txt`, follow the steps below (Windows / cmd examples).

1) Verify Python 3.10 is available (optional):

```cmd
py -3.10 --version
```

2) Create a Python 3.10 virtual environment in the project folder:

```cmd
py -3.10 -m venv .venv
```

3) Activate the virtual environment (Windows cmd):

```cmd
.venv\Scripts\activate
```

4) Install dependencies from `requirements.txt` (run inside the activated venv):

```cmd
pip install -r requirements.txt
```

After installing, run `python final_demo.py` from this folder with the virtual environment activated.

## Configuration and API keys
- The pipeline requires a Groq API key for `GroqChatClient`. The example `final_demo.py` currently constructs `TopicModelingPipeline(api_key='...')`. Do not leave API keys in source code.
- Recommended: store the API key in an environment variable and update `final_demo.py` to read it, e.g.:
  ```python
  import os
  api_key = os.environ.get("GROQ_API_KEY")
  pipeline = TopicModelingPipeline(api_key=api_key)
  ```
- Rotate and remove keys if they were accidentally committed.

## Output
- `topic_visualization_3d.html` — interactive 3D HTML file produced by Plotly. Open it in a browser to explore points, hover for chunk text, topic number, and topic keywords.

## Expected formats and assumptions
- The LLM is expected to return exactly one pair per chunk in the strict format: `(topic_number, "keyword1_keyword2_keyword3")`. `base_topic_modeller` will parse this. If the model deviates, parsing will fail.
- Topic numbers are integers and should map consistently across chunks.
- UMAP is used on TF-IDF vectors of chunks for projection; you can replace with other dimensionality reducers (t-SNE, PCA) if desired.




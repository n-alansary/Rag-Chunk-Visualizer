# RAG Chunk Visualizer 📊

**A compact guide to running the LLM-powered Topic Modeling and 3D Visualization pipeline.**

This project implements a pipeline to visualize text chunks based on **LLM-assigned topics** in an interactive 3D space.

## Features ✨

* **Intelligent Topic Assignment:** Uses a chat-based LLM (Groq chat) to assign a specific topic and representative keywords to each text chunk.
* **Vectorization & Projection:** Converts chunks to **TF-IDF vectors** and uses **UMAP** for non-linear dimensionality reduction into a 3D space.
* **Interactive Visualization:** Generates a stunning, interactive **Plotly 3D HTML scatter plot** (`topic_visualization_3d.html`).
* **Deep Inspection:** Hover over points in the 3D plot to view the **chunk text, assigned topic number, and keywords** for detailed inspection and analysis.



***

## 🚀 Getting Started

Follow these steps to set up the environment and run the demo.

### Prerequisites

* **Python 3.10+** (Recommended)
* A **Groq API Key** (for the LLM topic assignment).

### Environment Setup

It's highly recommended to use a virtual environment to manage dependencies.

1.  **Verify Python 3.10** (Optional):
    ```cmd
    py -3.10 --version
    ```

2.  **Create and Activate Virtual Environment:**
    ```cmd
    py -3.10 -m venv .venv
    .venv\Scripts\activate  # Windows (cmd)
    # OR
    source .venv/bin/activate # macOS/Linux
    ```

3.  **Install Dependencies:** (Ensure you have a `requirements.txt` file)
    ```cmd
    pip install -r requirements.txt
    ```

### Configuration: Set Your API Key

The pipeline requires a **Groq API Key**. **DO NOT** hardcode keys in `final_demo.py`.

**Recommended Approach (using Environment Variable):**

1.  Set the API key in your shell:
    * **Windows (cmd):** `set GROQ_API_KEY=your-api-key-here`
    * **macOS/Linux:** `export GROQ_API_KEY=your-api-key-here`

2.  Ensure `final_demo.py` is updated to read the key:
    ```python
    import os
    # ... inside final_demo.py
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    pipeline = TopicModelingPipeline(api_key=api_key, ...)
    ```

***

## 🏃 How to Run the Demo

Once your environment is active and the API key is set, execute the runner file:

```cmd
python final_demo.py
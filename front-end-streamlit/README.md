# Streamlit Frontend for RAG Assistant

This directory provides an alternative frontend built with Streamlit. It interacts with the
existing FastAPI backend (the same endpoints used by the static HTML frontend).

## Requirements

- Python 3.11+ (same environment as project)
- Add `streamlit` to your environment, e.g.:

```bash
pip install streamlit
```

- Ensure the backend is running (`uvicorn main:app --reload` or `python main.py`).

## Running

From the `front-end-streamlit` folder, execute:

```bash
streamlit run app.py
```

By default the app will assume the API is available at `http://localhost:8000`.
You can override this by creating a `.streamlit/secrets.toml` file with:

```toml
api_base = "https://your-domain.example.com"
```

## Features

- **Query interface** with adjustable retrieval/rerank parameters
- **File/URL/Text ingestion** with optional metadata
- **System information and collection statistics**
- **Health check** automatically on load

## File Layout

```
front-end-streamlit/
├── app.py            # Streamlit application code
└── README.md         # This file
```

## Notes

This frontend is designed for quick prototyping and light usage. For production,
consider deploying the Streamlit app on a server or as part of a container alongside
the FastAPI backend.
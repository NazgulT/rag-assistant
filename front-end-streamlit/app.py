import streamlit as st
import requests
import json
from pathlib import Path

# default API base URL
API_BASE = st.secrets.get("api_base", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("RAG Assistant - Streamlit Frontend")

# sidebar navigation
mode = st.sidebar.selectbox("Choose action", [
    "Query",
    "Ingest File",
    "Ingest URL",
    "Ingest Text",
    "System Info",
])


def api_post(endpoint: str, payload: dict = None, files=None):
    url = f"{API_BASE}{endpoint}"
    try:
        if files:
            resp = requests.post(url, files=files, data=payload)
        else:
            resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def api_get(endpoint: str):
    url = f"{API_BASE}{endpoint}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


if mode == "Query":
    st.header("Query the RAG System")
    with st.form(key="query_form"):
        query_text = st.text_area("Enter your question", height=150)
        k_retrieve = st.number_input("Retrieve count", min_value=1, max_value=50, value=5)
        k_rerank = st.number_input("Rerank count", min_value=1, max_value=50, value=3)
        use_reranking = st.checkbox("Use reranking", value=True)
        submitted = st.form_submit_button("Submit")
    if submitted:
        payload = {"text": query_text}
        params = {
            "k_retrieve": k_retrieve,
            "k_rerank": k_rerank,
            "use_reranking": use_reranking,
        }
        # build query string
        query_str = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        data, err = api_post(f"/api/v1/query{query_str}", payload)
        if err:
            st.error(err)
        else:
            st.json(data)

elif mode == "Ingest File":
    st.header("Ingest File")
    uploaded = st.file_uploader("Choose a file", type=["txt", "pdf", "doc", "docx", "md"])
    metadata = st.text_area("Metadata (JSON)", placeholder='{"author": "Jane"}')
    if st.button("Upload"):
        if uploaded is None:
            st.error("No file selected")
        else:
            files = {"file": (uploaded.name, uploaded.getvalue())}
            payload = {"metadata": metadata} if metadata else {}
            data, err = api_post("/api/v1/documents/ingest-file", payload, files=files)
            if err:
                st.error(err)
            else:
                st.json(data)

elif mode == "Ingest URL":
    st.header("Ingest URL")
    url_text = st.text_input("URL")
    metadata = st.text_area("Metadata (JSON)", placeholder='{"source": "web"}')
    if st.button("Ingest"):
        if not url_text:
            st.error("URL required")
        else:
            payload = {"url": url_text, "metadata": json.loads(metadata) if metadata else None}
            data, err = api_post("/api/v1/documents/ingest-url", payload)
            if err:
                st.error(err)
            else:
                st.json(data)

elif mode == "Ingest Text":
    st.header("Ingest Text")
    text_content = st.text_area("Text content")
    metadata = st.text_area("Metadata (JSON)", placeholder='{"source": "manual"}')
    if st.button("Ingest"):
        if not text_content:
            st.error("Text content required")
        else:
            payload = {"file_path": text_content, "metadata": json.loads(metadata) if metadata else None}
            data, err = api_post("/api/v1/documents/ingest-text", payload)
            if err:
                st.error(err)
            else:
                st.json(data)

elif mode == "System Info":
    st.header("System Information")
    if st.button("Load Info"):
        data, err = api_get("/api/v1/info")
        if err:
            st.error(err)
        else:
            st.json(data)
    if st.button("Load Collection Stats"):
        data, err = api_get("/api/v1/collections/stats")
        if err:
            st.error(err)
        else:
            st.json(data)

# health check on load
try:
    health, _ = api_get("/health")
    st.write("API Health:", health)
except:
    st.warning("Could not reach backend API")

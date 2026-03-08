// API base URL - adjust if needed
const API_BASE = window.location.origin.includes('localhost') ? 'http://localhost:8000' : window.location.origin;

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

// Utility functions
function showLoading(element) {
    element.innerHTML = '<div class="loading"></div> Loading...';
}

function showResult(element, data, isError = false) {
    const className = isError ? 'error' : 'success';
    element.innerHTML = `<div class="${className}"><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
}

function showError(element, message) {
    element.innerHTML = `<div class="error">${message}</div>`;
}

// API call wrapper
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        throw new Error(`API call failed: ${error.message}`);
    }
}

// Query form handler
document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const resultDiv = document.getElementById('query-result');

    showLoading(resultDiv);

    try {
        const queryData = {
            text: formData.get('query')
        };

        const params = new URLSearchParams({
            k_retrieve: formData.get('k_retrieve') || 5,
            k_rerank: formData.get('k_rerank') || 3,
            use_reranking: formData.get('use_reranking') === 'on'
        });

        const data = await apiCall(`/api/v1/query?${params}`, {
            method: 'POST',
            body: JSON.stringify(queryData)
        });

        showResult(resultDiv, data);
    } catch (error) {
        showError(resultDiv, error.message);
    }
});

// File upload handler
document.getElementById('file-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const resultDiv = document.getElementById('ingest-result');

    showLoading(resultDiv);

    try {
        const metadata = formData.get('metadata');
        if (metadata) {
            formData.set('metadata', metadata);
        }

        const response = await fetch(`${API_BASE}/api/v1/documents/ingest-file`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        showResult(resultDiv, data);
    } catch (error) {
        showError(resultDiv, error.message);
    }
});

// URL ingest handler
document.getElementById('url-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const resultDiv = document.getElementById('ingest-result');

    showLoading(resultDiv);

    try {
        const data = {
            url: formData.get('url'),
            metadata: formData.get('metadata') ? JSON.parse(formData.get('metadata')) : null
        };

        const result = await apiCall('/api/v1/documents/ingest-url', {
            method: 'POST',
            body: JSON.stringify(data)
        });

        showResult(resultDiv, result);
    } catch (error) {
        showError(resultDiv, error.message);
    }
});

// Text ingest handler
document.getElementById('text-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const resultDiv = document.getElementById('ingest-result');

    showLoading(resultDiv);

    try {
        const data = {
            file_path: formData.get('file_path'),
            metadata: formData.get('metadata') ? JSON.parse(formData.get('metadata')) : null
        };

        const result = await apiCall('/api/v1/documents/ingest-text', {
            method: 'POST',
            body: JSON.stringify(data)
        });

        showResult(resultDiv, result);
    } catch (error) {
        showError(resultDiv, error.message);
    }
});

// System info functions
async function loadSystemInfo() {
    const resultDiv = document.getElementById('info-result');
    showLoading(resultDiv);

    try {
        const data = await apiCall('/api/v1/info');
        showResult(resultDiv, data);
    } catch (error) {
        showError(resultDiv, error.message);
    }
}

async function loadCollectionStats() {
    const resultDiv = document.getElementById('info-result');
    showLoading(resultDiv);

    try {
        const data = await apiCall('/api/v1/collections/stats');
        showResult(resultDiv, data);
    } catch (error) {
        showError(resultDiv, error.message);
    }
}

// Initialize - check health on load
window.addEventListener('load', async () => {
    try {
        const health = await apiCall('/health');
        console.log('API Health:', health);
    } catch (error) {
        console.warn('API health check failed:', error.message);
        // Could show a warning to user if needed
    }
});
/**
 * Legal RAG Contract Risk Analyzer — Frontend Application
 * Handles PDF upload, document management, and chat-based queries.
 */

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const uploadZone     = document.getElementById('uploadZone');
const fileInput      = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressBar    = document.getElementById('progressBar');
const progressText   = document.getElementById('progressText');
const docList        = document.getElementById('docList');
const emptyState     = document.getElementById('emptyState');
const chatMessages   = document.getElementById('chatMessages');
const welcomeScreen  = document.getElementById('welcomeScreen');
const chatInput      = document.getElementById('chatInput');
const sendBtn        = document.getElementById('sendBtn');
const inputHint      = document.getElementById('inputHint');
const chatHeaderTitle  = document.getElementById('chatHeaderTitle');
const chatHeaderStatus = document.getElementById('chatHeaderStatus');
const toastContainer   = document.getElementById('toastContainer');
const suggestionsGrid  = document.getElementById('suggestionsGrid');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let documents = [];       // { doc_id, filename, num_chunks, text_length }
let activeDocId = null;
let isQuerying = false;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  loadDocuments();
  setupUploadZone();
  setupChatInput();
  setupSuggestions();
});

// ---------------------------------------------------------------------------
// Toast notifications
// ---------------------------------------------------------------------------
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const icons = { success: '✓', error: '✗', info: 'ℹ' };
  toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span> ${message}`;
  toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ---------------------------------------------------------------------------
// Document List
// ---------------------------------------------------------------------------
async function loadDocuments() {
  try {
    const res = await fetch('/api/documents');
    const data = await res.json();
    documents = data.documents || [];
    renderDocumentList();
  } catch (err) {
    console.error('Failed to load documents:', err);
  }
}

function renderDocumentList() {
  // Remove previous items (keep empty state)
  docList.querySelectorAll('.doc-item').forEach(el => el.remove());

  if (documents.length === 0) {
    emptyState.style.display = '';
    return;
  }
  emptyState.style.display = 'none';

  documents.forEach(doc => {
    const item = document.createElement('div');
    item.className = `doc-item${doc.doc_id === activeDocId ? ' active' : ''}`;
    item.dataset.docId = doc.doc_id;
    item.innerHTML = `
      <span class="doc-item-icon">📄</span>
      <div class="doc-item-info">
        <div class="doc-item-name" title="${doc.filename}">${doc.filename}</div>
        <div class="doc-item-meta">${doc.num_chunks} chunks · ${(doc.text_length / 1000).toFixed(1)}k chars</div>
      </div>
      <button class="doc-item-delete" title="Delete document" data-doc-id="${doc.doc_id}">🗑</button>
    `;

    // Select document
    item.addEventListener('click', (e) => {
      if (e.target.closest('.doc-item-delete')) return;
      selectDocument(doc.doc_id);
    });

    // Delete button
    item.querySelector('.doc-item-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteDocument(doc.doc_id);
    });

    docList.appendChild(item);
  });
}

function selectDocument(docId) {
  activeDocId = docId;
  const doc = documents.find(d => d.doc_id === docId);

  // Update UI
  renderDocumentList();
  chatInput.disabled = false;
  sendBtn.disabled = false;
  chatHeaderTitle.textContent = doc ? doc.filename : 'AI Risk Analyst';
  chatHeaderStatus.textContent = doc
    ? `Analyzing · ${doc.num_chunks} chunks indexed`
    : 'Ready';
  inputHint.textContent = 'Press Enter to send your query';

  // Show welcome screen only if no messages yet
  const hasMessages = chatMessages.querySelectorAll('.message').length > 0;
  if (!hasMessages) {
    welcomeScreen.style.display = '';
  }
}

async function deleteDocument(docId) {
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    documents = documents.filter(d => d.doc_id !== docId);
    if (activeDocId === docId) {
      activeDocId = null;
      chatInput.disabled = true;
      sendBtn.disabled = true;
      chatHeaderTitle.textContent = 'AI Risk Analyst';
      chatHeaderStatus.textContent = 'Ready — upload a document to begin';
      inputHint.textContent = 'Upload a document first to start querying';
    }
    renderDocumentList();
    showToast('Document deleted', 'success');
  } catch (err) {
    showToast('Failed to delete document', 'error');
  }
}

// ---------------------------------------------------------------------------
// File Upload
// ---------------------------------------------------------------------------
function setupUploadZone() {
  uploadZone.addEventListener('click', () => fileInput.click());

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFile(files[0]);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      uploadFile(fileInput.files[0]);
      fileInput.value = '';
    }
  });
}

async function uploadFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Please upload a PDF file', 'error');
    return;
  }

  // Show progress
  uploadProgress.classList.add('active');
  progressBar.style.width = '10%';
  progressText.textContent = 'Uploading document…';

  const formData = new FormData();
  formData.append('file', file);

  try {
    progressBar.style.width = '30%';
    progressText.textContent = 'Extracting text & creating chunks…';

    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });

    progressBar.style.width = '80%';
    progressText.textContent = 'Building vector index…';

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();

    progressBar.style.width = '100%';
    progressText.textContent = 'Done!';

    // Add to documents list
    documents.push({
      doc_id: data.doc_id,
      filename: data.filename,
      num_chunks: data.num_chunks,
      text_length: data.text_length,
    });

    // Auto-select the new document
    selectDocument(data.doc_id);
    renderDocumentList();

    showToast(`${data.filename} processed — ${data.num_chunks} chunks indexed`, 'success');

    setTimeout(() => {
      uploadProgress.classList.remove('active');
      progressBar.style.width = '0%';
    }, 1500);

  } catch (err) {
    progressBar.style.width = '0%';
    uploadProgress.classList.remove('active');
    showToast(err.message || 'Upload failed', 'error');
  }
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
function setupChatInput() {
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });

  sendBtn.addEventListener('click', sendQuery);
}

function setupSuggestions() {
  suggestionsGrid.querySelectorAll('.suggestion-card').forEach(card => {
    card.addEventListener('click', () => {
      if (!activeDocId) {
        showToast('Please upload and select a document first', 'info');
        return;
      }
      const query = card.dataset.query;
      chatInput.value = query;
      sendQuery();
    });
  });
}

async function sendQuery() {
  const query = chatInput.value.trim();
  if (!query || !activeDocId || isQuerying) return;

  isQuerying = true;
  chatInput.value = '';
  chatInput.disabled = true;
  sendBtn.disabled = true;

  // Hide welcome screen
  welcomeScreen.style.display = 'none';

  // Add user message
  addMessage(query, 'user');

  // Add loading indicator
  const loadingEl = addLoadingIndicator();

  try {
    const formData = new FormData();
    formData.append('doc_id', activeDocId);
    formData.append('query', query);

    const res = await fetch('/api/query', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Query failed');
    }

    const data = await res.json();

    // Remove loading
    loadingEl.remove();

    // Add AI response
    addAnalysisMessage(data);

  } catch (err) {
    loadingEl.remove();
    addMessage(`Error: ${err.message}`, 'assistant');
    showToast(err.message, 'error');
  } finally {
    isQuerying = false;
    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------
function addMessage(text, role) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const avatar = role === 'user' ? '👤' : '⚖️';

  msg.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-text">${escapeHtml(text)}</div>
    </div>
  `;

  chatMessages.appendChild(msg);
  scrollToBottom();
  return msg;
}

function addAnalysisMessage(data) {
  const msg = document.createElement('div');
  msg.className = 'message assistant';

  let clausesHtml = '';
  if (data.retrieved_clauses && data.retrieved_clauses.length > 0) {
    clausesHtml = `
      <div class="clauses-header">📎 Retrieved Clauses (${data.num_clauses})</div>
      ${data.retrieved_clauses.map(c => `
        <div class="clause-card">
          <div class="clause-header">
            <span class="clause-rank">Clause #${c.rank}</span>
            <span class="clause-distance">Relevance: ${(1 / (1 + c.distance) * 100).toFixed(1)}%</span>
          </div>
          <div class="clause-text">${escapeHtml(c.text)}</div>
        </div>
      `).join('')}
    `;
  }

  msg.innerHTML = `
    <div class="message-avatar">⚖️</div>
    <div class="message-content">
      <div class="analysis-section">
        <div class="analysis-header">🔍 Risk Analysis</div>
        <div class="analysis-response">${escapeHtml(data.analysis)}</div>
        ${clausesHtml}
      </div>
    </div>
  `;

  chatMessages.appendChild(msg);
  scrollToBottom();
}

function addLoadingIndicator() {
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant';
  wrapper.innerHTML = `
    <div class="message-avatar">⚖️</div>
    <div class="typing-indicator">
      <div class="typing-dots">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  chatMessages.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// DOM Elements
const videoStream = document.getElementById('videoStream');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const targetsList = document.getElementById('targetsList');
const resolutionSelect = document.getElementById('resolutionSelect');
const applyResolutionBtn = document.getElementById('applyResolutionBtn');
const resolutionStatus = document.getElementById('resolutionStatus');

// Video stream
let streamRetryTimer = null;

function startVideoStream() {
    const connect = () => {
        if (streamRetryTimer) {
            clearTimeout(streamRetryTimer);
            streamRetryTimer = null;
        }

        // Cache-bust reconnects so browser always re-establishes stream cleanly.
        videoStream.src = `/video_feed?t=${Date.now()}`;
    };

    videoStream.onerror = () => {
        streamRetryTimer = setTimeout(connect, 1000);
    };

    connect();
}

startVideoStream();

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#764ba2';
    uploadArea.style.background = 'rgba(102, 126, 234, 0.3)';
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#667eea';
    uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#667eea';
    uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Functions
async function handleFileUpload(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showStatus('Only image files are supported', 'error');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showStatus('File is too large (max 10MB)', 'error');
        return;
    }

    showStatus('Processing image...', 'loading');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload_target', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const result = await response.json();
        showStatus(`✓ ${result.filename} loaded!`, 'success');
        fileInput.value = '';
        
        // Refresh targets list
        await refreshTargets();
    } catch (error) {
        showStatus(`✗ ${error.message}`, 'error');
        console.error('Upload error:', error);
    }
}

async function refreshTargets() {
    try {
        const response = await fetch('/targets');
        const data = await response.json();
        
        renderTargets(data.targets, data.current);
    } catch (error) {
        console.error('Error fetching targets:', error);
    }
}

async function refreshResolution() {
    try {
        const response = await fetch('/camera_resolution');
        if (!response.ok) {
            throw new Error('Failed to read camera resolution');
        }

        const data = await response.json();
        if (Array.isArray(data.available) && data.available.length) {
            resolutionSelect.innerHTML = data.available
                .map(preset => `<option value="${preset}">${preset}</option>`)
                .join('');
        }

        if (data.current) {
            resolutionSelect.value = data.current;
        }
    } catch (error) {
        console.error('Resolution fetch error:', error);
    }
}

function showResolutionStatus(message, type) {
    resolutionStatus.textContent = message;
    resolutionStatus.className = `status-message show ${type}`;

    if (type !== 'loading') {
        setTimeout(() => {
            resolutionStatus.classList.remove('show');
        }, 3000);
    }
}

async function applyResolution() {
    const preset = resolutionSelect.value;
    if (!preset) return;

    applyResolutionBtn.disabled = true;
    showResolutionStatus(`Switching to ${preset}...`, 'loading');

    try {
        const response = await fetch(`/camera_resolution/${encodeURIComponent(preset)}`, {
            method: 'POST'
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Resolution switch failed');
        }

        resolutionSelect.value = result.current;
        showResolutionStatus(`✓ Camera set to ${result.current} (${result.actual_width}x${result.actual_height})`, 'success');
    } catch (error) {
        showResolutionStatus(`✗ ${error.message}`, 'error');
        console.error('Resolution apply error:', error);
    } finally {
        applyResolutionBtn.disabled = false;
    }
}

function renderTargets(targets, currentTarget) {
    if (targets.length === 0) {
        targetsList.innerHTML = '<p class="empty">No targets loaded yet</p>';
        return;
    }

    targetsList.innerHTML = targets.map(filename => `
        <div class="target-item ${filename === currentTarget ? 'active' : ''}" data-target="${filename}">
            <span class="target-name">📷 ${filename}</span>
            <div class="target-actions">
                <button class="btn-small btn-select" onclick="selectTarget('${filename}')">
                    ${filename === currentTarget ? '✓ Active' : 'Select'}
                </button>
                <button class="btn-small btn-delete" onclick="deleteTarget('${filename}')">Delete</button>
            </div>
        </div>
    `).join('');
}

async function selectTarget(filename) {
    try {
        const response = await fetch(`/set_target/${filename}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Failed to select target');
        }

        showStatus(`✓ Swapping to ${filename}`, 'success');
        await refreshTargets();
    } catch (error) {
        showStatus(`✗ ${error.message}`, 'error');
        console.error('Select error:', error);
    }
}

async function deleteTarget(filename) {
    if (!confirm(`Delete "${filename}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/target/${filename}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to delete target');
        }

        showStatus(`✓ ${filename} deleted`, 'success');
        await refreshTargets();
    } catch (error) {
        showStatus(`✗ ${error.message}`, 'error');
        console.error('Delete error:', error);
    }
}

function showStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-message show ${type}`;

    if (type !== 'loading') {
        setTimeout(() => {
            uploadStatus.classList.remove('show');
        }, 3000);
    }
}

applyResolutionBtn.addEventListener('click', () => {
    applyResolution();
});

// Load targets on page load
document.addEventListener('DOMContentLoaded', () => {
    refreshTargets();
    refreshResolution();
    
    // Refresh targets every 5 seconds
    setInterval(refreshTargets, 5000);

    console.log('✓ Face Swap Stream interface loaded');
    console.log('📸 Connect your webcam and upload a target face to begin!');
});

// DOM Elements
const canvas = document.getElementById('videoCanvas');
const ctx = canvas.getContext('2d');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const targetsList = document.getElementById('targetsList');

// Video stream
let streamImage = new Image();
let isStreaming = false;
let isFetching = false;

function startVideoStream() {
    isStreaming = true;
    streamImage.crossOrigin = "anonymous";
    
    async function fetchFrame() {
        if (!isStreaming || isFetching) return;
        
        isFetching = true;
        
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 5000);  // 5 second timeout
            
            const response = await fetch('/frame?t=' + Date.now(), { signal: controller.signal });
            clearTimeout(timeout);
            
            if (!response.ok) throw new Error('Frame fetch failed');
            const blob = await response.blob();
            
            const url = URL.createObjectURL(blob);
            streamImage.onload = () => {
                URL.revokeObjectURL(url);
                ctx.drawImage(streamImage, 0, 0);
            };
            streamImage.src = url;
        } catch (err) {
            if (err.name !== 'AbortError') {
                console.error('Stream error:', err);
            }
        }
        
        isFetching = false;
    }
    
    // Fixed 10 FPS interval for stable streaming
    setInterval(fetchFrame, 100);  // ~10 FPS
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

// Load targets on page load
document.addEventListener('DOMContentLoaded', () => {
    refreshTargets();
    
    // Refresh targets every 5 seconds
    setInterval(refreshTargets, 5000);

    console.log('✓ Face Swap Stream interface loaded');
    console.log('📸 Connect your webcam and upload a target face to begin!');
});

// DOM Elements
const videoStream = document.getElementById('videoStream');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const targetsList = document.getElementById('targetsList');
const resolutionSelect = document.getElementById('resolutionSelect');
const applyResolutionBtn = document.getElementById('applyResolutionBtn');
const resolutionStatus = document.getElementById('resolutionStatus');
const voicePitchInput = document.getElementById('voicePitch');
const voiceIndexRateInput = document.getElementById('voiceIndexRate');
const voiceBlockSecondsInput = document.getElementById('voiceBlockSeconds');
const voiceF0MethodSelect = document.getElementById('voiceF0Method');
const voiceInputDeviceSelect = document.getElementById('voiceInputDevice');
const voiceOutputDeviceSelect = document.getElementById('voiceOutputDevice');
const voiceGateThresholdInput = document.getElementById('voiceGateThreshold');
const voiceGateHoldMsInput = document.getElementById('voiceGateHoldMs');
const voiceGateReleaseMsInput = document.getElementById('voiceGateReleaseMs');
const voiceNoiseGateEnabledCheckbox = document.getElementById('voiceNoiseGateEnabled');
const voicePresetSelect = document.getElementById('voicePreset');
const voiceUseVirtualOutputCheckbox = document.getElementById('voiceUseVirtualOutput');
const voiceRefreshDevicesBtn = document.getElementById('voiceRefreshDevicesBtn');
const voiceStartBtn = document.getElementById('voiceStartBtn');
const voiceStopBtn = document.getElementById('voiceStopBtn');
const voiceStatus = document.getElementById('voiceStatus');

let recommendedVirtualOutputIndex = null;

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

function showVoiceStatus(message, type) {
    voiceStatus.textContent = message;
    voiceStatus.className = `status-message show ${type}`;
}

async function refreshVoiceStatus() {
    try {
        const response = await fetch('/voice/status');
        if (!response.ok) {
            throw new Error('Failed to read voice status');
        }

        const data = await response.json();
        if (data.running) {
            const outputLabel = data.output_device_name || 'system default';
            const routeLabel = data.using_virtual_output ? 'virtual' : 'regular';
            showVoiceStatus(`✓ Running on ${data.device} | out ${outputLabel} (${routeLabel}) | pitch ${data.pitch}`, 'success');
        } else {
            showVoiceStatus('Voice changer is stopped', 'loading');
        }
    } catch (error) {
        showVoiceStatus(`✗ ${error.message}`, 'error');
        console.error('Voice status error:', error);
    }
}

function buildDeviceOptions(selectEl, devices, includeAuto = true, virtualBadge = false) {
    const options = [];
    if (includeAuto) {
        options.push('<option value="auto">Auto (default)</option>');
    }

    for (const device of devices) {
        const tags = [];
        if (device.is_default) tags.push('default');
        if (virtualBadge && device.is_virtual) tags.push('virtual');
        const suffix = tags.length ? ` [${tags.join(', ')}]` : '';
        options.push(`<option value="${device.index}">${device.name}${suffix}</option>`);
    }

    selectEl.innerHTML = options.join('');
}

async function refreshVoiceDevices() {
    try {
        const response = await fetch('/voice/devices');
        if (!response.ok) {
            throw new Error('Failed to fetch audio devices');
        }

        const data = await response.json();
        const inputs = Array.isArray(data.inputs) ? data.inputs : [];
        const outputs = Array.isArray(data.outputs) ? data.outputs : [];
        const recommended = data.recommended_virtual_output;

        recommendedVirtualOutputIndex = recommended ? String(recommended.index) : null;

        buildDeviceOptions(voiceInputDeviceSelect, inputs, true, false);
        buildDeviceOptions(voiceOutputDeviceSelect, outputs, true, true);

        if (voiceUseVirtualOutputCheckbox.checked && recommendedVirtualOutputIndex !== null) {
            voiceOutputDeviceSelect.value = recommendedVirtualOutputIndex;
        }
    } catch (error) {
        console.error('Voice devices error:', error);
        showVoiceStatus(`✗ ${error.message}`, 'error');
    }
}

async function startVoice() {
    voiceStartBtn.disabled = true;
    showVoiceStatus('Starting voice changer...', 'loading');

    const preset = voicePresetSelect?.value || 'speech';
    const presetConfig = {
        speech: { index_rate: 0.40, protect: 0.33, block_seconds: 0.50, dry_mix: 0.0 },
        balanced: { index_rate: 0.35, protect: 0.40, block_seconds: 0.50, dry_mix: 0.0 },
        character: { index_rate: 0.50, protect: 0.33, block_seconds: 0.60, dry_mix: 0.0 },
    }[preset] || { index_rate: 0.40, protect: 0.33, block_seconds: 0.50, dry_mix: 0.0 };

    const payload = {
        pitch: Number(voicePitchInput.value || 0),
        index_rate: Number(voiceIndexRateInput.value || presetConfig.index_rate),
        block_seconds: Number(voiceBlockSecondsInput.value || presetConfig.block_seconds),
        f0_method: voiceF0MethodSelect.value || 'harvest',
        noise_gate_enabled: voiceNoiseGateEnabledCheckbox ? voiceNoiseGateEnabledCheckbox.checked : true,
        noise_gate_threshold: Number(voiceGateThresholdInput?.value || 0.025),
        noise_gate_hold_ms: Number(voiceGateHoldMsInput?.value || 140),
        noise_gate_release_ms: Number(voiceGateReleaseMsInput?.value || 70),
        protect: presetConfig.protect,
        dry_mix: presetConfig.dry_mix,
        input_device: voiceInputDeviceSelect.value === 'auto' ? null : voiceInputDeviceSelect.value,
        output_device: voiceOutputDeviceSelect.value === 'auto' ? null : voiceOutputDeviceSelect.value,
        use_virtual_output: !!voiceUseVirtualOutputCheckbox.checked
    };

    try {
        const response = await fetch('/voice/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to start voice changer');
        }

        showVoiceStatus(`✓ Voice changer running (${result.voice.device})`, 'success');
    } catch (error) {
        showVoiceStatus(`✗ ${error.message}`, 'error');
        console.error('Voice start error:', error);
    } finally {
        voiceStartBtn.disabled = false;
    }
}

async function stopVoice() {
    voiceStopBtn.disabled = true;
    showVoiceStatus('Stopping voice changer...', 'loading');

    try {
        const response = await fetch('/voice/stop', {
            method: 'POST'
        });
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to stop voice changer');
        }

        showVoiceStatus('Voice changer stopped', 'loading');
    } catch (error) {
        showVoiceStatus(`✗ ${error.message}`, 'error');
        console.error('Voice stop error:', error);
    } finally {
        voiceStopBtn.disabled = false;
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

voiceStartBtn.addEventListener('click', () => {
    startVoice();
});

voiceStopBtn.addEventListener('click', () => {
    stopVoice();
});

voiceRefreshDevicesBtn.addEventListener('click', () => {
    refreshVoiceDevices();
});

voiceUseVirtualOutputCheckbox.addEventListener('change', () => {
    if (voiceUseVirtualOutputCheckbox.checked && recommendedVirtualOutputIndex !== null) {
        voiceOutputDeviceSelect.value = recommendedVirtualOutputIndex;
    }
});

// Load targets on page load
document.addEventListener('DOMContentLoaded', () => {
    refreshTargets();
    refreshResolution();
    refreshVoiceDevices();
    refreshVoiceStatus();
    
    // Refresh targets every 5 seconds
    setInterval(refreshTargets, 5000);
    setInterval(refreshVoiceStatus, 5000);

    console.log('✓ Face Swap Stream interface loaded');
    console.log('📸 Connect your webcam and upload a target face to begin!');
});

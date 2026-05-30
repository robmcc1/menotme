const voicePitchInput = document.getElementById('voicePitch');
const voiceIndexRateInput = document.getElementById('voiceIndexRate');
const voiceBlockSecondsInput = document.getElementById('voiceBlockSeconds');
const voiceF0MethodSelect = document.getElementById('voiceF0Method');
const voiceInputDeviceSelect = document.getElementById('voiceInputDevice');
const voiceOutputDeviceSelect = document.getElementById('voiceOutputDevice');
const voiceGateThresholdInput = document.getElementById('voiceGateThreshold');
const voiceGateHoldMsInput = document.getElementById('voiceGateHoldMs');
const voiceGateReleaseMsInput = document.getElementById('voiceGateReleaseMs');
const voicePresetSelect = document.getElementById('voicePreset');
const voiceUseVirtualOutputCheckbox = document.getElementById('voiceUseVirtualOutput');
const voiceRefreshDevicesBtn = document.getElementById('voiceRefreshDevicesBtn');
const voiceStartBtn = document.getElementById('voiceStartBtn');
const voiceStopBtn = document.getElementById('voiceStopBtn');
const voiceStatus = document.getElementById('voiceStatus');

let recommendedVirtualOutputIndex = null;

function showVoiceStatus(message) {
    voiceStatus.textContent = message;
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
        showVoiceStatus(`Device error: ${error.message}`);
    }
}

async function refreshVoiceStatus() {
    try {
        const response = await fetch('/voice/status');
        if (!response.ok) {
            throw new Error('Failed to read voice status');
        }

        const data = await response.json();
        if (data.running) {
            const outName = data.output_device_name || 'default output';
            const backend = data.backend || 'unknown';
            showVoiceStatus(`Running (${backend}) -> ${outName}`);
        } else {
            showVoiceStatus('Voice changer is stopped');
        }
    } catch (error) {
        showVoiceStatus(`Status error: ${error.message}`);
    }
}

async function startVoice() {
    voiceStartBtn.disabled = true;
    showVoiceStatus('Starting voice changer...');

    const preset = voicePresetSelect?.value || 'speech';
    const presetConfig = {
        speech: { index_rate: 0.06, protect: 0.80, block_seconds: 0.32, dry_mix: 0.42 },
        balanced: { index_rate: 0.18, protect: 0.62, block_seconds: 0.28, dry_mix: 0.24 },
        character: { index_rate: 0.35, protect: 0.42, block_seconds: 0.24, dry_mix: 0.12 },
    }[preset] || { index_rate: 0.06, protect: 0.80, block_seconds: 0.32, dry_mix: 0.42 };

    const payload = {
        backend: 'rvc',
        pitch: Number(voicePitchInput.value || 0),
        index_rate: Number(voiceIndexRateInput.value || presetConfig.index_rate),
        block_seconds: Number(voiceBlockSecondsInput.value || presetConfig.block_seconds),
        f0_method: voiceF0MethodSelect.value || 'harvest',
        noise_gate_enabled: false,
        noise_gate_threshold: Number(voiceGateThresholdInput?.value || 0.008),
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

        showVoiceStatus(`Voice changer running (${result.voice.backend || 'unknown'})`);
    } catch (error) {
        showVoiceStatus(`Start error: ${error.message}`);
    } finally {
        voiceStartBtn.disabled = false;
    }
}

async function stopVoice() {
    voiceStopBtn.disabled = true;
    showVoiceStatus('Stopping voice changer...');

    try {
        const response = await fetch('/voice/stop', { method: 'POST' });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to stop voice changer');
        }

        showVoiceStatus('Voice changer stopped');
    } catch (error) {
        showVoiceStatus(`Stop error: ${error.message}`);
    } finally {
        voiceStopBtn.disabled = false;
    }
}

voiceStartBtn.addEventListener('click', startVoice);
voiceStopBtn.addEventListener('click', stopVoice);
voiceRefreshDevicesBtn.addEventListener('click', refreshVoiceDevices);
voiceUseVirtualOutputCheckbox.addEventListener('change', () => {
    if (voiceUseVirtualOutputCheckbox.checked && recommendedVirtualOutputIndex !== null) {
        voiceOutputDeviceSelect.value = recommendedVirtualOutputIndex;
    }
});

document.addEventListener('DOMContentLoaded', () => {
    refreshVoiceDevices();
    refreshVoiceStatus();
    setInterval(refreshVoiceStatus, 4000);
});

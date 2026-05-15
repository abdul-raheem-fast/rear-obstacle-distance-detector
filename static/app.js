// Reverse Parking Assistant - Mobile GUI JavaScript
// Polls /api/status and updates the UI in real-time

const POLL_INTERVAL = 250;          // ms between API polls
const MAX_GAUGE_DISTANCE = 5.0;     // metres (full-scale)
const GAUGE_ARC_LENGTH = 267;       // 270° of a circle with r=85 ≈ 267 px
let isFullscreen = false;
let pollTimer = null;
let connectionOk = true;
let failCount = 0;
const isLikelyMobile = /Android|iPhone|iPad|iPod|Mobile|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Configuration
const MAX_VISIBLE_SLOTS = 4;
const RESERVED_SLOTS = 4;
const MOBILE_PROCESS_INTERVAL_MS = 500;  // Process every 500ms on mobile
let lastProcessTime = 0;

// Object-class icons
const CLASS_ICONS = {
    person: '🧍', car: '🚗', bicycle: '🚲', motorcycle: '🏍️',
    bus: '🚌', truck: '🚛', dog: '🐕', cat: '🐈',
    chair: '🪑', obstacle: '🔶', default: '📦'
};

// Camera Stream State
const video = document.getElementById('cameraVideo');
const canvas = document.getElementById('cameraCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const videoFeed = document.getElementById('videoFeed');

let useDeviceCamera = true;
let isVideoLoopRunning = false;
let currentStream = null;
let processLoopId = 0;
let isSwitchingSource = false;
const MAX_UPLOAD_WIDTH = 640;
const MAX_UPLOAD_HEIGHT = 480;
const CAMERA_OPEN_TIMEOUT_MS = 10000;
const PROCESS_REQUEST_TIMEOUT_MS = 2500;

// Web Audio (mobile + desktop browser beeps)
let audioContext = null;
let audioUnlocked = false;
const AUDIO_COOLDOWNS_MS = {
    danger: 250,
    caution: 900,
    warning: 1400,
    safe: 999999
};
const lastAudioAt = {
    danger: 0,
    caution: 0,
    warning: 0,
    safe: 0
};

// Initialise
let currentServerMode = null;

document.addEventListener('DOMContentLoaded', async () => {
    bindAudioUnlock();
    // Fetch initial status to get the server's mode
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        applyServerMode(data.camera_mode || 'laptop', false);
    } catch {
        applyServerMode('laptop', false);
    }
    startPolling();
});

function applyServerMode(mode, fromUser) {
    if (currentServerMode === mode) return;
    currentServerMode = mode;
    
    // CRITICAL: Stop both camera sources before switching
    stopDeviceCamera();
    stopServerFeed();

    const stateLabel = document.getElementById('cameraSourceState');
    if (mode === 'mobile') {
        stateLabel.textContent = 'PHONE';
        if (isLikelyMobile) {
            // Mobile device - use phone camera
            useDeviceCamera = true;
            setCameraDisplayMode('device');
            startDeviceCamera(fromUser);
        } else {
            // Desktop viewing mobile mode - show message
            useDeviceCamera = false;
            setCameraDisplayMode('message');
            showCameraMessage('Using Phone Camera', 'Camera is active on mobile device');
            startPolling(); // Poll for detections from mobile
        }
    } else {
        // Laptop mode
        stateLabel.textContent = 'LAPTOP';
        if (isLikelyMobile) {
            // Mobile viewing laptop mode - show message
            useDeviceCamera = false;
            setCameraDisplayMode('message');
            showCameraMessage('Using Laptop Camera', 'Camera is active on server/laptop');
            startPolling(); // Poll for detections from laptop
        } else {
            // Desktop viewing laptop mode - show video feed
            useDeviceCamera = false;
            setCameraDisplayMode('server');
            startServerFeed();
        }
    }
    
    console.log(`[MODE] Switched to ${mode} mode, useDeviceCamera=${useDeviceCamera}`);
}

async function toggleCameraSource() {
    if (isSwitchingSource) return;
    isSwitchingSource = true;
    
    const btn = document.getElementById('btnCameraSource');
    const label = document.getElementById('cameraSourceState');
    if (btn) btn.disabled = true;
    if (label) label.textContent = 'SWITCHING';

    const newMode = (currentServerMode === 'laptop') ? 'mobile' : 'laptop';
    
    try {
        const response = await fetch('/api/switch_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: newMode })
        });
        
        const result = await response.json();
        if (result.success || result.camera_mode) {
            applyServerMode(result.camera_mode || newMode, true);
        }
    } catch (error) {
        console.error('Error switching camera:', error);
        if (label) label.textContent = currentServerMode === 'mobile' ? 'PHONE' : 'LAPTOP';
    } finally {
        isSwitchingSource = false;
        if (btn) btn.disabled = false;
    }
}
function bindAudioUnlock() {
    const unlock = () => unlockAudioContext();
    document.body.addEventListener('click', unlock, { once: true });
    document.body.addEventListener('touchstart', unlock, { once: true, passive: true });
}

function unlockAudioContext() {
    if (audioUnlocked) return;
    if (!audioContext) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        audioContext = new AudioCtx();
    }
    audioContext.resume().then(() => {
        audioUnlocked = true;
    }).catch(() => {
        audioUnlocked = false;
    });
}

function playBeep(freq, durationMs, volume) {
    if (!audioContext || !audioUnlocked) return;
    const now = audioContext.currentTime;
    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();

    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.value = 0;

    osc.connect(gain);
    gain.connect(audioContext.destination);

    const attack = 0.01;
    const release = 0.08;
    gain.gain.setValueAtTime(0.0, now);
    gain.gain.linearRampToValueAtTime(volume, now + attack);
    gain.gain.linearRampToValueAtTime(0.0, now + (durationMs / 1000) - release);

    osc.start(now);
    osc.stop(now + (durationMs / 1000));
}

function updateAudio(d) {
    if (!d || d.muted) return;
    if (!audioUnlocked) return;

    const level = d.warning_level || 'safe';
    const now = Date.now();
    const cooldown = AUDIO_COOLDOWNS_MS[level] || 1000;
    if (now - lastAudioAt[level] < cooldown) return;

    lastAudioAt[level] = now;

    if (level === 'danger') {
        playBeep(1000, 160, 0.35);
    } else if (level === 'caution') {
        playBeep(760, 220, 0.28);
    } else if (level === 'warning') {
        playBeep(560, 240, 0.22);
    }
}

// Removes the duplicate leftover functions

async function startDeviceCamera(fromUserAction) {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera API not supported in this browser.');
        }
        if (!window.isSecureContext) {
            throw new Error('Camera requires HTTPS.');
        }
        const constraintsToTry = [
            { video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } },
            { video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } } },
            { video: true }
        ];

        let lastError = null;
        for (const constraints of constraintsToTry) {
            try {
                currentStream = await openCameraWithTimeout(constraints, CAMERA_OPEN_TIMEOUT_MS);
                break;
            } catch (err) {
                lastError = err;
            }
        }
        if (!currentStream) {
            throw lastError || new Error('Unable to open camera stream.');
        }

        video.muted = true;
        video.srcObject = currentStream;
        video.onloadedmetadata = () => {
            video.play().catch(() => {
                if (fromUserAction) {
                    alert('Tap the page to allow camera playback.');
                }
            });
            const sourceW = video.videoWidth || 640;
            const sourceH = video.videoHeight || 480;
            const scale = Math.min(MAX_UPLOAD_WIDTH / sourceW, MAX_UPLOAD_HEIGHT / sourceH, 1);
            canvas.width = Math.max(1, Math.round(sourceW * scale));
            canvas.height = Math.max(1, Math.round(sourceH * scale));
            if (!isVideoLoopRunning) {
                isVideoLoopRunning = true;
                processLoopId += 1;
                processLoop(processLoopId);
            }
        };
    } catch (err) {
        const errName = err && err.name ? err.name : 'Error';
        const errMsg = err && err.message ? err.message : String(err);
        const details = `${errName}: ${errMsg}`;

        if (errName === 'NotReadableError') {
            // Usually means camera is busy (another app/tab/server capture has locked it).
            if (fromUserAction) {
                alert(
                    "Phone camera is busy and could not start.\n\n" +
                    "Close other camera apps and try again."
                );
            }
            return;
        }

        alert(
            "Camera error: " + details +
            "\n\nAllow camera permission and keep this tab on HTTPS (or localhost)."
        );
        setConnected(false);
    }
}

function stopDeviceCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    video.srcObject = null;
    video.pause();
    isVideoLoopRunning = false;
    processLoopId += 1;
}

function setCameraDisplayMode(mode) {
    if (mode === 'device') {
        video.style.display = 'none';  // Hide raw video
        videoFeed.style.display = 'block';  // Show processed image
        document.getElementById('cameraMessage')?.remove();
    } else if (mode === 'server') {
        video.style.display = 'none';
        videoFeed.style.display = 'block';
        document.getElementById('cameraMessage')?.remove();
    } else if (mode === 'message') {
        video.style.display = 'none';
        videoFeed.style.display = 'none';
        // Message will be added by showCameraMessage()
    }
}

function showCameraMessage(title, subtitle) {
    // Remove existing message
    document.getElementById('cameraMessage')?.remove();
    
    // Create message overlay
    const messageDiv = document.createElement('div');
    messageDiv.id = 'cameraMessage';
    messageDiv.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #1a1f30 0%, #0a0e1a 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        text-align: center;
        padding: 20px;
        z-index: 10;
    `;
    
    messageDiv.innerHTML = `
        <div style="font-size: 3rem; margin-bottom: 20px;">📷</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #f1f5f9; margin-bottom: 10px;">${title}</div>
        <div style="font-size: 0.9rem; color: #64748b;">${subtitle}</div>
    `;
    
    document.querySelector('.video-container').appendChild(messageDiv);
}

async function openCameraWithTimeout(constraints, timeoutMs) {
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Camera open timed out.')), timeoutMs);
    });
    return Promise.race([
        navigator.mediaDevices.getUserMedia(constraints),
        timeoutPromise
    ]);
}

// Process Loop (Device Camera)
async function processLoop(loopId) {
    // CRITICAL: Only process if using device camera AND in mobile mode
    if (!useDeviceCamera || !isVideoLoopRunning || loopId !== processLoopId) return;
    if (currentServerMode !== 'mobile') {
        console.log('[INFO] Stopping device camera (not in mobile mode)');
        stopDeviceCamera();
        return;
    }
    
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(() => processLoop(loopId));
        return;
    }
    
    // Throttle processing on mobile
    const now = Date.now();
    if (now - lastProcessTime < MOBILE_PROCESS_INTERVAL_MS) {
        requestAnimationFrame(() => processLoop(loopId));
        return;
    }
    lastProcessTime = now;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg', 0.5);  // Lower quality for speed
    
    try {
        const res = await fetchWithTimeout('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        }, PROCESS_REQUEST_TIMEOUT_MS);
        
        if (!res.ok) {
            if (res.status === 400) {
                console.log('[INFO] Server rejected mobile frame (not in mobile mode)');
                stopDeviceCamera();
                return;
            }
            throw new Error(res.status);
        }
        const data = await res.json();
        
        if (loopId === processLoopId && useDeviceCamera) {
            if (data.image) {
                // Prevent flicker by only updating if source is actually different
                if (videoFeed.src !== data.image) {
                    videoFeed.src = data.image;
                }
            }
            updateUI(data.stats);
            failCount = 0;
            if (!connectionOk) setConnected(true);
        }
    } catch (e) {
        if (loopId === processLoopId) {
            failCount++;
            if (failCount > 3 && connectionOk) setConnected(false);
        }
    }
    
    if (loopId === processLoopId) {
        requestAnimationFrame(() => processLoop(loopId));
    }
}

async function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timeout);
    }
}

// Polling Loop (Server Camera)
function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(fetchStatus, POLL_INTERVAL);
}

function startServerFeed() {
    // Only start video feed if in laptop mode
    if (currentServerMode === 'laptop') {
        videoFeed.src = "/video_feed?" + new Date().getTime();
    } else {
        // Show message that laptop camera is active
        videoFeed.src = "";
        videoFeed.alt = "Using Laptop Camera on Server";
    }
    startPolling();
}

function stopServerFeed() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
    videoFeed.src = "";
}

async function fetchStatus() {
    // CRITICAL: Only poll status when NOT using device camera
    if (useDeviceCamera) return;
    
    try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();
        failCount = 0;
        if (!connectionOk) setConnected(true);
        updateUI(data);
        
        if (data.camera_mode && data.camera_mode !== currentServerMode) {
             applyServerMode(data.camera_mode, false);
        }
    } catch (e) {
        failCount++;
        if (failCount > 8 && connectionOk) setConnected(false);
    }
}

function setConnected(ok) {
    connectionOk = ok;
    const badge = document.getElementById('connectionBadge');
    if (ok) {
        badge.classList.remove('disconnected');
        badge.querySelector('span:last-child').textContent = 'LIVE';
    } else {
        badge.classList.add('disconnected');
        badge.querySelector('span:last-child').textContent = 'OFFLINE';
    }
}

// Master UI update
function updateUI(d) {
    updateWarningBanner(d);
    updateGauge(d);
    updateDetections(d);
    updateOverlays(d);
    updateControls(d);
    updateSystem(d);
    updateAudio(d);
}

// Warning Banner
function updateWarningBanner(d) {
    const banner = document.getElementById('warningBanner');
    const title = document.getElementById('warningTitle');
    const subtitle = document.getElementById('warningSubtitle');
    const dist = document.getElementById('warningDistance');

    if (d.warning_level === 'danger') {
        banner.className = 'warning-banner danger';
        title.textContent = '🛑 STOP — TOO CLOSE';
        subtitle.textContent = `${d.closest_class || 'Object'} dangerously close`;
        dist.textContent = d.min_distance != null ? d.min_distance.toFixed(2) + 'm' : '--';
    } else if (d.warning_level === 'caution') {
        banner.className = 'warning-banner caution';
        title.textContent = '⚠️ CAUTION';
        subtitle.textContent = `${d.closest_class || 'Object'} approaching`;
        dist.textContent = d.min_distance != null ? d.min_distance.toFixed(2) + 'm' : '--';
    } else if (d.warning_level === 'warning') {
        banner.className = 'warning-banner caution warning-level';
        title.textContent = 'SLOW DOWN';
        subtitle.textContent = `${d.closest_class || 'Object'} in range`;
        dist.textContent = d.min_distance != null ? d.min_distance.toFixed(2) + 'm' : '--';
    } else {
        banner.className = 'warning-banner hidden';
    }
}

// Gauge
function updateGauge(d) {
    const arc = document.getElementById('gaugeArc');
    const number = document.getElementById('gaugeNumber');
    const cls = document.getElementById('closestClass');

    if (d.min_distance == null) {
        arc.style.strokeDashoffset = GAUGE_ARC_LENGTH;
        number.textContent = '--';
        number.style.color = 'var(--text-muted)';
        cls.textContent = 'No objects';
        arc.setAttribute('stroke', 'url(#gaugeGradientSafe)');
        return;
    }

    const clamped = Math.min(d.min_distance, MAX_GAUGE_DISTANCE);
    const ratio = clamped / MAX_GAUGE_DISTANCE;
    const offset = GAUGE_ARC_LENGTH - (ratio * GAUGE_ARC_LENGTH);

    arc.style.strokeDashoffset = offset;
    number.textContent = d.min_distance.toFixed(2);

    if (d.warning_level === 'danger') {
        arc.setAttribute('stroke', 'url(#gaugeGradientDanger)');
        number.style.color = 'var(--danger)';
    } else if (d.warning_level === 'caution' || d.warning_level === 'warning') {
        arc.setAttribute('stroke', 'url(#gaugeGradientCaution)');
        number.style.color = 'var(--caution)';
    } else {
        arc.setAttribute('stroke', 'url(#gaugeGradientSafe)');
        number.style.color = 'var(--safe)';
    }

    cls.textContent = d.closest_class || '--';
}

// Detection List
function updateDetections(d) {
    const list = document.getElementById('detectionList');

    if (!d.detections || d.detections.length === 0) {
        list.innerHTML = '';
        list.appendChild(createEmptyState());
        return;
    }

    // Sort by distance ascending
    const sorted = [...d.detections].sort((a, b) => a.distance - b.distance);

    list.innerHTML = '';
    
    // Add actual detection cards
    for (const det of sorted) {
        list.appendChild(createDetectionCard(det, d));
    }
    
    // Only fill remaining slots if we have FEWER than RESERVED_SLOTS
    // If we have MORE than RESERVED_SLOTS, just let it scroll
    if (sorted.length < RESERVED_SLOTS) {
        const remainingSlots = RESERVED_SLOTS - sorted.length;
        for (let i = 0; i < remainingSlots; i++) {
            list.appendChild(createEmptySlot(sorted.length + i + 1));
        }
    }
}

function createEmptyState() {
    const div = document.createElement('div');
    div.className = 'empty-state';
    div.id = 'emptyState';
    div.innerHTML = `
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.3-4.3"/>
        </svg>
        <p>No objects detected</p>
        <span>Point the camera at obstacles behind the car</span>`;
    return div;
}

function createEmptySlot(slotNumber) {
    const slot = document.createElement('div');
    slot.className = 'detection-slot';
    slot.innerHTML = `<span>Slot ${slotNumber}</span>`;
    return slot;
}

function createDetectionCard(det, data) {
    let level = 'safe';
    if (det.distance < (data.danger_distance || 0.5)) level = 'danger';
    else if (det.distance < (data.caution_distance || 1.0)) level = 'caution';

    const icon = CLASS_ICONS[det.class] || CLASS_ICONS.default;

    const card = document.createElement('div');
    card.className = `detection-card ${level}`;
    card.innerHTML = `
        <div class="det-icon ${level}">${icon}</div>
        <div class="det-info">
            <div class="det-class">${det.class}</div>
            <div class="det-confidence">Confidence: ${(det.confidence * 100).toFixed(0)}%</div>
        </div>
        <div class="det-distance ${level}">${det.distance.toFixed(2)}m</div>`;
    return card;
}

// Video overlay badges
function updateOverlays(d) {
    document.getElementById('fpsBadge').textContent = `${d.fps} FPS`;
    const modeBadge = document.getElementById('modeBadge');
    modeBadge.textContent = d.mode;
    modeBadge.classList.toggle('night', d.mode === 'NIGHT');
    document.getElementById('objectCountOverlay').textContent =
        `${d.object_count} object${d.object_count !== 1 ? 's' : ''}`;
}

// Controls
function updateControls(d) {
    const nightBtn = document.getElementById('btnNightMode');
    const autoBtn = document.getElementById('btnAutoNight');
    const muteBtn = document.getElementById('btnMute');
    const cameraBtn = document.getElementById('btnCameraSource');

    nightBtn.classList.toggle('active', d.night_mode);
    document.getElementById('nightModeState').textContent = d.night_mode ? 'ON' : 'OFF';

    autoBtn.classList.toggle('active', d.auto_night_mode);
    document.getElementById('autoNightState').textContent = d.auto_night_mode ? 'ON' : 'OFF';

    muteBtn.classList.toggle('active', d.muted);
    document.getElementById('muteState').textContent = d.muted ? 'MUTED' : 'ON';
    muteBtn.querySelector('.control-icon').textContent = d.muted ? '🔇' : '🔊';
    
    // Update camera mode display
    if (cameraBtn) {
        cameraBtn.classList.toggle('active', d.camera_mode === 'mobile');
        document.getElementById('cameraSourceState').textContent = 
            d.camera_mode === 'mobile' ? 'PHONE' : 'LAPTOP';
    }
}

// System info
function updateSystem(d) {
    document.getElementById('sysFps').textContent = d.fps;
    document.getElementById('sysObjects').textContent = d.object_count;
    document.getElementById('sysMode').textContent = d.mode;
    document.getElementById('sysLighting').textContent = d.lighting;
}

// Toggle actions
async function toggle(action) {
    try {
        await fetch('/api/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        });
    } catch (e) {
        console.error('Toggle failed:', e);
    }
}

// Fullscreen video
function goFullscreen() {
    isFullscreen = !isFullscreen;
    document.body.classList.toggle('fullscreen-video', isFullscreen);

    if (isFullscreen) {
        // Tap anywhere on the fullscreen video to exit
        document.getElementById('videoSection').addEventListener('click', exitFullscreen, { once: true });
    }
}

function exitFullscreen() {
    isFullscreen = false;
    document.body.classList.remove('fullscreen-video');
}

// Exit fullscreen on back/escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isFullscreen) exitFullscreen();
});

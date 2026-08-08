/* ============================================
   VidForge AI — Application Logic
   AWS Bedrock Integration with Video Generators & FFmpeg Stitching
   ============================================ */

const CONFIG = {
    API_BASE: '',
    POLL_INTERVAL_MS: 5000,
    MAX_POLL_ATTEMPTS: 720, // 60 minutes max for long multi-clip video jobs
};

// --- DOM Elements ---
const els = {
    promptInput: document.getElementById('prompt-input'),
    charCount: document.getElementById('char-count'),
    generateBtn: document.getElementById('generate-btn'),
    btnLoader: document.getElementById('btn-loader'),
    statusSection: document.getElementById('status-section'),
    statusLabel: document.getElementById('status-label'),
    statusDetail: document.getElementById('status-detail'),
    clipBadge: document.getElementById('clip-progress-badge'),
    progressBar: document.getElementById('progress-bar'),
    resultSection: document.getElementById('result-section'),
    resultVideo: document.getElementById('result-video'),
    downloadBtn: document.getElementById('download-btn'),
    newGenerationBtn: document.getElementById('new-generation-btn'),
    errorSection: document.getElementById('error-section'),
    errorMessage: document.getElementById('error-message'),
    retryBtn: document.getElementById('retry-btn'),
};

// --- State ---
let state = {
    isGenerating: false,
    selectedModel: 'amazon.nova-reel-v1:0', // default to Amazon Nova Reel
    selectedDuration: 60, // default 1 minute (60 seconds)
    jobId: null,
    invocationArn: null,
    pollTimer: null,
    startTime: null,
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', init);

function init() {
    if(els.promptInput) els.promptInput.addEventListener('input', updateCharCount);
    if(els.generateBtn) els.generateBtn.addEventListener('click', handleGenerate);
    if(els.newGenerationBtn) els.newGenerationBtn.addEventListener('click', resetToInitial);
    if(els.retryBtn) els.retryBtn.addEventListener('click', handleGenerate);

    // Model Button Selection
    const modelBtns = document.querySelectorAll('.model-btn');
    modelBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modelBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const mod = btn.getAttribute('data-model');
            if (mod) state.selectedModel = mod;
        });
    });

    // Duration Button Selection
    const durationBtns = document.querySelectorAll('.duration-btn');
    durationBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            durationBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const dur = parseInt(btn.getAttribute('data-duration'), 10);
            if (dur) state.selectedDuration = dur;
        });
    });

    updateCharCount();
}

function updateCharCount() {
    if(!els.promptInput || !els.charCount) return;
    const len = els.promptInput.value.length;
    els.charCount.textContent = `${len} / 1000`;

    els.charCount.classList.remove('near-limit', 'at-limit');
    if (len >= 1000) els.charCount.classList.add('at-limit');
    else if (len >= 900) els.charCount.classList.add('near-limit');
}

// --- Generate Handler ---
async function handleGenerate() {
    const textPrompt = els.promptInput.value.trim();
    if (!textPrompt) {
        shakeElement(els.promptInput);
        els.promptInput.focus();
        return;
    }

    if (state.isGenerating) return;

    setGenerating(true);
    hideSection(els.errorSection);
    hideSection(els.resultSection);
    showStatusSection();

    try {
        const payload = {
            prompt: textPrompt,
            model: state.selectedModel,
            duration: state.selectedDuration
        };

        const response = await fetch(`${CONFIG.API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || errorData.message || `API error (${response.status})`);
        }

        const data = await response.json();
        
        if (data.jobId) {
            state.jobId = data.jobId;
            state.invocationArn = null;
            startPolling();
        } else if (data.invocationArn) {
            state.invocationArn = data.invocationArn;
            state.jobId = null;
            startPolling();
        } else {
            throw new Error('No job ID or invocation ARN returned from server.');
        }

    } catch (err) {
        setGenerating(false);
        showError(err.message || 'Failed to start video generation. Please try again.');
    }
}

// --- Polling Logic ---
function startPolling() {
    let attempts = 0;
    state.startTime = Date.now();

    const poll = async () => {
        if (!state.isGenerating) return;
        attempts++;

        if (attempts > CONFIG.MAX_POLL_ATTEMPTS) {
            stopPolling();
            setGenerating(false);
            showError('Rendering timed out. Multi-clip video generation can take several minutes.');
            return;
        }

        try {
            let url = '';
            if (state.jobId) {
                url = `${CONFIG.API_BASE}/api/status?job_id=${encodeURIComponent(state.jobId)}`;
            } else {
                url = `${CONFIG.API_BASE}/api/status?arn=${encodeURIComponent(state.invocationArn)}`;
            }

            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch status');
            
            const data = await response.json();
            
            if (data.status === 'Completed' && data.videoUrl) {
                stopPolling();
                setGenerating(false);
                showResult(data.videoUrl);
            } else if (data.status === 'Failed' || data.error) {
                stopPolling();
                setGenerating(false);
                showError(data.error || 'Video rendering failed. Please try again.');
            } else {
                updateProgressData(data, attempts);
                state.pollTimer = setTimeout(poll, CONFIG.POLL_INTERVAL_MS);
            }
        } catch (err) {
            if (attempts < CONFIG.MAX_POLL_ATTEMPTS) {
                state.pollTimer = setTimeout(poll, CONFIG.POLL_INTERVAL_MS);
            } else {
                stopPolling();
                setGenerating(false);
                showError('Lost connection while checking rendering status.');
            }
        }
    };

    state.pollTimer = setTimeout(poll, CONFIG.POLL_INTERVAL_MS);
}

function stopPolling() {
    if (state.pollTimer) {
        clearTimeout(state.pollTimer);
        state.pollTimer = null;
    }
}

function updateProgressData(data, attempts) {
    if (data.progress !== undefined && els.progressBar) {
        els.progressBar.style.width = `${Math.min(100, Math.max(5, data.progress))}%`;
    } else if (els.progressBar) {
        const artProg = Math.min(95, attempts * 2);
        els.progressBar.style.width = `${artProg}%`;
    }

    const clipBadge = document.getElementById('clip-progress-badge');
    const statusLabel = document.getElementById('status-label');
    const statusDetail = document.getElementById('status-detail');

    if (data.totalClips && data.totalClips > 1) {
        if (clipBadge) {
            clipBadge.style.display = 'inline-block';
            clipBadge.textContent = `${data.completedClips || 0} / ${data.totalClips} Clips`;
        }

        if (data.status === 'Stitching') {
            if (statusLabel) statusLabel.textContent = 'Stitching Video';
            if (statusDetail) statusDetail.textContent = 'Combining all rendered clips into a seamless video with FFmpeg...';
        } else {
            if (statusLabel) statusLabel.textContent = 'Rendering Clips';
            if (statusDetail) statusDetail.textContent = `Generating clip ${Math.min((data.completedClips || 0) + 1, data.totalClips)} of ${data.totalClips} via AWS Bedrock...`;
        }
    } else {
        if (clipBadge) clipBadge.style.display = 'none';
        if (statusLabel) statusLabel.textContent = data.status || 'Rendering';
        if (statusDetail) statusDetail.textContent = 'Generating high-quality video...';
    }
}

// --- UI State Management ---
function setGenerating(isGen) {
    state.isGenerating = isGen;
    if(els.generateBtn) els.generateBtn.disabled = isGen;

    if (isGen) {
        if(els.generateBtn) els.generateBtn.classList.add('loading');
    } else {
        if(els.generateBtn) els.generateBtn.classList.remove('loading');
    }
}

function showStatusSection() {
    if(!els.statusSection) return;
    els.statusSection.style.display = 'block';
    if(els.progressBar) els.progressBar.style.width = '5%';
    if(els.statusLabel) els.statusLabel.textContent = 'Initializing Job';
    if(els.statusDetail) els.statusDetail.textContent = 'Setting up multi-clip generation task...';
    const clipBadge = document.getElementById('clip-progress-badge');
    if(clipBadge) clipBadge.style.display = 'none';

    els.statusSection.style.animation = 'none';
    els.statusSection.offsetHeight;
    els.statusSection.style.animation = '';
}

function showResult(videoUrl) {
    hideSection(els.statusSection);

    let resultVideo = document.getElementById('result-video');
    if (!resultVideo) {
        resultVideo = document.createElement('video');
        resultVideo.id = 'result-video';
        resultVideo.className = 'result-video';
        resultVideo.controls = true;
        resultVideo.autoplay = true;
        resultVideo.loop = true;
        const container = document.getElementById('video-container');
        if (container) {
            container.innerHTML = '';
            container.appendChild(resultVideo);
        }
    }
    
    if(resultVideo) resultVideo.src = videoUrl;
    
    if(els.downloadBtn) {
        els.downloadBtn.href = videoUrl;
        els.downloadBtn.download = "vidforge-stitched-video.mp4";
        els.downloadBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download Video
        `;
    }

    const resultTitle = document.querySelector('.result-title');
    if (resultTitle) {
        resultTitle.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
            Video Rendered & Stitched Successfully!
        `;
    }

    if(els.resultSection) els.resultSection.style.display = 'block';
    if(els.progressBar) els.progressBar.style.width = '100%';

    if(els.resultSection) {
        els.resultSection.style.animation = 'none';
        els.resultSection.offsetHeight;
        els.resultSection.style.animation = '';
    }
}

function showError(message) {
    hideSection(els.statusSection);
    if(els.errorMessage) els.errorMessage.textContent = message;
    if(els.errorSection) els.errorSection.style.display = 'block';

    if(els.errorSection) {
        els.errorSection.style.animation = 'none';
        els.errorSection.offsetHeight;
        els.errorSection.style.animation = '';
    }
}

function hideSection(section) {
    if(section) section.style.display = 'none';
}

function resetToInitial() {
    hideSection(els.resultSection);
    hideSection(els.errorSection);
    hideSection(els.statusSection);
    if(els.progressBar) els.progressBar.style.width = '0%';
    state.jobId = null;
    state.invocationArn = null;
    if(els.promptInput) els.promptInput.focus();
}

function shakeElement(el) {
    if(!el) return;
    el.style.animation = 'none';
    el.offsetHeight;
    el.style.animation = 'shake 0.5s ease';
    el.addEventListener('animationend', () => {
        el.style.animation = '';
    }, { once: true });
}

const shakeStyle = document.createElement('style');
shakeStyle.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 50%, 90% { transform: translateX(-4px); }
        30%, 70% { transform: translateX(4px); }
    }
`;
document.head.appendChild(shakeStyle);

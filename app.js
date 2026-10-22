/* ============================================================
   FaceAuth — Reconhecimento Facial — Application Logic
   ============================================================
   Utiliza face-api.js para:
   1. Detecção de rostos (TinyFaceDetector)
   2. Extração de landmarks faciais (68 pontos)
   3. Geração de embeddings (128 dimensões) via rede neural
   4. Comparação de embeddings por distância euclidiana
   5. Gravação contínua com convergência automática
   ============================================================ */

// ===================== GLOBALS =====================
const MODELS_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.14/model/';
const DETECTION_INTERVAL_MS = 200;
const SAMPLE_CAPTURE_INTERVAL_MS = 200;  // 5 capturas por segundo para amostragem dinâmica
const MIN_SAMPLES_BEFORE_CONVERGENCE = 25; // Exige amostras suficientes para cobrir poses
const MAX_SAMPLES = 60;
const MIN_REQUIRED_PERSPECTIVES = 3;      // Exige cobrir pelo menos 3 perspectivas faciais

// Limiar base de convergência
const BASE_CONVERGENCE_THRESHOLD = 0.040;
const OVERFITTING_SAFETY_MARGIN = 1.4;
const STABILITY_WINDOW = 6;
const MIN_CAPTURE_CONFIDENCE = 0.65;

let videoStream = null;
let isModelLoaded = false;
let isAuthActive = false;
let registeredDescriptor = null;
let registeredTemplates = [];  // Galeria de templates faciais em múltiplos ângulos/distâncias
let registeredName = '';
let collectedDescriptors = [];
let collectedSamplesMeta = []; // Amostras com metadata de orientação
let coveredPerspectives = new Set();
let detectionLoop = null;

// --- Estado da gravação ---
let isRecording = false;
let recordingLoop = null;
let convergenceHistory = [];  // Histórico de erros médios
let adaptiveThreshold = BASE_CONVERGENCE_THRESHOLD;

// Limiar calibrado do face-api.js para reconhecimento biométrico multi-ângulo e invariante a distância
const FIXED_AUTH_THRESHOLD = 0.55;
let serverUsers = []; // Usuários cadastrados no servidor do projeto

// ===================== DOM ELEMENTS =====================
const els = {
    loadingScreen: () => document.getElementById('loadingScreen'),
    loadingStatus: () => document.getElementById('loadingStatus'),
    progressBar: () => document.getElementById('progressBar'),
    cameraSection: () => document.getElementById('cameraSection'),
    videoFeed: () => document.getElementById('videoFeed'),
    overlayCanvas: () => document.getElementById('overlayCanvas'),
    cameraFrame: () => document.getElementById('cameraFrame'),
    scanLine: () => document.getElementById('scanLine'),
    authOverlay: () => document.getElementById('authOverlay'),
    authIcon: () => document.getElementById('authIcon'),
    authText: () => document.getElementById('authText'),
    faceCount: () => document.getElementById('faceCount'),
    authStatusText: () => document.getElementById('authStatusText'),
    distanceValue: () => document.getElementById('distanceValue'),
    headerStatus: () => document.getElementById('headerStatus'),
    serverBadge: () => document.getElementById('serverBadge'),
    serverStatusText: () => document.getElementById('serverStatusText'),
    btnStartRecording: () => document.getElementById('btnStartRecording'),
    btnStopRecording: () => document.getElementById('btnStopRecording'),
    btnReset: () => document.getElementById('btnReset'),
    userName: () => document.getElementById('userName'),
    convergenceBar: () => document.getElementById('convergenceBar'),
    convergenceValue: () => document.getElementById('convergenceValue'),
    convergenceThreshold: () => document.getElementById('convergenceThreshold'),
    sampleCountText: () => document.getElementById('sampleCountText'),
    convergenceStatus: () => document.getElementById('convergenceStatus'),
    recordingIndicator: () => document.getElementById('recordingIndicator'),
    convergenceChart: () => document.getElementById('convergenceChart'),
    authToggle: () => document.getElementById('authToggle'),
    toggleLabel: () => document.getElementById('toggleLabel'),
    registrationCard: () => document.getElementById('registrationCard'),
    userCard: () => document.getElementById('userCard'),
    registeredName: () => document.getElementById('registeredName'),
    registeredMeta: () => document.getElementById('registeredMeta'),
    userAvatar: () => document.getElementById('userAvatar'),
    embeddingBars: () => document.getElementById('embeddingBars'),
    logContainer: () => document.getElementById('logContainer'),
    toastContainer: () => document.getElementById('toastContainer'),
    btnImportProfile: () => document.getElementById('btnImportProfile'),
    btnExportProfile: () => document.getElementById('btnExportProfile'),
    btnChangeProfile: () => document.getElementById('btnChangeProfile'),
    profileFileInput: () => document.getElementById('profileFileInput'),
    importDivider: () => document.getElementById('importDivider'),
    poseIndicator: () => document.getElementById('poseIndicator'),
    tagFrontal: () => document.getElementById('tagFrontal'),
    tagLeft: () => document.getElementById('tagLeft'),
    tagRight: () => document.getElementById('tagRight'),
    tagDistance: () => document.getElementById('tagDistance'),
    guidanceTip: () => document.getElementById('guidanceTip'),
};

// ===================== SERVER API INTEGRATION =====================
const API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
    ? ''
    : 'http://localhost:3000';

async function checkServerStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        if (res.ok) {
            const data = await res.json();
            if (els.serverBadge()) els.serverBadge().className = 'server-badge online';
            if (els.serverStatusText()) els.serverStatusText().textContent = `Servidor Online (${data.totalRegisteredUsers} cad.)`;
            return true;
        }
    } catch (e) {
        if (els.serverBadge()) els.serverBadge().className = 'server-badge offline';
        if (els.serverStatusText()) els.serverStatusText().textContent = 'Servidor Offline';
    }
    return false;
}

async function syncUsersFromServer() {
    try {
        const res = await fetch(`${API_BASE}/api/users`);
        if (res.ok) {
            const data = await res.json();
            if (data.users && data.users.length > 0) {
                serverUsers = data.users.map(u => ({
                    id: u.id,
                    name: u.name,
                    descriptor: new Float32Array(u.descriptor),
                    templates: Array.isArray(u.templates)
                        ? u.templates.map(t => new Float32Array(t))
                        : [new Float32Array(u.descriptor)]
                }));

                // Se ainda não temos um usuário ativo exibido na UI, carrega o mais recente
                if (!registeredDescriptor && serverUsers.length > 0) {
                    const lastUser = serverUsers[serverUsers.length - 1];
                    registeredDescriptor = lastUser.descriptor;
                    registeredTemplates = lastUser.templates || [lastUser.descriptor];
                    registeredName = lastUser.name;
                    showRegisteredUser();
                }

                addLog(`${serverUsers.length} perfil(s) biométrico(s) sincronizado(s) com o servidor`, 'info');
            }
        }
    } catch (e) {
        console.warn('Não foi possível sincronizar usuários com o servidor:', e);
    }
}

async function saveUserToServer(name, descriptor, templates = []) {
    try {
        const res = await fetch(`${API_BASE}/api/users`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                descriptor: Array.from(descriptor),
                templates: templates.map(t => Array.from(t))
            })
        });
        if (res.ok) {
            addLog(`Cadastro de "${name}" salvo com sucesso no servidor do projeto`, 'success');
            await checkServerStatus();
            await syncUsersFromServer();
            return true;
        }
    } catch (e) {
        addLog(`Aviso: Servidor inacessível (${e.message}), mantido localmente`, 'warning');
    }
    return false;
}

async function deleteUserFromServer(nameOrId) {
    try {
        const res = await fetch(`${API_BASE}/api/users/${encodeURIComponent(nameOrId)}`, {
            method: 'DELETE'
        });
        if (res.ok) {
            addLog(`Cadastro de "${nameOrId}" excluído do servidor`, 'info');
            await checkServerStatus();
            await syncUsersFromServer();
            return true;
        }
    } catch (e) {
        console.warn('Erro ao deletar usuário do servidor:', e);
    }
    return false;
}

// ===================== INITIALIZATION =====================
document.addEventListener('DOMContentLoaded', async () => {
    createParticles();
    loadSavedData();
    await checkServerStatus();
    await syncUsersFromServer();
    await loadModels();
    await startCamera();
    showCameraSection();
    startDetection();
    setupEventListeners();
});

// ===================== PARTICLES BACKGROUND =====================
function createParticles() {
    const container = document.getElementById('bgParticles');
    const count = 30;
    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (8 + Math.random() * 12) + 's';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.width = (2 + Math.random() * 3) + 'px';
        particle.style.height = particle.style.width;
        particle.style.background = `hsl(${230 + Math.random() * 50}, 80%, ${55 + Math.random() * 20}%)`;
        container.appendChild(particle);
    }
}

// ===================== LOAD SAVED DATA =====================
function loadSavedData() {
    try {
        const saved = localStorage.getItem('faceauth_data');
        if (saved) {
            const data = JSON.parse(saved);
            registeredDescriptor = new Float32Array(data.descriptor);
            registeredName = data.name || 'Usuário';
            if (Array.isArray(data.templates) && data.templates.length > 0) {
                registeredTemplates = data.templates.map(t => new Float32Array(t));
            } else {
                registeredTemplates = [registeredDescriptor];
            }
            addLog('Dados de registro carregados do armazenamento local', 'info');
        }
    } catch (e) {
        console.warn('Erro ao carregar dados salvos:', e);
    }
}

// ===================== SAVE DATA =====================
function saveData() {
    try {
        const data = {
            version: '1.0',
            appName: 'FaceAuth',
            descriptor: Array.from(registeredDescriptor),
            templates: registeredTemplates.map(t => Array.from(t)),
            name: registeredName,
            timestamp: Date.now()
        };
        localStorage.setItem('faceauth_data', JSON.stringify(data));
    } catch (e) {
        console.warn('Erro ao salvar dados:', e);
    }
}

// ===================== LOAD MODELS =====================
async function loadModels() {
    const progress = els.progressBar();
    const status = els.loadingStatus();

    try {
        status.textContent = 'Carregando detector de rostos (TinyFaceDetector)...';
        progress.style.width = '15%';
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_URL);

        status.textContent = 'Carregando modelo de landmarks faciais...';
        progress.style.width = '40%';
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_URL);

        status.textContent = 'Carregando rede de reconhecimento facial...';
        progress.style.width = '70%';
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_URL);

        status.textContent = 'Carregando detector de expressões...';
        progress.style.width = '90%';
        await faceapi.nets.faceExpressionNet.loadFromUri(MODELS_URL);

        progress.style.width = '100%';
        status.textContent = 'Modelos carregados com sucesso!';
        isModelLoaded = true;

        updateHeaderStatus('Modelos carregados', 'active');
        addLog('Todos os modelos de IA carregados com sucesso', 'success');

    } catch (error) {
        status.textContent = 'Erro ao carregar modelos: ' + error.message;
        updateHeaderStatus('Erro nos modelos', 'error');
        addLog('Falha ao carregar modelos: ' + error.message, 'error');
        showToast('Erro ao carregar modelos de IA', 'error');
        throw error;
    }
}

// ===================== START CAMERA =====================
async function startCamera() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });

        const video = els.videoFeed();
        video.srcObject = videoStream;

        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });

        // Match canvas to video dimensions
        const canvas = els.overlayCanvas();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        updateHeaderStatus('Câmera ativa', 'active');
        addLog('Câmera inicializada com sucesso', 'success');

    } catch (error) {
        updateHeaderStatus('Sem câmera', 'error');
        addLog('Erro ao acessar câmera: ' + error.message, 'error');
        showToast('Não foi possível acessar a câmera', 'error');
        throw error;
    }
}

// ===================== SHOW CAMERA SECTION =====================
function showCameraSection() {
    els.loadingScreen().style.display = 'none';
    els.cameraSection().style.display = 'block';

    if (registeredDescriptor) {
        showRegisteredUser();
    }
}

// ===================== DETECTION LOOP =====================
function startDetection() {
    const video = els.videoFeed();
    const canvas = els.overlayCanvas();
    const ctx = canvas.getContext('2d');

    // Resolução 416 para capturar rostos mesmo a maior distância ou angulados
    const options = new faceapi.TinyFaceDetectorOptions({
        inputSize: 416,
        scoreThreshold: 0.45
    });

    detectionLoop = setInterval(async () => {
        if (!isModelLoaded || video.paused || video.ended) return;

        try {
            const detections = await faceapi
                .detectAllFaces(video, options)
                .withFaceLandmarks()
                .withFaceDescriptors();

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Update face count
            els.faceCount().textContent = detections.length;

            if (detections.length === 0) {
                if (els.authStatusText()) {
                    els.authStatusText().textContent = 'Aguardando';
                    els.authStatusText().className = 'info-value info-status-text';
                }
                els.distanceValue().textContent = '—';

                if (isAuthActive) {
                    hideAuthOverlay();
                    els.cameraFrame().classList.remove('authorized', 'denied');
                }
                return;
            }

            // Draw detections (mirrored to match video)
            const displaySize = { width: canvas.width, height: canvas.height };
            const resized = faceapi.resizeResults(detections, displaySize);

            ctx.save();
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);

            let matchedIdentity = null;

            resized.forEach(det => {
                const box = det.detection.box;

                // Determine color based on auth state
                let color = 'rgba(99, 102, 241, 0.8)';
                if (isAuthActive && (registeredDescriptor || serverUsers.length > 0)) {
                    const matchResult = matchDescriptorAgainstGallery(det.descriptor);
                    if (matchResult.matched) {
                        color = 'rgba(34, 197, 94, 0.8)';
                        matchedIdentity = matchResult.name;
                    } else {
                        color = 'rgba(239, 68, 68, 0.8)';
                    }
                }

                // Draw box
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.strokeRect(box.x, box.y, box.width, box.height);
                ctx.setLineDash([]);

                // Draw landmarks
                const landmarks = det.landmarks;
                const positions = landmarks.positions;
                ctx.fillStyle = color;
                positions.forEach(pt => {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 1.5, 0, Math.PI * 2);
                    ctx.fill();
                });

                // Rótulo da detecção quando autenticado
                if (isAuthActive && matchedIdentity) {
                    ctx.fillStyle = color;
                    ctx.font = '600 13px Inter, sans-serif';
                    ctx.fillText(
                        matchedIdentity,
                        box.x,
                        box.y - 6
                    );
                }
            });

            ctx.restore();

            // Authentication check
            if (isAuthActive && (registeredDescriptor || serverUsers.length > 0) && detections.length > 0) {
                performAuth(detections[0]);
            } else if (els.authStatusText()) {
                els.authStatusText().textContent = 'Detectado';
                els.authStatusText().className = 'info-value info-status-text';
            }

        } catch (e) {
            // Silently handle intermittent detection errors
        }
    }, DETECTION_INTERVAL_MS);
}

/**
 * Realiza o matching biométrico invariante a ângulo e distância
 * calculando a distância mínima em relação à galeria multi-template.
 */
function matchDescriptorAgainstGallery(descriptor) {
    let bestName = null;
    let minDistance = Infinity;

    // 1. Testa contra o perfil atual
    if (registeredDescriptor) {
        const templates = (registeredTemplates && registeredTemplates.length > 0)
            ? registeredTemplates
            : [registeredDescriptor];

        for (const t of templates) {
            const d = faceapi.euclideanDistance(descriptor, t);
            if (d < minDistance) minDistance = d;
        }

        if (minDistance < FIXED_AUTH_THRESHOLD) {
            bestName = registeredName;
        }
    }

    // 2. Testa contra os demais perfis do servidor
    if (!bestName && serverUsers.length > 0) {
        for (const u of serverUsers) {
            const userTemplates = (u.templates && u.templates.length > 0)
                ? u.templates
                : [u.descriptor];

            for (const t of userTemplates) {
                const d = faceapi.euclideanDistance(descriptor, t);
                if (d < minDistance) {
                    minDistance = d;
                    if (d < FIXED_AUTH_THRESHOLD) {
                        bestName = u.name;
                    }
                }
            }
            if (bestName) break;
        }
    }

    return {
        matched: !!bestName,
        name: bestName,
        distance: minDistance
    };
}

// ===================== AUTHENTICATION =====================
function performAuth(detection) {
    const match = matchDescriptorAgainstGallery(detection.descriptor);
    els.distanceValue().textContent = match.distance === Infinity ? '—' : match.distance.toFixed(3);

    if (match.matched) {
        // AUTHORIZED
        els.cameraFrame().classList.add('authorized');
        els.cameraFrame().classList.remove('denied');
        showAuthOverlay('✓ ACESSO AUTORIZADO', 'granted', match.name);
        els.scanLine().classList.remove('active');
        if (els.authStatusText()) {
            els.authStatusText().textContent = 'Autorizado';
            els.authStatusText().className = 'info-value info-status-text granted';
        }
    } else {
        // DENIED
        els.cameraFrame().classList.add('denied');
        els.cameraFrame().classList.remove('authorized');
        showAuthOverlay('✗ ACESSO NEGADO', 'denied-state', 'Não Reconhecido');
        els.scanLine().classList.add('active');
        if (els.authStatusText()) {
            els.authStatusText().textContent = 'Negado';
            els.authStatusText().className = 'info-value info-status-text denied';
        }
    }
}

// ===================== AUTH OVERLAY =====================
function showAuthOverlay(text, className, subtitle) {
    const overlay = els.authOverlay();
    const icon = els.authIcon();
    const textEl = els.authText();

    overlay.className = 'auth-overlay show ' + className;
    icon.textContent = className === 'granted' ? '🔓' : '🔒';
    textEl.textContent = text + (subtitle ? ` — ${subtitle}` : '');
}

function hideAuthOverlay() {
    els.authOverlay().className = 'auth-overlay';
}

// ===================== CONVERGENCE CALCULATIONS =====================

/**
 * Calcula o limiar adaptativo de convergência para evitar overfitting.
 * 
 * O limiar é pré-calculado com base em:
 * 1. Um limiar base (BASE_CONVERGENCE_THRESHOLD) derivado empiricamente
 *    da distribuição esperada de distâncias intra-pessoa em embeddings 128D.
 * 2. O desvio-padrão das distâncias recentes (janela deslizante),
 *    que captura a variabilidade natural do rosto sob condições reais.
 * 3. Uma margem de segurança (OVERFITTING_SAFETY_MARGIN) que impede
 *    que o modelo "decore" artefatos de iluminação/posição específicos.
 *
 * Fórmula: threshold = max(BASE, mean_recent_std * safety_margin)
 * 
 * Isso garante que:
 * - O limiar nunca fique abaixo do ruído mínimo esperado (anti-overfitting)
 * - O limiar se adapte à variabilidade real do rosto do usuário
 */
function computeAdaptiveThreshold() {
    if (collectedDescriptors.length < 3) {
        return BASE_CONVERGENCE_THRESHOLD;
    }

    // Calcula todas as distâncias par-a-par entre os descritores recentes
    const recentDescs = collectedDescriptors.slice(-STABILITY_WINDOW);
    const pairwiseDistances = [];

    for (let i = 0; i < recentDescs.length; i++) {
        for (let j = i + 1; j < recentDescs.length; j++) {
            pairwiseDistances.push(
                faceapi.euclideanDistance(recentDescs[i], recentDescs[j])
            );
        }
    }

    if (pairwiseDistances.length === 0) return BASE_CONVERGENCE_THRESHOLD;

    // Média e desvio-padrão das distâncias par-a-par
    const mean = pairwiseDistances.reduce((a, b) => a + b, 0) / pairwiseDistances.length;
    const variance = pairwiseDistances.reduce((a, b) => a + (b - mean) ** 2, 0) / pairwiseDistances.length;
    const std = Math.sqrt(variance);

    // O limiar adaptativo é o máximo entre o base e uma fração
    // baseada na variabilidade atual, com margem de segurança
    const adaptivePart = std * OVERFITTING_SAFETY_MARGIN;
    const threshold = Math.max(BASE_CONVERGENCE_THRESHOLD, adaptivePart);

    return threshold;
}

/**
 * Calcula o erro médio atual entre todos os descritores coletados
 * e o descritor médio (centróide).
 * 
 * Esse é o "erro de convergência" — quanto menor, mais estáveis
 * são as amostras coletadas.
 */
function computeConvergenceError() {
    if (collectedDescriptors.length < 2) return Infinity;

    // Calcula o centróide (descritor médio)
    const centroid = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
        let sum = 0;
        for (const desc of collectedDescriptors) {
            sum += desc[i];
        }
        centroid[i] = sum / collectedDescriptors.length;
    }

    // Normaliza o centróide
    let norm = 0;
    for (let i = 0; i < 128; i++) {
        norm += centroid[i] * centroid[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 0) {
        for (let i = 0; i < 128; i++) {
            centroid[i] /= norm;
        }
    }

    // Calcula distância média de cada amostra ao centróide
    let totalDist = 0;
    for (const desc of collectedDescriptors) {
        totalDist += faceapi.euclideanDistance(desc, centroid);
    }
    const avgError = totalDist / collectedDescriptors.length;

    return avgError;
}

/**
 * Verifica se a convergência foi alcançada de forma segura (sem overfitting).
 * 
 * Critérios:
 * 1. Mínimo de MIN_SAMPLES_BEFORE_CONVERGENCE amostras coletadas
 * 2. O erro atual está abaixo do limiar adaptativo
 * 3. O erro está estável (variação recente baixa)
 */
// ===================== POSE ANALYSIS =====================
/**
 * Analisa a orientação espacial (yaw/pitch) e escala/distância do rosto
 * a partir dos 68 landmarks faciais e bounding box.
 */
function analyzeHeadPose(detection, video) {
    const landmarks = detection.landmarks.positions;
    const box = detection.detection.box;

    // Ponto da ponta do nariz (landmark 30)
    const noseX = landmarks[30].x;

    // Cantos externos dos olhos (36 esquerdo, 45 direito)
    const leftEyeX = landmarks[36].x;
    const rightEyeX = landmarks[45].x;
    const eyeCenter = (leftEyeX + rightEyeX) / 2;
    const eyeDist = Math.max(1, Math.abs(rightEyeX - leftEyeX));

    // Yaw horizontal (razão de deslocamento do nariz em relação ao centro dos olhos)
    const yawRatio = (noseX - eyeCenter) / eyeDist;

    // Escala/Distância (fração da largura da face sobre o vídeo)
    const boxRatio = box.width / Math.max(1, video.videoWidth);

    let yawTag = 'frontal';
    let yawLabel = 'Frontal';
    if (yawRatio < -0.10) {
        yawTag = 'direita';
        yawLabel = 'Virado p/ Direita';
    } else if (yawRatio > 0.10) {
        yawTag = 'esquerda';
        yawLabel = 'Virado p/ Esquerda';
    }

    let distTag = 'medio';
    let distLabel = 'Médio';
    if (boxRatio > 0.32) {
        distTag = 'perto';
        distLabel = 'Perto';
    } else if (boxRatio < 0.20) {
        distTag = 'longe';
        distLabel = 'Longe';
    }

    return {
        yawTag,
        yawLabel,
        distTag,
        distLabel,
        yawRatio,
        boxRatio,
        fullLabel: `${yawLabel} • ${distLabel}`
    };
}

/**
 * Verifica se a convergência foi alcançada cobrindo perspectivas suficientes
 * e sem overfitting.
 */
function checkConvergence() {
    const numSamples = collectedDescriptors.length;

    // Requer número mínimo de amostras
    if (numSamples < MIN_SAMPLES_BEFORE_CONVERGENCE) {
        return { converged: false, reason: 'amostras_insuficientes' };
    }

    // Requer cobertura mínima de perspectivas (ex: Frontal + Virado + Variação de Distância)
    if (coveredPerspectives.size < MIN_REQUIRED_PERSPECTIVES && numSamples < MAX_SAMPLES) {
        return { converged: false, reason: 'falta_angulos' };
    }

    if (numSamples >= MAX_SAMPLES) {
        return { converged: true, reason: 'max_amostras' };
    }

    const currentError = computeConvergenceError();
    adaptiveThreshold = computeAdaptiveThreshold();

    if (currentError < adaptiveThreshold && coveredPerspectives.size >= MIN_REQUIRED_PERSPECTIVES) {
        return { converged: true, reason: 'convergido', error: currentError };
    }

    return { converged: false, reason: 'em_progresso', error: currentError };
}

// ===================== RECORDING & AUTO-CAPTURE =====================

/**
 * Inicia a gravação contínua do rosto com guia multi-ângulo e amostragem rápida.
 */
async function startRecording() {
    if (!isModelLoaded) {
        showToast('Modelos ainda não carregados', 'error');
        return;
    }

    isRecording = true;
    collectedDescriptors = [];
    collectedSamplesMeta = [];
    coveredPerspectives = new Set();
    convergenceHistory = [];
    adaptiveThreshold = BASE_CONVERGENCE_THRESHOLD;

    // Atualiza UI
    els.btnStartRecording().style.display = 'none';
    els.btnStopRecording().style.display = 'flex';
    if (els.btnImportProfile()) els.btnImportProfile().style.display = 'none';
    if (els.importDivider()) els.importDivider().style.display = 'none';
    els.recordingIndicator().classList.add('active');
    els.convergenceStatus().textContent = 'Iniciando captura multi-ângulo...';
    els.convergenceStatus().className = 'convergence-status status-recording';

    // Reseta tags visuais de cobertura
    ['tagFrontal', 'tagLeft', 'tagRight', 'tagDistance'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('covered');
    });

    if (els.poseIndicator()) {
        els.poseIndicator().textContent = 'Posicionando...';
    }
    if (els.guidanceTip()) {
        els.guidanceTip().textContent = 'Olhe para a câmera, vire levemente a cabeça para os lados e varie a distância.';
    }

    clearConvergenceChart();

    els.cameraFrame().classList.add('recording');
    els.scanLine().classList.add('active');

    addLog('Gravação facial iniciada — capture poses frontais, viradas e diferentes distâncias', 'info');
    showToast('Gravação iniciada! Mova a cabeça suavemente em diferentes posições.', 'info');

    // Loop de captura rápida (5x por segundo)
    recordingLoop = setInterval(async () => {
        if (!isRecording) return;
        await captureAutoSample();
    }, SAMPLE_CAPTURE_INTERVAL_MS);
}

/**
 * Para a gravação e finaliza o registro.
 */
function stopRecording(autoConverged = false) {
    isRecording = false;

    if (recordingLoop) {
        clearInterval(recordingLoop);
        recordingLoop = null;
    }

    els.btnStopRecording().style.display = 'none';
    els.recordingIndicator().classList.remove('active');
    els.cameraFrame().classList.remove('recording');
    els.scanLine().classList.remove('active');

    if (collectedDescriptors.length < 5) {
        els.btnStartRecording().style.display = 'flex';
        if (els.btnImportProfile()) els.btnImportProfile().style.display = 'flex';
        if (els.importDivider()) els.importDivider().style.display = 'flex';
        els.convergenceStatus().textContent = 'Amostras insuficientes';
        els.convergenceStatus().className = 'convergence-status status-error';
        showToast('Poucas amostras capturadas. Tente novamente.', 'error');
        addLog('Gravação interrompida — amostras insuficientes', 'warning');
        return;
    }

    finishRegistration(autoConverged);
}

/**
 * Captura uma amostra automaticamente com análise de pose e perspectiva.
 */
async function captureAutoSample() {
    const video = els.videoFeed();
    const options = new faceapi.TinyFaceDetectorOptions({
        inputSize: 416,
        scoreThreshold: 0.5
    });

    try {
        const detection = await faceapi
            .detectSingleFace(video, options)
            .withFaceLandmarks()
            .withFaceDescriptor();

        if (!detection) {
            els.convergenceStatus().textContent = 'Nenhum rosto detectado...';
            els.convergenceStatus().className = 'convergence-status status-warning';
            return;
        }

        if (detection.detection.score < MIN_CAPTURE_CONFIDENCE) {
            els.convergenceStatus().textContent = 'Iluminação baixa ou rosto distante';
            els.convergenceStatus().className = 'convergence-status status-warning';
            return;
        }

        // Analisa a pose da cabeça e distância
        const pose = analyzeHeadPose(detection, video);

        // Atualiza crachá de orientação atual
        if (els.poseIndicator()) {
            els.poseIndicator().textContent = pose.fullLabel;
        }

        // Registra perspectivas cobertas
        if (pose.yawTag === 'frontal') {
            coveredPerspectives.add('frontal');
            if (els.tagFrontal()) els.tagFrontal().classList.add('covered');
        } else if (pose.yawTag === 'esquerda') {
            coveredPerspectives.add('esquerda');
            if (els.tagLeft()) els.tagLeft().classList.add('covered');
        } else if (pose.yawTag === 'direita') {
            coveredPerspectives.add('direita');
            if (els.tagRight()) els.tagRight().classList.add('covered');
        }

        if (pose.distTag === 'perto' || pose.distTag === 'longe') {
            coveredPerspectives.add('distancia');
            if (els.tagDistance()) els.tagDistance().classList.add('covered');
        }

        // Atualiza dica em tempo real para orientar o usuário
        if (els.guidanceTip()) {
            if (!coveredPerspectives.has('esquerda')) {
                els.guidanceTip().textContent = '👉 Vire suavemente o rosto para a esquerda...';
            } else if (!coveredPerspectives.has('direita')) {
                els.guidanceTip().textContent = '👈 Vire suavemente o rosto para a direita...';
            } else if (!coveredPerspectives.has('distancia')) {
                els.guidanceTip().textContent = '🔍 Aproxime-se e afaste-se um pouco da câmera...';
            } else {
                els.guidanceTip().textContent = '✓ Excelente cobertura de ângulos! Finalizando registro...';
            }
        }

        // Guarda amostra e metadados
        collectedDescriptors.push(detection.descriptor);
        collectedSamplesMeta.push({
            descriptor: detection.descriptor,
            pose: pose
        });

        const count = collectedDescriptors.length;

        // Métricas de convergência
        const currentError = computeConvergenceError();
        adaptiveThreshold = computeAdaptiveThreshold();
        convergenceHistory.push(currentError);

        updateConvergenceUI(currentError, count);

        // Flash sutil na borda
        els.cameraFrame().style.boxShadow = '0 0 35px rgba(99, 102, 241, 0.45)';
        setTimeout(() => {
            els.cameraFrame().style.boxShadow = '';
        }, 150);

        // Verifica convergência com cobertura multi-ângulo
        const result = checkConvergence();

        if (result.converged) {
            const reason = result.reason === 'max_amostras'
                ? 'limite máximo de amostras atingido'
                : 'ângulos e convergência facial completados com sucesso';

            addLog(`✓ Registro multi-ângulo concluído: ${reason}`, 'success');
            showToast('Rosto registrado em múltiplos ângulos com sucesso!', 'success');
            stopRecording(true);
            return;
        }

    } catch (e) {
        // Silently handle intermittent errors
    }
}

/**
 * Atualiza a interface de convergência em tempo real.
 */
function updateConvergenceUI(error, sampleCount) {
    // Barra de progresso de convergência
    // Mapeia o erro para uma porcentagem (1.0 = 0%, 0 = 100%)
    const maxError = 0.5;
    const progress = Math.max(0, Math.min(100, (1 - error / maxError) * 100));
    els.convergenceBar().style.width = progress + '%';

    // Cor da barra baseada no progresso
    if (progress > 80) {
        els.convergenceBar().style.background = 'linear-gradient(90deg, #22c55e, #4ade80)';
    } else if (progress > 50) {
        els.convergenceBar().style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
    } else {
        els.convergenceBar().style.background = 'var(--accent-gradient)';
    }

    // Valores numéricos
    els.convergenceValue().textContent = error === Infinity ? '—' : error.toFixed(4);
    els.convergenceThreshold().textContent = adaptiveThreshold.toFixed(4);
    els.sampleCountText().textContent = sampleCount;

    // Status textual
    if (sampleCount < MIN_SAMPLES_BEFORE_CONVERGENCE) {
        els.convergenceStatus().textContent = `Coletando amostras... (${sampleCount}/${MIN_SAMPLES_BEFORE_CONVERGENCE} mín.)`;
        els.convergenceStatus().className = 'convergence-status status-recording';
    } else if (error < adaptiveThreshold) {
        els.convergenceStatus().textContent = 'Convergindo...';
        els.convergenceStatus().className = 'convergence-status status-converging';
    } else {
        els.convergenceStatus().textContent = 'Gravando — mova levemente o rosto';
        els.convergenceStatus().className = 'convergence-status status-recording';
    }

    // Atualiza gráfico mini de convergência
    drawConvergenceChart();
}

/**
 * Desenha o mini-gráfico de convergência em tempo real.
 */
function drawConvergenceChart() {
    const canvas = els.convergenceChart();
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    const cw = w / 2;
    const ch = h / 2;

    ctx.clearRect(0, 0, cw, ch);

    if (convergenceHistory.length < 2) return;

    const maxVal = Math.max(...convergenceHistory, adaptiveThreshold) * 1.2;
    const points = convergenceHistory.slice(-40); // Últimos 40 pontos

    // Linha do limiar
    const threshY = ch - (adaptiveThreshold / maxVal) * (ch - 8) - 4;
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, threshY);
    ctx.lineTo(cw, threshY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label do limiar
    ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText('limiar', cw - 30, threshY - 3);

    // Gradiente sob a curva
    const gradient = ctx.createLinearGradient(0, 0, 0, ch);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0.02)');

    // Linha de erro
    ctx.beginPath();
    const step = cw / Math.max(points.length - 1, 1);

    points.forEach((val, i) => {
        const x = i * step;
        const y = ch - (val / maxVal) * (ch - 8) - 4;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = 'rgba(99, 102, 241, 0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Preenchimento
    ctx.lineTo((points.length - 1) * step, ch);
    ctx.lineTo(0, ch);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Ponto atual
    const lastX = (points.length - 1) * step;
    const lastY = ch - (points[points.length - 1] / maxVal) * (ch - 8) - 4;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = points[points.length - 1] < adaptiveThreshold
        ? 'rgba(34, 197, 94, 1)'
        : 'rgba(99, 102, 241, 1)';
    ctx.fill();
}

function clearConvergenceChart() {
    const canvas = els.convergenceChart();
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    els.convergenceBar().style.width = '0%';
    els.convergenceValue().textContent = '—';
    els.convergenceThreshold().textContent = BASE_CONVERGENCE_THRESHOLD.toFixed(4);
    els.sampleCountText().textContent = '0';
}

// ===================== REGISTRATION =====================
/**
 * Conclui o cadastro criando o centróide e a galeria de templates biométricos
 * multi-ângulo para reconhecimento invariante a poses e distâncias.
 */
function finishRegistration(autoConverged = false) {
    const name = els.userName().value.trim() || 'Usuário';

    if (collectedDescriptors.length === 0) return;

    // 1. Calcula o centróide médio
    const centroid = new Float32Array(128);
    for (let i = 0; i < 128; i++) {
        let sum = 0;
        for (const desc of collectedDescriptors) {
            sum += desc[i];
        }
        centroid[i] = sum / collectedDescriptors.length;
    }

    let norm = 0;
    for (let i = 0; i < 128; i++) {
        norm += centroid[i] * centroid[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 0) {
        for (let i = 0; i < 128; i++) {
            centroid[i] /= norm;
        }
    }

    // 2. Extrai templates biométricos representativos de cada perspectiva
    const templates = [centroid];

    // Busca o melhor exemplar de cada categoria de pose registrada
    const categories = ['frontal', 'esquerda', 'direita', 'perto', 'longe'];
    categories.forEach(cat => {
        const matchingSamples = collectedSamplesMeta.filter(s =>
            (s.pose && s.pose.yawTag === cat) || (s.pose && s.pose.distTag === cat)
        );

        if (matchingSamples.length > 0) {
            // Pega o exemplar mais próximo do centróide dessa categoria
            const exemplar = matchingSamples[Math.floor(matchingSamples.length / 2)].descriptor;
            templates.push(exemplar);
        }
    });

    // Se ainda tivermos menos de 5 templates, adiciona amostras adicionais bem espaçadas
    if (templates.length < 5 && collectedDescriptors.length >= 5) {
        const step = Math.floor(collectedDescriptors.length / 4);
        for (let k = 0; k < collectedDescriptors.length; k += step) {
            templates.push(collectedDescriptors[k]);
            if (templates.length >= 8) break;
        }
    }

    registeredDescriptor = centroid;
    registeredTemplates = templates;
    registeredName = name;

    const finalError = computeConvergenceError();

    // Persiste localmente e no servidor
    saveData();
    saveUserToServer(name, centroid, templates);

    showRegisteredUser();

    const methodLabel = autoConverged ? 'convergência multi-ângulo automática' : 'parada manual';
    addLog(
        `Rosto de "${name}" registrado com sucesso! (${collectedDescriptors.length} amostras, ${templates.length} templates de ângulo/distância)`,
        'success'
    );
    showToast(`Rosto de "${name}" cadastrado com sucesso em múltiplos ângulos!`, 'success');

    collectedDescriptors = [];
    collectedSamplesMeta = [];
    convergenceHistory = [];
}

function showRegisteredUser() {
    els.registrationCard().style.display = 'none';
    els.userCard().style.display = 'block';

    els.registeredName().textContent = registeredName;
    els.registeredMeta().textContent = `Embedding 128D • ${new Date().toLocaleDateString('pt-BR')}`;

    // Avatar initial
    els.userAvatar().textContent = registeredName.charAt(0).toUpperCase();

    // Render embedding bars
    renderEmbeddingBars();

    // Enable auth toggle
    els.authToggle().disabled = false;
}

function renderEmbeddingBars() {
    const container = els.embeddingBars();
    container.innerHTML = '';

    if (!registeredDescriptor) return;

    // Find min and max
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < registeredDescriptor.length; i++) {
        if (registeredDescriptor[i] < min) min = registeredDescriptor[i];
        if (registeredDescriptor[i] > max) max = registeredDescriptor[i];
    }
    const range = max - min || 1;

    for (let i = 0; i < registeredDescriptor.length; i++) {
        const bar = document.createElement('div');
        bar.classList.add('emb-bar');
        const normalized = (registeredDescriptor[i] - min) / range;
        const height = 8 + normalized * 36;
        const hue = 230 + normalized * 60;
        bar.style.height = height + 'px';
        bar.style.background = `hsl(${hue}, 70%, ${45 + normalized * 25}%)`;
        container.appendChild(bar);
    }
}

function resetRegistration() {
    const oldName = registeredName;
    registeredDescriptor = null;
    registeredName = '';
    collectedDescriptors = [];
    convergenceHistory = [];

    localStorage.removeItem('faceauth_data');

    // Remove do servidor
    if (oldName) {
        deleteUserFromServer(oldName);
    }

    els.userCard().style.display = 'none';
    els.registrationCard().style.display = 'block';
    els.btnStartRecording().style.display = 'flex';
    els.btnStopRecording().style.display = 'none';
    if (els.btnImportProfile()) els.btnImportProfile().style.display = 'flex';
    if (els.importDivider()) els.importDivider().style.display = 'flex';
    els.userName().value = '';

    // Reseta convergência
    clearConvergenceChart();
    els.convergenceStatus().textContent = 'Aguardando início da gravação';
    els.convergenceStatus().className = 'convergence-status';

    // Disable auth
    isAuthActive = false;
    els.authToggle().checked = false;
    els.authToggle().disabled = false;
    els.toggleLabel().textContent = 'Verificação Desligada';
    hideAuthOverlay();
    els.cameraFrame().classList.remove('authorized', 'denied');
    els.scanLine().classList.remove('active');

    addLog('Registro facial removido', 'warning');
    showToast('Registro removido com sucesso', 'info');
}

// ===================== EVENT LISTENERS =====================
function setupEventListeners() {
    // Start recording button
    els.btnStartRecording().addEventListener('click', () => {
        startRecording();
    });

    // Stop recording button
    els.btnStopRecording().addEventListener('click', () => {
        stopRecording(false);
    });

    // Reset registration
    els.btnReset().addEventListener('click', () => {
        if (confirm('Tem certeza que deseja remover o registro facial?')) {
            resetRegistration();
        }
    });

    // Auth toggle
    els.authToggle().addEventListener('change', (e) => {
        isAuthActive = e.target.checked;

        if (isAuthActive) {
            if (!registeredDescriptor) {
                showToast('Registre um rosto primeiro!', 'error');
                e.target.checked = false;
                isAuthActive = false;
                return;
            }
            els.toggleLabel().textContent = 'Verificação Ligada';
            els.scanLine().classList.add('active');
            addLog('Modo de autenticação ativado', 'info');
            showToast('Verificação em tempo real ativada', 'info');
        } else {
            els.toggleLabel().textContent = 'Verificação Desligada';
            els.scanLine().classList.remove('active');
            hideAuthOverlay();
            els.cameraFrame().classList.remove('authorized', 'denied');
            els.distanceValue().textContent = '—';
            addLog('Modo de autenticação desativado', 'info');
        }
    });

    // JSON Export button
    if (els.btnExportProfile()) {
        els.btnExportProfile().addEventListener('click', () => {
            exportProfileJSON();
        });
    }

    // JSON Import button (from registration card)
    if (els.btnImportProfile()) {
        els.btnImportProfile().addEventListener('click', () => {
            els.profileFileInput().click();
        });
    }

    // JSON Change Profile button (from user card)
    if (els.btnChangeProfile()) {
        els.btnChangeProfile().addEventListener('click', () => {
            els.profileFileInput().click();
        });
    }

    // Hidden file input change listener
    if (els.profileFileInput()) {
        els.profileFileInput().addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                importProfileFromFile(e.target.files[0]);
            }
            e.target.value = '';
        });
    }

    // Setup drag and drop for JSON file anywhere
    setupDragAndDrop();
}

// ===================== EXPORT & IMPORT PROFILE JSON =====================

/**
 * Exporta os parâmetros faciais registrados para um arquivo JSON baixável pelo usuário.
 */
function exportProfileJSON() {
    if (!registeredDescriptor) {
        showToast('Nenhum perfil facial registrado para exportar!', 'error');
        return;
    }

    const payload = {
        version: '1.0',
        appName: 'FaceAuth',
        name: registeredName || 'Usuário',
        exportedAt: new Date().toISOString(),
        parameters: {
            dimensions: registeredDescriptor.length,
            authThreshold: FIXED_AUTH_THRESHOLD,
            descriptor: Array.from(registeredDescriptor),
            templates: (registeredTemplates && registeredTemplates.length > 0)
                ? registeredTemplates.map(t => Array.from(t))
                : [Array.from(registeredDescriptor)]
        }
    };

    const jsonStr = JSON.stringify(payload, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const safeName = (registeredName || 'usuario')
        .toLowerCase()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/[^a-z0-9]/gi, '_')
        .replace(/_+/g, '_');
    const filename = `faceauth_parametros_${safeName}.json`;

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    addLog(`Parâmetros faciais exportados para "${filename}"`, 'success');
    showToast(`Arquivo "${filename}" baixado com sucesso!`, 'success');
}

/**
 * Lê e valida um arquivo JSON com os parâmetros faciais e carrega no sistema.
 */
function importProfileFromFile(file) {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.json') && file.type && file.type !== 'application/json') {
        showToast('Por favor, selecione um arquivo JSON válido (.json)', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const rawContent = e.target.result;
            const data = JSON.parse(rawContent);

            // Suporta múltiplos formatos:
            // 1. { parameters: { descriptor: [...], recommendedThreshold: 0.45 }, name: "..." }
            // 2. { descriptor: [...], name: "...", threshold: 0.45 }
            // 3. Array direto de 128 números
            let descriptorArray = null;
            let name = data.name || (data.parameters && data.parameters.name) || 'Usuário';
            let threshold = null;

            if (data.parameters && Array.isArray(data.parameters.descriptor)) {
                descriptorArray = data.parameters.descriptor;
                threshold = data.parameters.recommendedThreshold;
            } else if (Array.isArray(data.descriptor)) {
                descriptorArray = data.descriptor;
                threshold = data.threshold || data.recommendedThreshold;
            } else if (Array.isArray(data) && data.length === 128) {
                descriptorArray = data;
            }

            if (!descriptorArray || descriptorArray.length !== 128) {
                throw new Error('O arquivo não contém um vetor de características faciais válido (são necessárias 128 dimensões).');
            }

            // Valida se todos os elementos são numéricos finitos
            const isValidNumbers = descriptorArray.every(val => typeof val === 'number' && Number.isFinite(val));
            if (!isValidNumbers) {
                throw new Error('Os valores do embedding facial no arquivo JSON são inválidos.');
            }

            // Normaliza o vetor caso não esteja perfeitamente unitário
            const floatDesc = new Float32Array(descriptorArray);
            let norm = 0;
            for (let i = 0; i < 128; i++) {
                norm += floatDesc[i] * floatDesc[i];
            }
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (let i = 0; i < 128; i++) {
                    floatDesc[i] /= norm;
                }
            }

            // Atribui ao estado
            registeredDescriptor = floatDesc;
            registeredName = name;

            // Extrai templates se presentes no JSON
            if (data.parameters && Array.isArray(data.parameters.templates)) {
                registeredTemplates = data.parameters.templates.map(t => new Float32Array(t));
            } else if (Array.isArray(data.templates)) {
                registeredTemplates = data.templates.map(t => new Float32Array(t));
            } else {
                registeredTemplates = [floatDesc];
            }

            // Persiste localmente e no servidor
            saveData();
            saveUserToServer(name, floatDesc, registeredTemplates);

            // Atualiza UI
            showRegisteredUser();

            // Se a verificação facial estava ativa, reseta overlays
            if (isAuthActive) {
                hideAuthOverlay();
                els.cameraFrame().classList.remove('authorized', 'denied');
            }

            addLog(`Parâmetros faciais de "${registeredName}" carregados com sucesso via JSON (128D)`, 'success');
            showToast(`Perfil de "${registeredName}" importado e sincronizado com o servidor!`, 'success');

        } catch (err) {
            console.error('Erro ao importar JSON:', err);
            addLog(`Falha ao importar arquivo JSON: ${err.message}`, 'error');
            showToast(`Erro no arquivo JSON: ${err.message}`, 'error');
        }
    };

    reader.onerror = () => {
        showToast('Erro ao ler o arquivo JSON selecionado.', 'error');
    };

    reader.readAsText(file);
}

/**
 * Configura arrastar e soltar (drag & drop) para arquivos JSON em qualquer parte da tela.
 */
function setupDragAndDrop() {
    const cards = [els.registrationCard(), els.userCard(), els.cameraFrame()];

    ['dragenter', 'dragover'].forEach(name => {
        window.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);

        cards.forEach(card => {
            if (!card) return;
            card.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                card.classList.add('drag-over');
            }, false);
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        window.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);

        cards.forEach(card => {
            if (!card) return;
            card.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                card.classList.remove('drag-over');
            }, false);
        });
    });

    window.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.name.toLowerCase().endsWith('.json') || file.type === 'application/json') {
                importProfileFromFile(file);
            } else {
                showToast('Apenas arquivos de parâmetros em JSON (.json) são aceitos.', 'warning');
            }
        }
    }, false);
}

// ===================== UI HELPERS =====================
function updateHeaderStatus(text, state) {
    const header = els.headerStatus();
    const dot = header.querySelector('.status-dot');
    const label = header.querySelector('.status-text');

    dot.className = 'status-dot';
    if (state) dot.classList.add(state);
    label.textContent = text;
}

function addLog(message, type = 'info') {
    const container = els.logContainer();
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;

    const now = new Date();
    const time = now.toLocaleTimeString('pt-BR');

    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-message">${message}</span>
    `;

    // Add at the top
    container.insertBefore(entry, container.firstChild);

    // Limit entries
    while (container.children.length > 30) {
        container.removeChild(container.lastChild);
    }
}

function showToast(message, type = 'info') {
    const container = els.toastContainer();
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        success: '✓',
        error: '✗',
        info: 'ℹ',
        warning: '⚠'
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || 'ℹ'}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 4000);
}

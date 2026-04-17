/* ─────────────────────────────────────────────────────────────
   DreamForge — app.js
   Stable Diffusion v1.5 Web Panel
   ───────────────────────────────────────────────────────────── */

// ════════════════════════════════════════
// State
// ════════════════════════════════════════
const state = {
  selectedWidth:  512,
  selectedHeight: 512,
  isGenerating:   false,
  modelReady:     false,
  pollInterval:   null,
  progressTimer:  null,
  imageCount:     0,
};

// ════════════════════════════════════════
// DOM refs
// ════════════════════════════════════════
const dom = {
  statusDot:       document.getElementById('statusDot'),
  statusText:      document.getElementById('statusText'),
  promptInput:     document.getElementById('promptInput'),
  negativeInput:   document.getElementById('negativeInput'),
  stepsRange:      document.getElementById('stepsRange'),
  stepsVal:        document.getElementById('stepsVal'),
  cfgRange:        document.getElementById('cfgRange'),
  cfgVal:          document.getElementById('cfgVal'),
  seedInput:       document.getElementById('seedInput'),
  randomSeedBtn:   document.getElementById('randomSeedBtn'),
  generateBtn:     document.getElementById('generateBtn'),
  generateBtnText: document.getElementById('generateBtnText'),
  galleryArea:     document.getElementById('galleryArea'),
  progressCard:    document.getElementById('progressCard'),
  progressLabel:   document.getElementById('progressLabel'),
  progressFill:    document.getElementById('progressFill'),
  emptyState:      document.getElementById('emptyState'),
  toastContainer:  document.getElementById('toastContainer'),
};

// ════════════════════════════════════════
// Init
// ════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  bindRanges();
  bindSizeButtons();
  bindSeedRandom();
  bindGenerateButton();
  restoreHistory();
  startStatusPolling();
});

// ════════════════════════════════════════
// Ranges
// ════════════════════════════════════════
function bindRanges() {
  dom.stepsRange.addEventListener('input', () => {
    dom.stepsVal.textContent = dom.stepsRange.value;
  });
  dom.cfgRange.addEventListener('input', () => {
    dom.cfgVal.textContent = parseFloat(dom.cfgRange.value).toFixed(1);
  });
}

// ════════════════════════════════════════
// Size buttons
// ════════════════════════════════════════
function bindSizeButtons() {
  document.querySelectorAll('.size-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.size-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.selectedWidth  = parseInt(btn.dataset.w);
      state.selectedHeight = parseInt(btn.dataset.h);
    });
  });
}

// ════════════════════════════════════════
// Seed random
// ════════════════════════════════════════
function bindSeedRandom() {
  dom.randomSeedBtn.addEventListener('click', () => {
    dom.seedInput.value = Math.floor(Math.random() * 2147483647);
  });
}

// ════════════════════════════════════════
// Status polling
// ════════════════════════════════════════
function startStatusPolling() {
  checkStatus();
  state.pollInterval = setInterval(checkStatus, 3000);
}

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();
    updateStatusUI(data);

    if (data.status === 'ready' || data.status === 'error') {
      clearInterval(state.pollInterval);
      state.pollInterval = null;
    }
  } catch {
    // server not ready yet, keep polling
  }
}

function updateStatusUI(data) {
  const dot  = dom.statusDot;
  const text = dom.statusText;

  dot.className = 'status-dot';

  switch (data.status) {
    case 'loading':
      dot.classList.add('loading');
      text.textContent = 'Загрузка модели...';
      state.modelReady = false;
      setGenerateEnabled(false, 'Ожидание модели...');
      break;

    case 'ready':
      dot.classList.add('ready');
      const gpuInfo = data.gpu_name ? `GPU: ${data.gpu_name}` : 'CPU режим';
      text.textContent = `Готово · ${gpuInfo}`;
      state.modelReady = true;
      setGenerateEnabled(true, '✦ Сгенерировать');
      break;

    case 'error':
      dot.classList.add('error');
      text.textContent = 'Ошибка модели';
      state.modelReady = false;
      setGenerateEnabled(false, 'Ошибка модели');
      showToast('Модель не загрузилась: ' + (data.error || 'неизвестная ошибка'), 'error');
      break;
  }
}

function setGenerateEnabled(enabled, text) {
  dom.generateBtn.disabled = !enabled || state.isGenerating;
  dom.generateBtnText.textContent = text;
}

// ════════════════════════════════════════
// Generate
// ════════════════════════════════════════
function bindGenerateButton() {
  dom.generateBtn.addEventListener('click', handleGenerate);

  // Ctrl+Enter shortcut
  dom.promptInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleGenerate();
  });
}

async function handleGenerate() {
  const prompt = dom.promptInput.value.trim();
  if (!prompt) {
    showToast('Введите описание изображения', 'info');
    dom.promptInput.focus();
    return;
  }
  if (!state.modelReady || state.isGenerating) return;

  // Build request body
  const body = {
    prompt,
    negative_prompt: dom.negativeInput.value.trim(),
    steps:           parseInt(dom.stepsRange.value),
    cfg_scale:       parseFloat(dom.cfgRange.value),
    width:           state.selectedWidth,
    height:          state.selectedHeight,
    seed:            parseInt(dom.seedInput.value) || -1,
  };

  setGenerating(true);
  showProgress(body.steps);

  try {
    const res = await fetch('/api/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });

    clearProgress();

    if (res.status === 429) {
      showToast('Другая генерация уже идёт, попробуйте позже', 'info');
      return;
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Неизвестная ошибка' }));
      throw new Error(err.detail);
    }

    const data = await res.json();
    addImageCard(data, body);
    dom.seedInput.value = data.seed_used;   // show actual seed used
    showToast(`Готово за ${data.time_seconds}с`, 'success');

  } catch (err) {
    clearProgress();
    showToast('Ошибка: ' + err.message, 'error');
  } finally {
    setGenerating(false);
  }
}

// ════════════════════════════════════════
// Progress animation
// ════════════════════════════════════════
function showProgress(steps) {
  dom.emptyState.style.display = 'none';
  dom.progressCard.classList.add('visible');

  // Fake progress bar — fills over estimated time
  // rough estimate: ~0.5s per step on GPU, much more on CPU
  const estimatedMs = steps * 600;
  let elapsed = 0;
  const interval = 200;

  clearInterval(state.progressTimer);
  dom.progressFill.style.width = '0%';

  state.progressTimer = setInterval(() => {
    elapsed += interval;
    const pct = Math.min(95, (elapsed / estimatedMs) * 100);
    dom.progressFill.style.width = pct + '%';
    dom.progressLabel.textContent = `Генерация... ${Math.round(pct)}%`;
  }, interval);
}

function clearProgress() {
  clearInterval(state.progressTimer);
  dom.progressFill.style.width = '100%';
  setTimeout(() => {
    dom.progressCard.classList.remove('visible');
    dom.progressFill.style.width = '0%';
  }, 400);
}

// ════════════════════════════════════════
// Image card
// ════════════════════════════════════════
function addImageCard(data, body) {
  state.imageCount++;

  // Remove empty state
  const empty = document.getElementById('emptyState');
  if (empty) empty.remove();

  const card = document.createElement('div');
  card.className = 'image-card';
  card.id = `img-card-${state.imageCount}`;

  const imgSrc = `data:image/png;base64,${data.image_base64}`;

  const deviceClass = data.device === 'cuda' ? 'device-gpu' : 'device-cpu';
  const deviceLabel = data.device === 'cuda' ? '⚡ GPU' : '💻 CPU';

  card.innerHTML = `
    <img src="${imgSrc}"
         alt="Сгенерированное изображение: ${escapeHtml(body.prompt)}"
         loading="lazy" />
    <div class="image-card-footer">
      <div class="image-meta">
        <div class="image-prompt">${escapeHtml(body.prompt)}</div>
        <div class="image-tags">
          <span class="tag">${body.width}×${body.height}</span>
          <span class="tag">${body.steps} steps</span>
          <span class="tag">CFG ${body.cfg_scale}</span>
          <span class="tag">🌱 ${data.seed_used}</span>
          <span class="tag ${deviceClass}">${deviceLabel}</span>
          <span class="tag">⏱ ${data.time_seconds}s</span>
        </div>
      </div>
      <button class="btn-download" data-img="${imgSrc}" data-seed="${data.seed_used}"
              aria-label="Скачать изображение">
        ⬇ Скачать
      </button>
    </div>
  `;

  // Insert after progress card
  dom.progressCard.insertAdjacentElement('afterend', card);

  // Bind download
  card.querySelector('.btn-download').addEventListener('click', (e) => {
    const a = document.createElement('a');
    a.href = e.currentTarget.dataset.img;
    a.download = `dreamforge_${e.currentTarget.dataset.seed}.png`;
    a.click();
  });

  // Save to session storage
  saveToHistory({ imgSrc, data, body });
}

// ════════════════════════════════════════
// Session history (sessionStorage)
// ════════════════════════════════════════
function saveToHistory(entry) {
  try {
    const history = JSON.parse(sessionStorage.getItem('df_history') || '[]');
    history.unshift(entry);
    // Keep last 10
    sessionStorage.setItem('df_history', JSON.stringify(history.slice(0, 10)));
  } catch { /* sessionStorage full or unavailable */ }
}

function restoreHistory() {
  try {
    const history = JSON.parse(sessionStorage.getItem('df_history') || '[]');
    if (!history.length) return;

    // Remove empty state from DOM first
    const empty = document.getElementById('emptyState');
    if (empty) empty.remove();

    // Re-render in reverse so newest is on top
    history.reverse().forEach(entry => {
      addImageCard(entry.data, entry.body);
    });
    showToast(`Восстановлено ${history.length} из истории`, 'info');
  } catch { }
}

// ════════════════════════════════════════
// Generating state
// ════════════════════════════════════════
function setGenerating(flag) {
  state.isGenerating = flag;
  dom.generateBtn.disabled = flag || !state.modelReady;
  dom.generateBtnText.textContent = flag ? 'Генерация...' : '✦ Сгенерировать';
}

// ════════════════════════════════════════
// Toast notifications
// ════════════════════════════════════════
const TOAST_ICONS = { success: '✅', error: '❌', info: 'ℹ️' };

function showToast(message, type = 'info', duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${TOAST_ICONS[type] || 'ℹ️'}</span> <span>${escapeHtml(message)}</span>`;
  dom.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast-out');
    toast.addEventListener('animationend', () => toast.remove());
  }, duration);
}

// ════════════════════════════════════════
// Utils
// ════════════════════════════════════════
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

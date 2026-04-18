/* ══════════════════════════════════════════════════════
   DreamForge — app.js
   Stable Diffusion v1.5 Web Panel
   ══════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────
const state = {
  selectedWidth:     512,
  selectedHeight:    512,
  isGenerating:      false,
  modelReady:        false,
  pollInterval:      null,
  progressTimer:     null,
  imageCount:        0,
  activeModel:       null,
  activeLora:        null,
  activeVae:         null,
  activeEmbeddings:  [],     // list of active embedding filenames
};

// ── DOM refs ───────────────────────────────────────────
const $ = id => document.getElementById(id);

const dom = {
  statusDot:        $('statusDot'),
  statusText:       $('statusText'),
  currentModelName: $('currentModelName'),
  currentLoraName:  $('currentLoraName'),
  modelPill:        $('modelPill'),
  loraPill:         $('loraPill'),
  promptInput:      $('promptInput'),
  negativeInput:    $('negativeInput'),
  stepsRange:       $('stepsRange'),
  stepsVal:         $('stepsVal'),
  cfgRange:         $('cfgRange'),
  cfgVal:           $('cfgVal'),
  seedInput:        $('seedInput'),
  randomSeedBtn:    $('randomSeedBtn'),
  generateBtn:      $('generateBtn'),
  generateBtnText:  $('generateBtnText'),
  galleryArea:      $('galleryArea'),
  progressCard:     $('progressCard'),
  progressLabel:    $('progressLabel'),
  progressFill:     $('progressFill'),
  emptyState:       $('emptyState'),
  toastContainer:   $('toastContainer'),
  loraStrengthRange:       $('loraStrengthRange'),
  loraStrengthVal:         $('loraStrengthVal'),
  unloadLoraBtn:           $('unloadLoraBtn'),
  unloadVaeBtn:            $('unloadVaeBtn'),
  embedPill:               $('embedPill'),
  embedPillText:           $('embedPillText'),
  unloadAllEmbeddingsBtn:  $('unloadAllEmbeddingsBtn'),
};

// ── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  bindRanges();
  bindSizeButtons();
  dom.randomSeedBtn.addEventListener('click', () => {
    dom.seedInput.value = Math.floor(Math.random() * 2147483647);
  });
  dom.generateBtn.addEventListener('click', handleGenerate);
  dom.promptInput.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleGenerate();
  });

  // Upload bindings
  bindUpload('model');
  bindUpload('lora');
  bindUpload('vae');
  bindUpload('embedding');

  // Drag & drop on upload zones
  bindDragDrop('uploadZoneModel',     'model');
  bindDragDrop('uploadZoneLora',      'lora');
  bindDragDrop('uploadZoneVae',       'vae');
  bindDragDrop('uploadZoneEmbedding', 'embedding');

  restoreHistory();
  startStatusPolling();
});

// ── Tabs ───────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.id === `tab-${name}`);
    b.setAttribute('aria-selected', b.id === `tab-${name}`);
  });
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('hidden', p.id !== `panel-${name}`);
  });

  // Lazy-load file lists when visiting tabs
  if (name === 'models') loadModelsList();
  if (name === 'addons') { loadLorasList(); loadVaeList(); loadEmbeddingsList(); }
}

// ── Ranges ─────────────────────────────────────────────
function bindRanges() {
  dom.stepsRange.addEventListener('input', () => dom.stepsVal.textContent = dom.stepsRange.value);
  dom.cfgRange.addEventListener('input',   () => dom.cfgVal.textContent   = parseFloat(dom.cfgRange.value).toFixed(1));
  dom.loraStrengthRange.addEventListener('input', () =>
    dom.loraStrengthVal.textContent = parseFloat(dom.loraStrengthRange.value).toFixed(2));
}

// ── Size buttons ───────────────────────────────────────
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

// ── Status polling ─────────────────────────────────────
function startStatusPolling() {
  checkStatus();
  state.pollInterval = setInterval(checkStatus, 3000);
}

async function checkStatus() {
  try {
    const res  = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();
    updateStatusUI(data);
    if (data.status === 'ready' || data.status === 'error') {
      clearInterval(state.pollInterval);
      state.pollInterval = null;
    }
  } catch { /* keep polling */ }
}

function updateStatusUI(data) {
  const { statusDot, statusText } = dom;
  statusDot.className = 'status-dot';

  if (data.status === 'loading') {
    statusDot.classList.add('loading');
    statusText.textContent = 'Загрузка модели...';
    state.modelReady = false;
    setGenerateEnabled(false, 'Ожидание модели...');
  } else if (data.status === 'ready') {
    statusDot.classList.add('ready');
    statusText.textContent = data.gpu_name ? `GPU: ${data.gpu_name}` : 'CPU режим';
    state.modelReady = true;
    setGenerateEnabled(true, '✦ Сгенерировать');
  } else if (data.status === 'error') {
    statusDot.classList.add('error');
    statusText.textContent = 'Ошибка модели';
    state.modelReady = false;
    setGenerateEnabled(false, 'Ошибка');
    showToast('Модель не загрузилась: ' + (data.error || '?'), 'error');
  }

  // Update model pill
  if (data.current_model) {
    dom.currentModelName.textContent = data.current_model;
    state.activeModel = data.current_model;
  }

  // LoRA pill
  if (data.active_lora) {
    dom.currentLoraName.textContent = data.active_lora;
    dom.loraPill.style.display = 'flex';
    dom.unloadLoraBtn.style.display = '';
    state.activeLora = data.active_lora;
  } else {
    dom.loraPill.style.display = 'none';
    dom.unloadLoraBtn.style.display = 'none';
    state.activeLora = null;
  }

  // VAE
  if (data.active_vae) {
    dom.unloadVaeBtn.style.display = '';
    state.activeVae = data.active_vae;
  } else {
    dom.unloadVaeBtn.style.display = 'none';
    state.activeVae = null;
  }

  // Embeddings pill
  const embeds = data.active_embeddings || [];
  state.activeEmbeddings = embeds;
  if (embeds.length > 0) {
    dom.embedPill.style.display = 'flex';
    dom.embedPillText.textContent = embeds.length === 1
      ? embeds[0].replace(/\.[^.]+$/, '')
      : `${embeds.length} embeddings`;
    dom.unloadAllEmbeddingsBtn.style.display = '';
  } else {
    dom.embedPill.style.display = 'none';
    dom.unloadAllEmbeddingsBtn.style.display = 'none';
  }
}

// ── File lists ─────────────────────────────────────────
async function loadModelsList() {
  const container = $('modelsList');
  const res  = await safeFetch('/api/models/list');
  if (!res) return;
  renderFileList(container, res.models, 'model', state.activeModel);
}

async function loadLorasList() {
  const container = $('lorasList');
  const res  = await safeFetch('/api/loras/list');
  if (!res) return;
  renderFileList(container, res.loras, 'lora', state.activeLora);
}

async function loadVaeList() {
  const container = $('vaeList');
  const res  = await safeFetch('/api/vae/list');
  if (!res) return;
  renderFileList(container, res.vaes, 'vae', state.activeVae);
}

async function loadEmbeddingsList() {
  const container = $('embeddingsList');
  const res = await safeFetch('/api/embeddings/list');
  if (!res) return;
  renderEmbeddingList(container, res.embeddings, state.activeEmbeddings);
}

function renderFileList(container, files, type, activeFile) {
  if (!files || !files.length) {
    container.innerHTML = '<div class="file-list-empty">Нет файлов</div>';
    return;
  }
  container.innerHTML = '';
  files.forEach(f => {
    const isActive = f.name === activeFile;
    const item = document.createElement('div');
    item.className = 'file-item' + (isActive ? ' active' : '');
    item.innerHTML = `
      <span class="file-item-name" title="${escHtml(f.name)}">${escHtml(f.name)}</span>
      <span class="file-item-size">${f.size_mb} MB</span>
      <button class="btn-load ${isActive ? 'active-btn' : ''}"
              onclick="loadFile('${type}','${escHtml(f.name)}', this)">
        ${isActive ? '✓ Активна' : 'Загрузить'}
      </button>
    `;
    container.appendChild(item);
  });
}

// ── Load model / lora / vae by filename ───────────────
async function loadFile(type, filename, btn) {
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spin"></span>';

  try {
    let url, body;
    if (type === 'model') {
      url  = '/api/models/load';
      body = { filename };
      showToast(`Переключение на ${filename}... (может занять минуту)`, 'info', 8000);
    } else if (type === 'lora') {
      url  = '/api/loras/load';
      body = { filename, strength: parseFloat(dom.loraStrengthRange.value) };
      showToast(`Загружаю LoRA: ${filename}`, 'info');
    } else if (type === 'vae') {
      url  = '/api/vae/load';
      body = { filename };
      showToast(`Загружаю VAE: ${filename}`, 'info');
    }

    const res  = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Ошибка');

    showToast(`${type === 'model' ? 'Модель' : type === 'lora' ? 'LoRA' : 'VAE'} загружена: ${filename}`, 'success');

    // Refresh state
    await checkStatus();
    if (type === 'model')  { loadModelsList(); startStatusPolling(); }
    if (type === 'lora')   loadLorasList();
    if (type === 'vae')    loadVaeList();

  } catch (err) {
    showToast('Ошибка: ' + err.message, 'error');
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

// ── Unload LoRA / VAE / Embeddings ────────────────────
async function unloadLora() {
  const res = await safeFetchPost('/api/loras/load', { filename: '', strength: 0 });
  if (res) { showToast('LoRA выгружена', 'success'); await checkStatus(); loadLorasList(); }
}

async function unloadVae() {
  const res = await safeFetchPost('/api/vae/load', { filename: '' });
  if (res) { showToast('VAE снят', 'success'); await checkStatus(); loadVaeList(); }
}

async function toggleEmbedding(filename, btn) {
  const isActive = state.activeEmbeddings.includes(filename);
  const origHTML = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spin"></span>';
  try {
    if (isActive) {
      // Unload this embedding
      const res = await safeFetchPost('/api/embeddings/unload', { filename });
      if (res) showToast(`Embedding выгружен: ${filename}`, 'success');
    } else {
      // Load this embedding
      const res = await safeFetchPost('/api/embeddings/load', { filename });
      if (res) {
        const token = res.token || filename.replace(/\.[^.]+$/, '');
        showToast(`Embedding загружен: <${token}>. Используй в промпте!`, 'success', 6000);
      }
    }
    await checkStatus();
    loadEmbeddingsList();
  } catch (err) {
    showToast('Ошибка: ' + err.message, 'error');
    btn.disabled = false;
    btn.innerHTML = origHTML;
  }
}

async function unloadAllEmbeddings() {
  const res = await safeFetchPost('/api/embeddings/unload-all', {});
  if (res) { showToast('Все Embeddings выгружены', 'success'); await checkStatus(); loadEmbeddingsList(); }
}

// ── Upload ─────────────────────────────────────────────
function triggerUpload(type) {
  $(`fileInput${capitalize(type)}`).click();
}

function bindUpload(type) {
  const input = $(`fileInput${capitalize(type)}`);
  if (!input) return;
  input.addEventListener('change', e => {
    if (e.target.files[0]) uploadFile(type, e.target.files[0]);
    input.value = '';
  });
}

function bindDragDrop(zoneId, type) {
  const zone = $(zoneId);
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(type, file);
  });
}

async function uploadFile(type, file) {
  const progressWrap = $(`uploadProgress${capitalize(type)}`);
  const bar          = $(`uploadBar${capitalize(type)}`);
  const label        = $(`uploadLabel${capitalize(type)}`);

  progressWrap.classList.remove('hidden');
  bar.style.setProperty('--pct', '0%');
  label.textContent = '0%';

  const formData = new FormData();
  formData.append('file', file);

  try {
    // Use XMLHttpRequest for upload progress
    await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `/api/upload/${type}`);

      xhr.upload.addEventListener('progress', e => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          bar.style.setProperty('--pct', pct + '%');
          label.textContent = pct + '%';
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          const data = JSON.parse(xhr.responseText);
          showToast(`Загружено: ${data.filename} (${data.size_mb} MB)`, 'success');
          resolve(data);
        } else {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || 'Ошибка загрузки'));
        }
      });

      xhr.addEventListener('error', () => reject(new Error('Сетевая ошибка')));
      xhr.send(formData);
    });

    // Refresh the relevant list
    if (type === 'model')     loadModelsList();
    if (type === 'lora')      loadLorasList();
    if (type === 'vae')       loadVaeList();
    if (type === 'embedding') loadEmbeddingsList();

  } catch (err) {
    showToast('Ошибка загрузки: ' + err.message, 'error');
  } finally {
    setTimeout(() => progressWrap.classList.add('hidden'), 1500);
  }
}

// ── Generate ───────────────────────────────────────────
async function handleGenerate() {
  const prompt = dom.promptInput.value.trim();
  if (!prompt)                return (showToast('Введите описание', 'info'), dom.promptInput.focus());
  if (!state.modelReady || state.isGenerating) return;

  const body = {
    prompt,
    negative_prompt: dom.negativeInput.value.trim(),
    steps:           parseInt(dom.stepsRange.value),
    cfg_scale:       parseFloat(dom.cfgRange.value),
    width:           state.selectedWidth,
    height:          state.selectedHeight,
    seed:            parseInt(dom.seedInput.value) || -1,
    lora_strength:   parseFloat(dom.loraStrengthRange.value),
  };

  setGenerating(true);
  showProgress(body.steps);

  try {
    const res = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    clearProgress();

    if (res.status === 429) return showToast('Другая генерация идёт, подождите', 'info');
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Неизвестная ошибка' }));
      throw new Error(err.detail);
    }
    const data = await res.json();
    dom.seedInput.value = data.seed_used;
    addImageCard(data, body);
    showToast(`Готово за ${data.time_seconds}с`, 'success');
  } catch (err) {
    clearProgress();
    showToast('Ошибка: ' + err.message, 'error');
  } finally {
    setGenerating(false);
  }
}

// ── Progress ───────────────────────────────────────────
function showProgress(steps) {
  if (dom.emptyState) dom.emptyState.style.display = 'none';
  dom.progressCard.classList.add('visible');
  const estimatedMs = steps * 600;
  let elapsed = 0;
  clearInterval(state.progressTimer);
  dom.progressFill.style.width = '0%';
  state.progressTimer = setInterval(() => {
    elapsed += 200;
    const pct = Math.min(95, (elapsed / estimatedMs) * 100);
    dom.progressFill.style.width = pct + '%';
    dom.progressLabel.textContent = `Генерация... ${Math.round(pct)}%`;
  }, 200);
}

function clearProgress() {
  clearInterval(state.progressTimer);
  dom.progressFill.style.width = '100%';
  setTimeout(() => {
    dom.progressCard.classList.remove('visible');
    dom.progressFill.style.width = '0%';
  }, 400);
}

// ── Image card ─────────────────────────────────────────
function addImageCard(data, body) {
  state.imageCount++;
  const empty = document.getElementById('emptyState');
  if (empty) empty.remove();

  const card   = document.createElement('div');
  card.className = 'image-card';
  card.id = `img-card-${state.imageCount}`;

  const imgSrc     = `data:image/png;base64,${data.image_base64}`;
  const deviceCls  = data.device === 'cuda' ? 'device-gpu' : 'device-cpu';
  const deviceLbl  = data.device === 'cuda' ? '⚡ GPU' : '💻 CPU';

  const loraTags = data.lora ? `<span class="tag tag-lora">✧ ${escHtml(data.lora)}</span>` : '';
  const vaeTags  = data.vae  ? `<span class="tag tag-vae">◈ ${escHtml(data.vae)}</span>`  : '';

  card.innerHTML = `
    <img src="${imgSrc}" alt="${escHtml(body.prompt)}" loading="lazy" />
    <div class="image-card-footer">
      <div class="image-meta">
        <div class="image-prompt">${escHtml(body.prompt)}</div>
        <div class="image-tags">
          <span class="tag">${body.width}×${body.height}</span>
          <span class="tag">${body.steps} steps</span>
          <span class="tag">CFG ${body.cfg_scale}</span>
          <span class="tag">🌱 ${data.seed_used}</span>
          <span class="tag ${deviceCls}">${deviceLbl}</span>
          <span class="tag">⏱ ${data.time_seconds}s</span>
          ${loraTags}${vaeTags}
        </div>
      </div>
      <button class="btn-download" data-img="${imgSrc}" data-seed="${data.seed_used}">⬇ Скачать</button>
    </div>`;

  dom.progressCard.insertAdjacentElement('afterend', card);
  card.querySelector('.btn-download').addEventListener('click', e => {
    const a = document.createElement('a');
    a.href     = e.currentTarget.dataset.img;
    a.download = `dreamforge_${e.currentTarget.dataset.seed}.png`;
    a.click();
  });
  saveToHistory({ imgSrc, data, body });
}

// ── Session history ────────────────────────────────────
function saveToHistory(entry) {
  try {
    const h = JSON.parse(sessionStorage.getItem('df_history') || '[]');
    h.unshift(entry);
    sessionStorage.setItem('df_history', JSON.stringify(h.slice(0, 10)));
  } catch { }
}

function restoreHistory() {
  try {
    const h = JSON.parse(sessionStorage.getItem('df_history') || '[]');
    if (!h.length) return;
    const empty = document.getElementById('emptyState');
    if (empty) empty.remove();
    h.reverse().forEach(e => addImageCard(e.data, e.body));
    showToast(`Восстановлено ${h.length} из истории`, 'info');
  } catch { }
}

// ── Generating state ───────────────────────────────────
function setGenerating(flag) {
  state.isGenerating = flag;
  dom.generateBtn.disabled = flag || !state.modelReady;
  dom.generateBtnText.textContent = flag ? 'Генерация...' : '✦ Сгенерировать';
}

function setGenerateEnabled(enabled, text) {
  dom.generateBtn.disabled = !enabled || state.isGenerating;
  dom.generateBtnText.textContent = text;
}

// ── Toast ──────────────────────────────────────────────
const ICONS = { success: '✅', error: '❌', info: 'ℹ️' };
function showToast(msg, type = 'info', duration = 4000) {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${ICONS[type]}</span><span>${escHtml(msg)}</span>`;
  dom.toastContainer.appendChild(t);
  setTimeout(() => { t.classList.add('toast-out'); t.addEventListener('animationend', () => t.remove()); }, duration);
}

// ── Fetch helpers ──────────────────────────────────────
async function safeFetch(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

async function safeFetchPost(url, body) {
  try {
    const res = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
    return await res.json();
  } catch (err) { showToast('Ошибка: ' + err.message, 'error'); return null; }
}

// ── Utils ──────────────────────────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ── Embedding list renderer (multi-select toggle) ──────
function renderEmbeddingList(container, files, activeList) {
  if (!files || !files.length) {
    container.innerHTML = '<div class="file-list-empty">Нет Embedding файлов</div>';
    return;
  }
  container.innerHTML = '';
  files.forEach(f => {
    const isActive = activeList.includes(f.name);
    const token    = f.name.replace(/\.[^.]+$/, '');
    const item = document.createElement('div');
    item.className = 'file-item' + (isActive ? ' active' : '');
    item.id = `emb-item-${escHtml(f.name)}`;
    item.innerHTML = `
      <span class="file-item-name" title="${escHtml(f.name)}">${escHtml(f.name)}</span>
      <span class="file-item-size">${f.size_mb} MB</span>
      <div style="display:flex;gap:4px;align-items:center">
        <button class="btn-load btn-copy-token" title="Скопировать токен"
                onclick="navigator.clipboard.writeText('<${escHtml(token)}>')
                  .then(()=>showToast('Скопировано: <${escHtml(token)}>', 'success'))">
          📋
        </button>
        <button class="btn-load ${isActive ? 'active-btn' : ''}"
                id="emb-btn-${escHtml(f.name)}"
                onclick="toggleEmbedding('${escHtml(f.name)}', this)">
          ${isActive ? '✓ Активен' : '+ Включить'}
        </button>
      </div>`;
    container.appendChild(item);
  });
}

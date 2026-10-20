/* ── AI Family Photo Sorter — Frontend JavaScript ─────────────────── */
'use strict';

const API = '';  // same origin

// ─── State ───────────────────────────────────────────────────────────────────
let currentPage = 'setup';
let scanInterval = null;
let highPage = 1, reviewPage = 1;
const PAGE_SIZE = 24;
let identities = [];
let thresholdData = {};
let currentJobId = null;

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  loadHardwareStatus();
  loadIdentities();
  setInterval(loadHardwareStatus, 30000);
  setInterval(checkReadiness, 3000);
});

// ─── Navigation ───────────────────────────────────────────────────────────────
function initNav() {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      const page = el.dataset.page;
      navigateTo(page);
    });
  });
}

function navigateTo(page) {
  if (page === 'matched') page = 'gallery-high';
  if (page === 'review') page = 'gallery-review';
  currentPage = page;
  document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  const pageEl = document.getElementById(`page-${page}`);
  if (pageEl) pageEl.classList.remove('hidden');
  const navEl = document.getElementById(`nav-${page}`);
  if (navEl) navEl.classList.add('active');

  // Page-specific loads
  if (page === 'scan') {
    startStatusPolling();
  }
  if (page === 'gallery-high') {
    loadHighGallery(1);
  }
  if (page === 'gallery-review') {
    loadReviewGallery(1);
  }
  if (page === 'reports') {
    loadThresholds();
    loadJobSummary();
  }
}

// ─── Hardware Status ──────────────────────────────────────────────────────────
async function loadHardwareStatus() {
  try {
    const data = await apiFetch('/api/hardware');
    const hw = data.hardware;
    const model = data.model;
    const mem = data.memory;

    // Sidebar
    document.getElementById('hw-gpu').textContent = hw.gpu_name || 'N/A';
    document.getElementById('hw-model').textContent = model.loaded ? '✓ Ready' : (model.error ? '✗ Error' : 'Loading…');
    document.getElementById('hw-ram').textContent = `${mem.available_mb} MB free`;

    // Setup page model card
    document.getElementById('info-model-name').textContent = 'buffalo_l (InsightFace ArcFace)';
    document.getElementById('info-gpu').textContent = hw.gpu_name;
    document.getElementById('info-provider').textContent = hw.active_provider;
    document.getElementById('info-vram').textContent = hw.gpu_vram_mb ? `${hw.gpu_vram_mb} MB` : 'N/A';

    const badge = document.getElementById('model-badge');
    if (model.loaded) {
      badge.textContent = hw.gpu_available ? '✓ GPU Ready' : '✓ CPU Ready';
      badge.className = 'badge badge-success';
    } else if (model.error) {
      badge.textContent = '✗ Load Failed';
      badge.className = 'badge badge-error';
    } else {
      badge.textContent = '⟳ Loading Model…';
      badge.className = 'badge badge-info';
    }
  } catch (e) {
    console.error('Hardware fetch failed:', e);
  }
}

// ─── Identity Management ──────────────────────────────────────────────────────
async function loadIdentities() {
  try {
    identities = await apiFetch('/api/identities');
    renderIdentityGrid();
    checkReadiness();
  } catch (e) {
    console.error('Failed to load identities:', e);
  }
}

const IDENTITY_EMOJIS = { me: '🙋', mother: '👩', father: '👨', sister: '👧' };

function renderIdentityGrid() {
  const grid = document.getElementById('identity-grid');
  if (!grid) return;
  grid.innerHTML = '';

  identities.forEach(ident => {
    const card = document.createElement('div');
    card.className = `identity-card ${ident.ready ? 'ready' : ''}`;
    card.id = `identity-card-${ident.identity_id}`;
    card.innerHTML = `
      <div class="identity-header">
        <div class="flex gap-2" style="align-items:center;gap:10px;min-width:0;">
          <div class="identity-avatar" id="avatar-${ident.identity_id}">
            <span>${IDENTITY_EMOJIS[ident.identity_id] || '👤'}</span>
          </div>
          <div style="min-width:0;">
            <div class="identity-name" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${ident.display_name}</div>
            <div class="identity-sub">${ident.reference_count} reference${ident.reference_count !== 1 ? 's' : ''}</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
          ${ident.ready
            ? '<span class="badge badge-success">Ready</span>'
            : '<span class="badge badge-warning">No refs</span>'}
          <button class="identity-card-del-btn" onclick="deleteIdentity('${ident.identity_id}')" title="Delete ${ident.display_name}">✕</button>
        </div>
      </div>

      <label class="include-scan-label">
        <input type="checkbox" id="include-${ident.identity_id}" ${ident.is_enabled !== false ? 'checked' : ''} onchange="toggleIdentity('${ident.identity_id}')" />
        <span>Include in search</span>
      </label>

      <div class="ref-thumbnails" id="refs-${ident.identity_id}" style="margin-top:10px;">
        ${ident.reference_count === 0 ? '<div class="ref-thumb-empty">+</div>' : ''}
      </div>
      <label class="upload-label">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        Add Reference Photo
        <input type="file" class="upload-input" accept="image/*" multiple onchange="uploadReference('${ident.identity_id}', this)" />
      </label>
      <div class="identity-thresholds">
        <span class="threshold-pill">High: ${ident.high_threshold}</span>
        <span class="threshold-pill">Review: ${ident.review_threshold}</span>
      </div>
    `;
    grid.appendChild(card);

    // Load existing reference thumbnails
    loadReferenceThumbnails(ident.identity_id);
  });
}

async function loadReferenceThumbnails(identityId) {
  try {
    const refs = await apiFetch(`/api/identities/${identityId}/references`);
    const container = document.getElementById(`refs-${identityId}`);
    if (!container) return;
    container.innerHTML = '';

    if (refs.length === 0) {
      container.innerHTML = '<div class="ref-thumb-empty">+</div>';
      return;
    }

    refs.forEach(ref => {
      const wrap = document.createElement('div');
      wrap.className = 'ref-thumb-wrap';

      if (ref.thumbnail_url) {
        const img = document.createElement('img');
        img.className = 'ref-thumb';
        img.src = ref.thumbnail_url;
        img.title = `Quality: ${(ref.quality_score * 100).toFixed(0)}%\n${ref.quality_notes || ''}`;
        img.onerror = () => img.replaceWith(createRefPlaceholder(ref));
        wrap.appendChild(img);
      } else {
        wrap.appendChild(createRefPlaceholder(ref));
      }

      // Delete reference button
      const delBtn = document.createElement('button');
      delBtn.className = 'ref-del-btn';
      delBtn.innerHTML = '×';
      delBtn.title = 'Remove this photo';
      delBtn.onclick = (e) => deleteReferencePhoto(identityId, ref.id, e);
      wrap.appendChild(delBtn);

      container.appendChild(wrap);
    });
  } catch (e) {
    console.error(`Failed to load refs for ${identityId}:`, e);
  }
}

function createRefPlaceholder(ref) {
  const el = document.createElement('div');
  el.className = 'ref-thumb-empty';
  el.textContent = ref.face_detected ? '✓' : '✗';
  el.title = ref.quality_notes || '';
  el.style.color = ref.face_detected ? 'var(--green)' : 'var(--red)';
  return el;
}

async function deleteReferencePhoto(identityId, refId, event) {
  if (event) event.stopPropagation();
  if (!confirm('Remove this reference photo?')) return;
  try {
    await apiFetch(`/api/identities/${identityId}/references/${refId}`, 'DELETE');
    showToast('Reference photo removed.', 'info');
    await loadIdentities();
  } catch (e) {
    showToast(`Failed to remove photo: ${e.message}`, 'error');
  }
}

async function deleteIdentity(identityId) {
  const ident = identities.find(i => i.identity_id === identityId);
  const name = ident ? ident.display_name : identityId;
  if (!confirm(`Delete ${name}? All registered reference photos for this person will also be removed.`)) return;

  try {
    await apiFetch(`/api/identities/${identityId}`, 'DELETE');
    showToast(`Deleted ${name}.`, 'info');
    await loadIdentities();
  } catch (e) {
    showToast(`Failed to delete member: ${e.message}`, 'error');
  }
}

async function toggleIdentity(identityId) {
  try {
    await apiFetch(`/api/identities/${identityId}/toggle`, 'PATCH');
    const ident = identities.find(i => i.identity_id === identityId);
    if (ident) {
      const cb = document.getElementById(`include-${identityId}`);
      ident.is_enabled = cb ? cb.checked : true;
    }
    checkReadiness();
  } catch (e) {
    console.error('Failed to toggle identity:', e);
  }
}

function openAddMemberModal() {
  const modal = document.getElementById('add-member-modal');
  if (modal) modal.classList.remove('hidden');
  const input = document.getElementById('new-member-name');
  if (input) {
    input.value = '';
    setTimeout(() => input.focus(), 50);
  }
}

function closeAddMemberModal() {
  const modal = document.getElementById('add-member-modal');
  if (modal) modal.classList.add('hidden');
}

async function submitAddMember() {
  const nameInput = document.getElementById('new-member-name');
  const name = nameInput?.value?.trim();
  if (!name) {
    showToast('Please enter a name for the family member.', 'warning');
    return;
  }
  const high = parseFloat(document.getElementById('new-member-high')?.value || '0.45');
  const review = parseFloat(document.getElementById('new-member-review')?.value || '0.35');

  try {
    const res = await apiFetch('/api/identities', 'POST', {
      display_name: name,
      high_threshold: isNaN(high) ? 0.45 : high,
      review_threshold: isNaN(review) ? 0.35 : review,
    });
    if (res.success) {
      showToast(`✓ Added ${name}!`, 'success');
      closeAddMemberModal();
      await loadIdentities();
    }
  } catch (e) {
    showToast(`Failed to add member: ${e.message}`, 'error');
  }
}

async function uploadReference(identityId, inputEl) {
  const files = Array.from(inputEl.files);
  if (!files.length) return;

  for (const file of files) {
    const formData = new FormData();
    formData.append('file', file);

    showToast(`Uploading ${file.name}…`, 'info');

    try {
      const result = await fetch(`${API}/api/identities/${identityId}/references`, {
        method: 'POST',
        body: formData,
      }).then(r => r.json());

      if (result.success) {
        const notes = result.quality_notes?.length
          ? `\nNotes: ${result.quality_notes.join(', ')}`
          : '';
        showToast(`✓ Face registered for ${identityId}. Quality: ${(result.quality_score * 100).toFixed(0)}%${notes}`, 'success');
      } else {
        showToast(`⚠ ${result.message}`, 'warning');
      }
    } catch (e) {
      showToast(`✗ Upload failed: ${e.message}`, 'error');
    }
  }

  // Reload
  await loadIdentities();
}

// ─── Readiness Check ──────────────────────────────────────────────────────────
async function checkReadiness() {
  const container = document.getElementById('readiness-check');
  if (!container) return;

  const hw = await apiFetch('/api/hardware').catch(() => null);
  const checks = [];

  if (hw) {
    checks.push({ ok: hw.model.loaded, label: `Face recognition model: ${hw.model.loaded ? 'Ready' : 'Loading…'}` });
    checks.push({ ok: hw.hardware.gpu_available, label: `GPU acceleration: ${hw.hardware.gpu_available ? 'Available (' + hw.hardware.active_provider + ')' : 'CPU mode (slower)'}` });
  }

  const sourceVal = document.getElementById('source-folder')?.value?.trim();
  checks.push({ ok: !!sourceVal, label: `Source folder: ${sourceVal ? 'Set' : 'Not configured'}` });

  const selected = identities.filter(i => {
    const cb = document.getElementById(`include-${i.identity_id}`);
    return cb ? cb.checked : (i.is_enabled !== false);
  });
  const readySelected = selected.filter(i => i.ready);

  checks.push({
    ok: readySelected.length > 0,
    label: selected.length === 0
      ? 'No family members selected for search'
      : `Selected for search: ${readySelected.length} of ${selected.length} ready with photos`
  });

  container.innerHTML = checks.map(c =>
    `<div class="readiness-item ${c.ok ? 'ok' : 'warn'}">${c.ok ? '✓' : '⚠'} ${c.label}</div>`
  ).join('');

  const allReady = checks.filter(c => c.label.includes('model') || c.label.includes('Source') || c.label.includes('Selected')).every(c => c.ok);
  const startBtn = document.getElementById('start-scan-btn');
  if (startBtn) startBtn.disabled = !allReady;
}

// ─── Folder Browsing ──────────────────────────────────────────────────────────
let fbTargetInputId = null;
let fbCurrentPath = '';
let fbCurrentParent = null;
let fbAllSubfolders = [];
let fbQuickShortcutsLoaded = false;

async function browseFolder(inputId) {
  fbTargetInputId = inputId;
  const modal = document.getElementById('folder-browser-modal');
  if (!modal) return;

  const titleMap = {
    'source-folder': '📁 Select Source Folder (Wedding Photos)',
    'dest-high': '✅ Select Destination A (High-Confidence Matches)',
    'dest-review': '🔍 Select Destination B (Manual Review Candidates)',
  };
  const titleEl = document.getElementById('folder-browser-title');
  if (titleEl) titleEl.textContent = titleMap[inputId] || 'Browse Folder';

  modal.classList.remove('hidden');

  if (!fbQuickShortcutsLoaded) {
    await fbLoadQuickShortcuts();
  }

  const existingVal = document.getElementById(inputId)?.value?.trim();
  await fbNavigatePath(existingVal || '');
}

function closeFolderBrowser() {
  const modal = document.getElementById('folder-browser-modal');
  if (modal) modal.classList.add('hidden');
}

async function fbLoadQuickShortcuts() {
  try {
    const data = await apiFetch('/api/browse/quick-paths');
    const container = document.getElementById('fb-quick-shortcuts');
    if (!container) return;
    container.innerHTML = '';

    // Render drive buttons
    (data.drives || []).forEach(drive => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'fb-shortcut-btn';
      btn.innerHTML = `💾 <strong>${drive.path}</strong> ${drive.free_gb ? '<span style="color:var(--text-muted);font-size:0.7rem;">(' + drive.free_gb + ' GB)</span>' : ''}`;
      btn.onclick = () => fbNavigatePath(drive.path);
      container.appendChild(btn);
    });

    // Render system folder shortcuts
    (data.shortcuts || []).forEach(sc => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'fb-shortcut-btn';
      btn.innerHTML = `${sc.icon || '📁'} ${sc.name}`;
      btn.onclick = () => fbNavigatePath(sc.path);
      container.appendChild(btn);
    });

    fbQuickShortcutsLoaded = true;
  } catch (e) {
    console.error('Failed to load quick shortcuts:', e);
  }
}

async function fbNavigatePath(targetPath) {
  const statusMsg = document.getElementById('fb-status-msg');
  const pathInput = document.getElementById('fb-current-path');

  if (statusMsg) {
    statusMsg.style.color = 'var(--text-muted)';
    statusMsg.textContent = 'Loading folder…';
  }

  try {
    const res = await apiFetch('/api/browse/list', 'POST', { path: targetPath });

    fbCurrentPath = res.current_path;
    fbCurrentParent = res.parent_path;
    fbAllSubfolders = res.subfolders || [];

    if (pathInput) pathInput.value = fbCurrentPath;

    const btnUp = document.getElementById('fb-btn-up');
    if (btnUp) btnUp.disabled = !fbCurrentParent;

    const preview = document.getElementById('fb-selection-preview');
    if (preview) {
      preview.innerHTML = `Selected: <strong style="color:var(--accent-light);">${fbCurrentPath}</strong>`;
      preview.title = fbCurrentPath;
    }

    if (res.error) {
      if (statusMsg) {
        statusMsg.style.color = 'var(--amber)';
        statusMsg.textContent = `⚠ ${res.error}`;
      }
    } else {
      if (statusMsg) {
        statusMsg.style.color = 'var(--text-muted)';
        statusMsg.textContent = `${fbAllSubfolders.length} subfolder${fbAllSubfolders.length !== 1 ? 's' : ''}`;
      }
    }

    const filterInput = document.getElementById('fb-filter-input');
    if (filterInput) filterInput.value = '';

    fbRenderFolderList(fbAllSubfolders);
  } catch (e) {
    if (statusMsg) {
      statusMsg.style.color = 'var(--red)';
      statusMsg.textContent = `Error: ${e.message}`;
    }
  }
}

function fbRenderFolderList(folders) {
  const container = document.getElementById('fb-folder-list');
  if (!container) return;
  container.innerHTML = '';

  if (!folders || folders.length === 0) {
    container.innerHTML = `
      <div style="padding: 2.5rem 1rem; text-align: center; color: var(--text-muted); font-size: 0.85rem;">
        No subfolders found in this directory.
      </div>
    `;
    return;
  }

  folders.forEach(f => {
    const item = document.createElement('div');
    item.className = 'fb-item';
    item.innerHTML = `
      <span style="font-size: 1.1rem; line-height: 1;">📁</span>
      <span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${f.name}</span>
      <span style="font-size: 0.72rem; color: var(--text-muted); opacity: 0.8;">Open ➔</span>
    `;

    // Click to enter directory
    item.onclick = () => fbNavigatePath(f.path);
    container.appendChild(item);
  });
}

function fbNavigateUp() {
  if (fbCurrentParent) {
    fbNavigatePath(fbCurrentParent);
  }
}

function fbFilterFolders(query) {
  const q = (query || '').toLowerCase().trim();
  if (!q) {
    fbRenderFolderList(fbAllSubfolders);
    return;
  }
  const filtered = fbAllSubfolders.filter(f => f.name.toLowerCase().includes(q));
  fbRenderFolderList(filtered);
}

async function fbOpenCreateFolderPrompt() {
  if (!fbCurrentPath) {
    showToast('Please select a parent folder first.', 'warning');
    return;
  }
  const name = prompt(`Create new folder inside:\n${fbCurrentPath}\n\nEnter folder name:`);
  if (!name || !name.trim()) return;

  try {
    const res = await apiFetch('/api/browse/create-folder', 'POST', {
      parent_path: fbCurrentPath,
      folder_name: name.trim(),
    });
    if (res.success && res.path) {
      showToast(`✓ Created folder: ${name}`, 'success');
      await fbNavigatePath(res.path);
    }
  } catch (e) {
    showToast(`Failed to create folder: ${e.message}`, 'error');
  }
}

function fbSelectCurrentFolder() {
  if (!fbTargetInputId || !fbCurrentPath) {
    closeFolderBrowser();
    return;
  }
  const input = document.getElementById(fbTargetInputId);
  if (input) {
    input.value = fbCurrentPath;
    validateFolderInputs();
    showToast(`✓ Set folder: ${fbCurrentPath}`, 'success');
  }
  closeFolderBrowser();
}

async function triggerNativeFolderPicker() {
  showToast('Opening Windows File Explorer picker…', 'info');
  try {
    const result = await apiFetch('/api/browse-folder', 'POST', {});
    if (!result.cancelled && result.path) {
      fbCurrentPath = result.path;
      fbSelectCurrentFolder();
    }
  } catch (e) {
    showToast('Native picker failed: ' + e.message, 'warning');
  }
}

async function validateFolderInputs() {
  const source = document.getElementById('source-folder')?.value?.trim();
  const high = document.getElementById('dest-high')?.value?.trim();
  const review = document.getElementById('dest-review')?.value?.trim();
  const msgEl = document.getElementById('folder-validation-msg');

  if (!source || !high || !review) {
    msgEl.className = 'validation-msg';
    return;
  }

  try {
    const result = await apiFetch('/api/validate-folders', 'POST', {
      source_folder: source,
      dest_folder_high: high,
      dest_folder_review: review,
    });
    if (result.valid) {
      msgEl.className = 'validation-msg success';
      msgEl.textContent = '✓ Folder paths are valid';
    } else {
      msgEl.className = 'validation-msg error';
      msgEl.textContent = `✗ ${result.error}`;
    }
  } catch (e) {
    msgEl.className = 'validation-msg';
  }
}

// Add listeners for folder inputs
document.addEventListener('DOMContentLoaded', () => {
  ['source-folder', 'dest-high', 'dest-review'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('blur', validateFolderInputs);
  });
});

// ─── Scan Control ─────────────────────────────────────────────────────────────
async function startScan() {
  const source = document.getElementById('source-folder')?.value?.trim();
  const high = document.getElementById('dest-high')?.value?.trim();
  const review = document.getElementById('dest-review')?.value?.trim();
  const rule = document.getElementById('matching-rule')?.value;
  const recursive = document.getElementById('recursive-scan')?.checked;

  if (!source || !high || !review) {
    showToast('Please configure all three folder paths.', 'error');
    return;
  }

  // Collect selected family members for this search
  const selectedIdentities = [];
  identities.forEach(ident => {
    const cb = document.getElementById(`include-${ident.identity_id}`);
    if (cb && cb.checked) {
      selectedIdentities.push(ident.identity_id);
    }
  });

  if (selectedIdentities.length === 0) {
    showToast('Please select at least one family member to search for.', 'error');
    return;
  }

  try {
    const result = await apiFetch('/api/scan/start', 'POST', {
      source_folder: source,
      dest_folder_high: high,
      dest_folder_review: review,
      matching_rule: rule,
      recursive_scan: recursive,
      selected_identities: selectedIdentities,
    });
    currentJobId = result.job_id;
    showToast('✓ Scan started!', 'success');
    navigateTo('scan');
    startStatusPolling();
  } catch (e) {
    showToast(`✗ Failed to start scan: ${e.message}`, 'error');
  }
}

async function pauseScan() {
  try {
    await apiFetch('/api/scan/pause', 'POST');
    showToast('Scan paused.', 'info');
    document.getElementById('btn-pause').classList.add('hidden');
    document.getElementById('btn-resume').classList.remove('hidden');
  } catch (e) {
    showToast(`Failed to pause: ${e.message}`, 'error');
  }
}

async function resumeScan() {
  try {
    await apiFetch('/api/scan/resume', 'POST');
    showToast('Scan resumed.', 'success');
    document.getElementById('btn-resume').classList.add('hidden');
    document.getElementById('btn-pause').classList.remove('hidden');
  } catch (e) {
    showToast(`Failed to resume: ${e.message}`, 'error');
  }
}

async function cancelScan() {
  if (!confirm('Cancel the scan? Progress will be saved.')) return;
  try {
    await apiFetch('/api/scan/cancel', 'POST');
    showToast('Scan cancelled.', 'warning');
    stopStatusPolling();
  } catch (e) {
    showToast(`Failed to cancel: ${e.message}`, 'error');
  }
}

// ─── Status Polling ───────────────────────────────────────────────────────────
function startStatusPolling() {
  stopStatusPolling();
  updateScanStatus();
  scanInterval = setInterval(updateScanStatus, 1500);
}
function stopStatusPolling() {
  if (scanInterval) { clearInterval(scanInterval); scanInterval = null; }
}

async function updateScanStatus() {
  try {
    const s = await apiFetch('/api/scan/status');
    updateDashboard(s);

    // Update badges
    document.getElementById('badge-high').textContent = s.high_confidence_matches || 0;
    document.getElementById('badge-review').textContent = s.review_candidates || 0;

    if (s.status === 'completed' || s.status === 'cancelled' || s.status === 'error') {
      stopStatusPolling();
      if (s.status === 'completed') {
        showToast(`✓ Scan complete! ${s.high_confidence_matches} matched, ${s.review_candidates} for review.`, 'success');
      }
    }
  } catch (e) {
    console.debug('Status poll failed:', e);
  }
}

function updateDashboard(s) {
  // Status dot
  const dot = document.getElementById('status-dot');
  dot.className = `scan-status-dot ${s.status === 'running' ? 'running' : s.status === 'paused' ? 'paused' : s.status === 'error' ? 'error' : ''}`;

  // Labels
  const statusLabels = { idle: 'Idle', running: 'Scanning…', paused: 'Paused', completed: 'Completed', cancelled: 'Cancelled', error: 'Error' };
  document.getElementById('scan-status-text').textContent = statusLabels[s.status] || s.status;
  document.getElementById('scan-current-file').textContent = s.current_file || '';

  // Progress
  document.getElementById('progress-bar').style.width = `${s.progress_pct || 0}%`;
  document.getElementById('progress-label').textContent =
    `${s.processed_files} / ${s.total_files} files (${s.progress_pct || 0}%)`;

  // Stats
  document.getElementById('stat-total').textContent = s.total_files || 0;
  document.getElementById('stat-processed').textContent = s.processed_files || 0;
  document.getElementById('stat-high').textContent = s.high_confidence_matches || 0;
  document.getElementById('stat-review').textContent = s.review_candidates || 0;
  document.getElementById('stat-skipped').textContent = s.skipped_files || 0;
  document.getElementById('stat-failed').textContent = s.failed_files || 0;

  // Performance
  // Performance & Prominent ETA
  document.getElementById('perf-speed').textContent = s.speed_fps > 0 ? `${s.speed_fps.toFixed(2)} img/s` : '—';
  document.getElementById('perf-ram').textContent = s.memory_available_mb ? `${s.memory_available_mb} MB` : '—';
  document.getElementById('perf-ram-pct').textContent = s.memory_percent ? `${s.memory_percent}%` : '—';

  const etaBadge = document.getElementById('scan-eta-badge');
  if (s.status === 'running') {
    if (s.eta_seconds && s.eta_seconds > 0) {
      const mins = Math.floor(s.eta_seconds / 60);
      const secs = Math.floor(s.eta_seconds % 60);
      const etaStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
      if (etaBadge) {
        etaBadge.style.color = 'var(--accent-light)';
        etaBadge.textContent = `⏱ Estimated Time to Finish: ~${etaStr} (${s.speed_fps > 0 ? s.speed_fps.toFixed(1) + ' img/s' : ''})`;
      }
      document.getElementById('perf-eta').textContent = etaStr;
    } else {
      if (etaBadge) etaBadge.textContent = '⏱ Estimated Time to Finish: Calculating…';
      document.getElementById('perf-eta').textContent = 'Calculating…';
    }
  } else if (s.status === 'completed') {
    if (etaBadge) {
      etaBadge.style.color = 'var(--green)';
      etaBadge.textContent = `✓ Scan Complete! Processed ${s.processed_files} photos.`;
    }
    document.getElementById('perf-eta').textContent = 'Finished';
  } else if (s.status === 'paused') {
    if (etaBadge) {
      etaBadge.style.color = 'var(--amber)';
      etaBadge.textContent = `⏸ Scan Paused — Click 'Resume' to continue`;
    }
    document.getElementById('perf-eta').textContent = 'Paused';
  } else if (s.status === 'cancelled') {
    if (etaBadge) {
      etaBadge.style.color = 'var(--red)';
      etaBadge.textContent = `⏹ Scan Stopped / Cancelled`;
    }
    document.getElementById('perf-eta').textContent = 'Stopped';
  } else {
    if (etaBadge) etaBadge.textContent = '⏱ Estimated Time to Finish: —';
    document.getElementById('perf-eta').textContent = '—';
  }

  // Scan control buttons visibility (Pause, Resume, Stop, View Matches, Start New Scan)
  const btnPause = document.getElementById('btn-pause');
  const btnResume = document.getElementById('btn-resume');
  const btnCancel = document.getElementById('btn-cancel');
  const btnViewMatches = document.getElementById('btn-view-matches');
  const btnNewScan = document.getElementById('btn-new-scan');

  if (s.status === 'running') {
    btnPause?.classList.remove('hidden');
    btnResume?.classList.add('hidden');
    btnCancel?.classList.remove('hidden');
    btnViewMatches?.classList.add('hidden');
    btnNewScan?.classList.add('hidden');
  } else if (s.status === 'paused') {
    btnPause?.classList.add('hidden');
    btnResume?.classList.remove('hidden');
    btnCancel?.classList.remove('hidden');
    btnViewMatches?.classList.add('hidden');
    btnNewScan?.classList.add('hidden');
  } else if (s.status === 'completed' || s.status === 'cancelled' || s.status === 'error') {
    // When scan operations end, disappear pause/resume/stop buttons and show finished actions!
    btnPause?.classList.add('hidden');
    btnResume?.classList.add('hidden');
    btnCancel?.classList.add('hidden');
    if (btnViewMatches) {
      btnViewMatches.classList.remove('hidden');
      btnViewMatches.textContent = `👁 View Matched Photos (${s.high_confidence_matches || 0})`;
    }
    btnNewScan?.classList.remove('hidden');
  } else {
    // idle
    btnPause?.classList.remove('hidden');
    btnResume?.classList.add('hidden');
    btnCancel?.classList.add('hidden');
    btnViewMatches?.classList.add('hidden');
    btnNewScan?.classList.add('hidden');
  }
}

// ─── High-Confidence Gallery ──────────────────────────────────────────────────
let availableScanJobs = [];

async function populateJobFilter() {
  try {
    const jobs = await apiFetch('/api/reports/summary');
    availableScanJobs = jobs || [];
    const select = document.getElementById('high-job-filter');
    if (!select) return;

    const currentVal = select.value;
    select.innerHTML = '<option value="">All Scans</option>';

    availableScanJobs.forEach(j => {
      const opt = document.createElement('option');
      opt.value = j.job_id;
      const srcName = j.source_folder ? j.source_folder.split(/[\\/]/).filter(Boolean).pop() : 'Scan';
      opt.textContent = `Job #${j.job_id} — ${srcName} (${j.high_confidence_matches || 0} matches)`;
      select.appendChild(opt);
    });

    if (currentJobId && availableScanJobs.some(j => j.job_id == currentJobId)) {
      select.value = currentJobId;
    } else if (currentVal && availableScanJobs.some(j => j.job_id == currentVal)) {
      select.value = currentVal;
    }
  } catch (e) {
    console.debug('Failed to populate job filter:', e);
  }
}

function onJobFilterChange() {
  const select = document.getElementById('high-job-filter');
  currentJobId = select?.value ? parseInt(select.value) : null;
  loadHighGallery(1);
}

async function loadHighGallery(page) {
  highPage = page;
  try {
    await populateJobFilter();

    const params = new URLSearchParams({ page, page_size: PAGE_SIZE });
    const select = document.getElementById('high-job-filter');
    const selectedJob = select?.value || (currentJobId ? String(currentJobId) : '');

    if (selectedJob) {
      params.set('job_id', selectedJob);
    }
    const data = await apiFetch(`/api/gallery/high-confidence?${params}`);

    document.getElementById('high-total-count').textContent = `${data.total} photos`;
    renderHighGallery(data.items);
    renderPagination('high-pagination', data.total, page, PAGE_SIZE, loadHighGallery);
  } catch (e) {
    console.error('Gallery load failed:', e);
  }
}

function renderHighGallery(items) {
  const grid = document.getElementById('high-gallery-grid');
  grid.innerHTML = '';
  if (!items.length) {
    grid.innerHTML = '<div class="gallery-empty">No high-confidence matches found for this scan. Select another scan from the dropdown above.</div>';
    return;
  }
  items.forEach(item => {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    card.onclick = () => openPreview(item.image_file_id, item.filename);

    const thumbSrc = item.thumbnail_url || `/api/preview/${item.image_file_id}`;

    card.innerHTML = `
      <img class="gallery-thumb" src="${thumbSrc}" alt="${item.filename}" loading="lazy" onerror="this.onerror=null; this.src='/api/preview/${item.image_file_id}';">
      <div class="gallery-info">
        <div class="gallery-filename" title="${item.file_path}">${item.filename}</div>
        <div class="gallery-ids">
          ${(item.matched_identities || []).map(id =>
            `<span class="gallery-id-chip">${id}</span>`
          ).join('')}
        </div>
        <div class="gallery-score">Score: ${(item.best_score * 100).toFixed(1)}%</div>
        ${item.export_verified ? '<div class="gallery-verified">✓ Verified copy</div>' : ''}
      </div>
    `;
    grid.appendChild(card);
  });
}

// ─── Review Gallery ───────────────────────────────────────────────────────────
async function loadReviewGallery(page) {
  reviewPage = page;
  try {
    const params = new URLSearchParams({ page, page_size: 12 });
    if (currentJobId) params.set('job_id', currentJobId);
    const data = await apiFetch(`/api/gallery/review?${params}`);

    document.getElementById('review-total-count').textContent = `${data.total} pending`;
    renderReviewGallery(data.items);
    renderPagination('review-pagination', data.total, page, 12, loadReviewGallery);
  } catch (e) {
    console.error('Review load failed:', e);
  }
}

const IDENTITY_NAMES = { me: 'Me', mother: 'Mother', father: 'Father', sister: 'Sister' };

function renderReviewGallery(items) {
  const grid = document.getElementById('review-gallery-grid');
  grid.innerHTML = '';
  if (!items.length) {
    grid.innerHTML = '<div class="gallery-empty">No pending review candidates. 🎉</div>';
    return;
  }

  items.forEach(item => {
    const card = document.createElement('div');
    card.className = 'review-card';
    card.id = `review-card-${item.image_file_id}`;

    const facesHtml = (item.faces_detail || []).slice(0, 4).map(face => {
      const topMatch = (face.matches || []).sort((a, b) => b.similarity - a.similarity)[0];
      return `
        <div class="review-face">
          ${face.thumbnail_url ? `<img class="review-face-thumb" src="${face.thumbnail_url}" alt="face">` : '<div class="review-face-thumb" style="background:var(--bg-base);border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;">👤</div>'}
          <div class="review-face-info">
            <div class="review-face-match">${topMatch ? IDENTITY_NAMES[topMatch.identity_id] || topMatch.identity_id : 'Unknown'}</div>
            <div class="review-face-score">${topMatch ? (topMatch.similarity * 100).toFixed(1) + '%' : '—'}</div>
            ${face.is_small ? '<div style="color:var(--amber);font-size:0.7rem">⚠ Small face</div>' : ''}
            ${face.is_blurry ? '<div style="color:var(--amber);font-size:0.7rem">⚠ Blurry</div>' : ''}
          </div>
        </div>
      `;
    }).join('');

    card.innerHTML = `
      <img class="review-main-img" src="/api/preview/${item.image_file_id}" alt="${item.filename}"
           onclick="openPreview(${item.image_file_id}, '${item.filename.replace(/'/g, "\\'")}')"
           onerror="this.src=''" loading="lazy">
      <div class="review-content">
        <div class="review-filename">${item.filename}</div>
        ${item.has_quality_warnings ? `<div class="review-warnings">⚠ ${item.quality_warning_details}</div>` : ''}
        <div class="review-faces">${facesHtml || '<span class="text-muted">No faces with embeddings</span>'}</div>
        <div class="review-actions">
          <button class="btn btn-success btn-sm" onclick="reviewDecide(${item.image_file_id}, 'approved')">✓ Approve</button>
          <button class="btn btn-danger btn-sm" onclick="reviewDecide(${item.image_file_id}, 'rejected')">✗ Reject</button>
          <button class="btn btn-outline btn-sm" onclick="reviewDecide(${item.image_file_id}, 'corrected')">↑ Promote to High</button>
          <button class="btn btn-outline btn-sm" onclick="openPreview(${item.image_file_id}, '${item.filename.replace(/'/g, "\\'")}')">🔍 Full View</button>
        </div>
      </div>
    `;
    grid.appendChild(card);
  });
}

async function reviewDecide(imageFileId, decision) {
  try {
    const result = await apiFetch('/api/review/decide', 'POST', {
      image_file_id: imageFileId,
      decision: decision,
    });
    if (result.success) {
      const labels = { approved: 'Approved', rejected: 'Rejected', corrected: 'Promoted to high-confidence' };
      showToast(`✓ ${labels[decision] || decision}`, 'success');
      const card = document.getElementById(`review-card-${imageFileId}`);
      if (card) {
        card.style.opacity = '0.4';
        card.style.pointerEvents = 'none';
        setTimeout(() => card.remove(), 600);
      }
      // Update review badge count
      const badge = document.getElementById('badge-review');
      const cur = parseInt(badge.textContent) || 0;
      badge.textContent = Math.max(0, cur - 1);
    }
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
  }
}

// ─── Preview Modal ────────────────────────────────────────────────────────────
function openPreview(imageFileId, filename) {
  const modal = document.getElementById('preview-modal');
  const img = document.getElementById('modal-img');
  const caption = document.getElementById('modal-caption');

  img.src = `/api/preview/${imageFileId}`;
  caption.textContent = filename;
  modal.classList.remove('hidden');
}

function closeModal() {
  document.getElementById('preview-modal').classList.add('hidden');
  document.getElementById('modal-img').src = '';
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('preview-modal')?.addEventListener('click', e => {
    if (e.target === document.getElementById('preview-modal')) closeModal();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeModal();
  });
});

// ─── Reports & Thresholds ─────────────────────────────────────────────────────
async function loadThresholds() {
  try {
    identities = await apiFetch('/api/identities');
    const grid = document.getElementById('threshold-grid');
    if (!grid) return;

    thresholdData = {};
    grid.innerHTML = identities.map(ident => {
      thresholdData[ident.identity_id] = {
        high: ident.high_threshold,
        review: ident.review_threshold
      };
      return `
        <div class="threshold-card">
          <div class="threshold-identity">${IDENTITY_EMOJIS[ident.identity_id] || '👤'} ${ident.display_name}</div>
          <div class="threshold-row">
            <span class="threshold-label">High-confidence threshold</span>
            <input class="threshold-input" type="number" min="0" max="1" step="0.01"
              id="thresh-high-${ident.identity_id}" value="${ident.high_threshold}"
              title="Similarity score above this → copied to Destination A" />
          </div>
          <div class="threshold-row">
            <span class="threshold-label">Review threshold</span>
            <input class="threshold-input" type="number" min="0" max="1" step="0.01"
              id="thresh-review-${ident.identity_id}" value="${ident.review_threshold}"
              title="Similarity score above this → copied to Destination B for review" />
          </div>
        </div>
      `;
    }).join('');
  } catch (e) {
    console.error('Thresholds load failed:', e);
  }
}

async function saveThresholds() {
  const promises = identities.map(ident => {
    const high = parseFloat(document.getElementById(`thresh-high-${ident.identity_id}`)?.value);
    const review = parseFloat(document.getElementById(`thresh-review-${ident.identity_id}`)?.value);
    if (isNaN(high) || isNaN(review)) return Promise.resolve();
    return apiFetch(`/api/identities/${ident.identity_id}/thresholds`, 'PUT', {
      identity_id: ident.identity_id,
      high_threshold: high,
      review_threshold: review,
    });
  });
  await Promise.all(promises);
  showToast('✓ Thresholds saved. Re-scan to apply to new images.', 'success');
}

async function loadJobSummary() {
  try {
    const jobs = await apiFetch('/api/reports/summary');
    const container = document.getElementById('job-summary-table');
    if (!jobs || !jobs.length) {
      container.innerHTML = '<p class="text-muted text-sm" style="padding:1rem;">No scan jobs in history.</p>';
      return;
    }
    container.innerHTML = `
      <table class="job-table" style="width:100%; border-collapse:collapse;">
        <thead><tr>
          <th>Job ID</th>
          <th>Status</th>
          <th>Source Folder</th>
          <th>Discovered</th>
          <th>Processed</th>
          <th>High Matches</th>
          <th>Review</th>
          <th>Date</th>
          <th style="text-align:center;">Actions</th>
        </tr></thead>
        <tbody>
          ${jobs.map(j => `
            <tr>
              <td><strong>#${j.job_id}</strong></td>
              <td><span class="badge badge-${j.status === 'completed' ? 'success' : j.status === 'error' ? 'error' : j.status === 'running' ? 'info' : 'warning'}">${j.status}</span></td>
              <td title="${j.source_folder}"><code style="font-size:0.8rem;">${truncatePath(j.source_folder, 32)}</code></td>
              <td>${j.total_files}</td>
              <td>${j.processed_files}</td>
              <td><span style="font-weight:600; color:var(--green);">${j.high_confidence_matches || 0}</span></td>
              <td>${j.review_candidates || 0}</td>
              <td>${j.created_at ? new Date(j.created_at).toLocaleDateString() : '—'}</td>
              <td style="text-align:center; white-space:nowrap;">
                ${(j.high_confidence_matches > 0) ? `
                  <button class="btn btn-outline btn-sm" style="padding:2px 8px; font-size:0.75rem; margin-right:4px;" onclick="viewJobMatches(${j.job_id})" title="View Matched Photos">
                    👁 View (${j.high_confidence_matches})
                  </button>
                ` : ''}
                <button class="btn btn-danger-outline btn-sm" style="padding:2px 8px; font-size:0.75rem;" onclick="deleteJob(${j.job_id})" title="Delete this scan history">
                  🗑️
                </button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch (e) {
    console.error('Jobs load failed:', e);
  }
}

function viewJobMatches(jobId) {
  currentJobId = jobId;
  navigateTo('gallery-high');
  const select = document.getElementById('high-job-filter');
  if (select) {
    select.value = String(jobId);
  }
  loadHighGallery(1);
}

async function deleteJob(jobId) {
  if (!confirm(`Are you sure you want to delete scan history for Job #${jobId}?`)) return;
  try {
    const res = await apiFetch(`/api/reports/jobs/${jobId}`, 'DELETE');
    if (res.success) {
      showToast(`✓ Scan Job #${jobId} deleted.`, 'success');
      loadJobSummary();
      updateScanStatus();
    }
  } catch (e) {
    showToast(`Failed to delete job: ${e.message}`, 'error');
  }
}

async function cleanEmptyJobs() {
  try {
    const res = await apiFetch('/api/reports/jobs/clean-empty', 'POST');
    showToast(`✓ Removed ${res.deleted_count || 0} empty / pending jobs from history.`, 'success');
    loadJobSummary();
    updateScanStatus();
  } catch (e) {
    showToast(`Failed to clean jobs: ${e.message}`, 'error');
  }
}

async function clearAllJobs() {
  if (!confirm('Are you sure you want to completely clear all scan history? (This resets scan records and gallery classifications, but never touches source or exported photos).')) return;
  try {
    const res = await apiFetch('/api/reports/jobs/clear-all', 'POST');
    if (res.success) {
      showToast('✓ All scan history and classifications cleared.', 'success');
      currentJobId = null;
      loadJobSummary();
      updateScanStatus();
      if (currentPage === 'gallery-high') loadHighGallery(1);
    }
  } catch (e) {
    showToast(`Failed to clear history: ${e.message}`, 'error');
  }
}

async function viewErrors() {
  const card = document.getElementById('error-log-card');
  const content = document.getElementById('error-log-content');
  card.classList.toggle('hidden');
  if (!card.classList.contains('hidden')) {
    try {
      const errors = await apiFetch('/api/reports/errors');
      if (!errors.length) {
        content.innerHTML = '<p class="text-muted text-sm">No errors logged.</p>';
        return;
      }
      content.innerHTML = errors.map(e => `
        <div class="error-log-entry">
          <div class="err-time">${e.timestamp ? new Date(e.timestamp).toLocaleString() : '—'}</div>
          <div class="err-type">${e.error_type}</div>
          <div class="err-msg">${e.message}</div>
          ${e.file_path ? `<div class="text-muted text-sm">${e.file_path}</div>` : ''}
        </div>
      `).join('');
    } catch (ex) {
      content.innerHTML = `<p class="text-muted">Failed to load: ${ex.message}</p>`;
    }
  }
}

async function clearEmbeddings() {
  if (!confirm('This will clear all cached recognition data. Source and exported files are NOT affected. Continue?')) return;
  try {
    await apiFetch('/api/cache/clear-embeddings', 'POST');
    showToast('✓ Cached recognition data cleared.', 'success');
  } catch (e) {
    showToast(`Failed: ${e.message}`, 'error');
  }
}

// ─── Pagination ───────────────────────────────────────────────────────────────
function renderPagination(containerId, total, currentPage, pageSize, loadFn) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const totalPages = Math.ceil(total / pageSize);
  if (totalPages <= 1) { container.innerHTML = ''; return; }

  let html = '';
  if (currentPage > 1) html += `<button class="page-btn" onclick="${loadFn.name}(${currentPage - 1})">← Prev</button>`;
  for (let p = Math.max(1, currentPage - 2); p <= Math.min(totalPages, currentPage + 2); p++) {
    html += `<button class="page-btn ${p === currentPage ? 'active' : ''}" onclick="${loadFn.name}(${p})">${p}</button>`;
  }
  if (currentPage < totalPages) html += `<button class="page-btn" onclick="${loadFn.name}(${currentPage + 1})">Next →</button>`;
  container.innerHTML = html;
}

// ─── Toast Notifications ──────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─── API Helper ───────────────────────────────────────────────────────────────
async function apiFetch(path, method = 'GET', body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body !== null) opts.body = JSON.stringify(body);
  const resp = await fetch(`${API}${path}`, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ─── Utility ──────────────────────────────────────────────────────────────────
function truncatePath(p, maxLen) {
  if (!p) return '—';
  if (p.length <= maxLen) return p;
  return '…' + p.slice(-maxLen);
}

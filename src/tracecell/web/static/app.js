/* ==========================================================================
   TraceCell 证据账本工作台 · 前端逻辑
   零依赖、零构建：只用浏览器原生 API。
   ========================================================================== */

'use strict';

/* ── 后端接口契约 ──────────────────────────────────────────────────── */
const API = {
  health: '/api/health',
  samples: '/api/samples',
  ledger: (docId) => `/api/ledger/${encodeURIComponent(docId)}`,
  image: (docId) => `/api/image/${encodeURIComponent(docId)}`,
  crop: (docId, regionId) => `/api/crop/${encodeURIComponent(docId)}/${encodeURIComponent(regionId)}`,
  checklist: (docId) => `/api/checklist/${encodeURIComponent(docId)}`,
  analyze: '/api/analyze',
};

/* ── 领域词典 ──────────────────────────────────────────────────────── */
const STATUS_META = {
  accepted: { label: '免检通过', cls: 'ok' },
  repaired: { label: '已修正', cls: 'info' },
  pending: { label: '待人工', cls: 'warn' },
  escalated: { label: '已升级', cls: 'danger' },
  disputed: { label: '存在矛盾', cls: 'violet' },
};

const SEVERITY_META = {
  critical: { label: '致命', cls: 'sev-critical' },
  high: { label: '高', cls: 'sev-high' },
  medium: { label: '中', cls: 'sev-medium' },
  low: { label: '低', cls: 'sev-low' },
};

const SEVERITY_ORDER = { critical: 0, high: 1, medium: 2, low: 3 };

const VERDICT_META = {
  auto_pass: { label: '免检通过', cls: 'ok' },
  spot_check: { label: '抽检', cls: 'warn' },
  full_review: { label: '全检', cls: 'danger' },
};

const ACTION_META = {
  accept: { label: '放行', cls: 'ok' },
  reparse: { label: '重解析', cls: 'info' },
  best_of_n: { label: '择优', cls: 'info' },
  best: { label: '择优执行', cls: 'info' },
  vlm_escalation: { label: '升级复核', cls: 'violet' },
  human_review: { label: '转人工', cls: 'warn' },
};

const ORIGIN_META = {
  primary: '首次解析',
  'reparse-4x': '重解析 4x',
  'reparse-16x': '重解析 16x',
  'vlm-review': '多模态复核',
};

const CANDIDATE_VERDICT_META = {
  scored: { label: '已评分', cls: 'ok' },
  not_scored: { label: '未评分', cls: 'muted' },
  failed: { label: '评分失败', cls: 'danger' },
};

const CATEGORY_META = {
  text: '文本',
  table: '表格',
  formula: '公式',
  figure: '图示',
  seal: '印章',
};

/* 区域状态 → 核验动作的对照（用于账本未回传 decisions 时的轨迹还原） */
const STATUS_ACTION_META = {
  accepted: { label: '放行', cls: 'ok' },
  repaired: { label: '重解析/择优修正', cls: 'info' },
  escalated: { label: '升级复核', cls: 'violet' },
  pending: { label: '转人工', cls: 'warn' },
  disputed: { label: '存在矛盾', cls: 'danger' },
};

const PROVIDER_META = {
  replay: '回放引擎',
  minicpm: 'MiniCPM',
};

const SORT_CYCLE = { none: 'desc', desc: 'asc', asc: 'none' };
const SORT_LABEL = { none: '默认顺序', desc: '降序', asc: '升序' };
const SORT_ARROW = { none: '', desc: '↓', asc: '↑' };

/* ── 小工具 ────────────────────────────────────────────────────────── */
const $ = (selector, root) => (root || document).querySelector(selector);
const $$ = (selector, root) => Array.prototype.slice.call((root || document).querySelectorAll(selector));

function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function fmt(value, digits) {
  const n = toNumber(value);
  return n === null ? '—' : n.toFixed(digits === undefined ? 3 : digits);
}

function fmtPercent(value, digits) {
  const n = toNumber(value);
  if (n === null) return '—';
  return (n * 100).toFixed(digits === undefined ? 1 : digits) + '%';
}

function clamp01(value) {
  const n = toNumber(value);
  if (n === null) return 0;
  return Math.min(1, Math.max(0, n));
}

function confidenceClass(confidence) {
  const n = toNumber(confidence);
  if (n === null) return 'muted';
  if (n >= 0.9) return 'ok';
  if (n >= 0.6) return 'warn';
  return 'danger';
}

function statusMeta(status) {
  return STATUS_META[status] || { label: status || '未知', cls: 'muted' };
}

function severityMeta(severity) {
  return SEVERITY_META[severity] || { label: severity || '未知', cls: 'sev-low' };
}

function actionMeta(action) {
  return ACTION_META[action] || { label: action || '未知', cls: 'muted' };
}

function originLabel(origin) {
  return ORIGIN_META[origin] || origin || '未知来源';
}

function tag(text, cls) {
  return `<span class="tag ${esc(cls)}">${esc(text)}</span>`;
}

/* ── 网络 ──────────────────────────────────────────────────────────── */
async function apiGet(path) {
  const response = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

async function readError(response) {
  const raw = await response.text().catch(() => '');
  let detail = raw;
  try {
    const data = JSON.parse(raw);
    if (data && data.detail) detail = String(data.detail);
  } catch (err) {
    // 不是 JSON，按原样使用
  }
  detail = (detail || '').trim();
  if (!detail || detail === 'Internal Server Error') {
    return `服务端返回 ${response.status}${response.statusText ? ' ' + response.statusText : ''}`;
  }
  return detail;
}

/* ── 全局状态 ──────────────────────────────────────────────────────── */
const state = {
  docId: null,
  ledger: null,
  samples: [],
  localDocs: [],
  checklist: null,
  sortDir: 'none',
  drawer: null,          // { field, regionId }
  health: null,
  loading: false,
  error: null,
};

/* ── 提示条 ────────────────────────────────────────────────────────── */
function showToast(message, kind) {
  const wrap = $('#toastWrap');
  if (!wrap) return;
  const toast = document.createElement('div');
  toast.className = `toast is-${kind || 'info'}`;
  toast.textContent = message;
  wrap.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 220);
  }, 3200);
}

/* ── 启动 ──────────────────────────────────────────────────────────── */
async function boot() {
  bindEvents();
  loadHealth();
  await loadSamples();
  const docs = allDocs();
  if (docs.length) {
    await selectDocument(docs[0].doc_id);
  } else {
    state.ledger = null;
    renderAll();
  }
}

function bindEvents() {
  $('#docSelect').addEventListener('change', (event) => {
    const docId = event.target.value;
    if (docId) selectDocument(docId);
  });

  $('#confHeader').addEventListener('click', toggleSort);
  $('#confHeader').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleSort();
    }
  });

  $('#fieldBody').addEventListener('click', (event) => {
    const chip = event.target.closest('.rid[data-region]');
    if (chip) {
      event.stopPropagation();
      openRegion(chip.getAttribute('data-region'));
      return;
    }
    const row = event.target.closest('tr[data-index]');
    if (row) openField(Number(row.getAttribute('data-index')));
  });

  $('#fieldBody').addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const row = event.target.closest('tr[data-index]');
    if (!row) return;
    event.preventDefault();
    openField(Number(row.getAttribute('data-index')));
  });

  $('#decisionBody').addEventListener('click', (event) => {
    const chip = event.target.closest('.rid[data-region]');
    if (chip) openRegion(chip.getAttribute('data-region'));
  });

  $('#exportCsvBtn').addEventListener('click', exportCsv);

  $('#uploadBtn').addEventListener('click', () => openModal('uploadModal'));
  $('#uploadForm').addEventListener('submit', handleUpload);

  $('#drawerClose').addEventListener('click', closeDrawer);
  $('#scrim').addEventListener('click', closeDrawer);

  $$('[data-close]').forEach((button) => {
    button.addEventListener('click', () => closeModal(button.getAttribute('data-close')));
  });

  $$('.modal').forEach((modal) => {
    modal.addEventListener('click', (event) => {
      if (event.target === modal) closeModal(modal.id);
    });
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    const open = $$('.modal').filter((modal) => !modal.hidden).pop();
    if (open) {
      closeModal(open.id);
      return;
    }
    if (state.drawer) closeDrawer();
  });
}

/* ── 健康检查 ──────────────────────────────────────────────────────── */
async function loadHealth() {
  const chip = $('#healthChip');
  const text = $('#healthText');
  try {
    const health = await apiGet(API.health);
    state.health = health;
    chip.classList.add('is-ok');
    chip.classList.remove('is-down');
    const provider = PROVIDER_META[health.provider] || health.provider || '未知';
    text.textContent = `${provider} · v${health.version || '—'}`;
  } catch (err) {
    chip.classList.add('is-down');
    chip.classList.remove('is-ok');
    text.textContent = '后端未连接';
  }
}

/* ── 文档列表 ──────────────────────────────────────────────────────── */
async function loadSamples() {
  try {
    const samples = await apiGet(API.samples);
    state.samples = Array.isArray(samples) ? samples : [];
  } catch (err) {
    state.samples = [];
    showToast(`样例列表加载失败：${err.message}`, 'error');
  }
  renderDocLists();
}

function allDocs() {
  const merged = [];
  const seen = {};
  state.localDocs.concat(state.samples).forEach((doc) => {
    if (!doc || !doc.doc_id || seen[doc.doc_id]) return;
    seen[doc.doc_id] = true;
    merged.push(doc);
  });
  return merged;
}

function renderDocLists() {
  const docs = allDocs();
  const select = $('#docSelect');
  const list = $('#docList');

  $('#railCount').textContent = docs.length ? `${docs.length} 份` : '';

  if (!docs.length) {
    select.innerHTML = '<option value="">暂无文档</option>';
    select.disabled = true;
    list.innerHTML = '<li class="doc-empty">当前没有可用的文档，点右上角「上传图片」开始核验。</li>';
    return;
  }

  select.disabled = false;
  select.innerHTML = docs
    .map((doc) => `<option value="${esc(doc.doc_id)}">${esc(doc.label || doc.doc_id)}</option>`)
    .join('');
  if (state.docId) select.value = state.docId;

  list.innerHTML = docs
    .map((doc) => {
      const active = doc.doc_id === state.docId ? ' is-active' : '';
      const meta = [doc.schema, doc.description].filter(Boolean).join(' · ');
      return `<li><button type="button" class="doc-item${active}" data-doc="${esc(doc.doc_id)}">
        <span class="doc-item-name">${esc(doc.label || doc.doc_id)}</span>
        <span class="doc-item-meta">${esc(doc.doc_id)}${meta ? ' · ' + esc(meta) : ''}</span>
      </button></li>`;
    })
    .join('');

  $$('.doc-item', list).forEach((item) => {
    item.addEventListener('click', () => selectDocument(item.getAttribute('data-doc')));
  });
}

/* ── 载入账本 ──────────────────────────────────────────────────────── */
async function selectDocument(docId, fallback) {
  state.docId = docId;
  state.checklist = null;
  state.ledger = null;
  state.loading = true;
  state.error = null;
  renderDocLists();
  renderAll();

  let ledger = null;
  try {
    try {
      ledger = await apiGet(API.ledger(docId));
    } catch (inner) {
      if (!fallback) throw inner;
      ledger = fallback;
    }
    state.ledger = ledger;
    state.docId = ledger.doc_id || docId;
    state.loading = false;
    renderAll();
    loadChecklist(state.docId);
  } catch (err) {
    state.loading = false;
    state.error = err.message;
    state.ledger = null;
    renderAll();
    showToast(`账本加载失败：${err.message}`, 'error');
  }
}

async function loadChecklist(docId) {
  try {
    const checklist = await apiGet(API.checklist(docId));
    state.checklist = Array.isArray(checklist) ? checklist : [];
  } catch (err) {
    state.checklist = null;
  }
  renderChecklist();
}

/* ── 渲染入口 ──────────────────────────────────────────────────────── */
function renderAll() {
  renderSummary();
  renderFields();
  renderDecisions();
  renderBudget();
  renderSampling();
  renderFindings();
  renderChecklist();
  renderDocLists();
}

function showLedgerLoading() {
  $('#emptyTitle').textContent = '正在生成账本…';
  $('#emptyHint').textContent = '首次载入需要跑一遍版面解析与核验，请稍候。';
}

function showLedgerError(message) {
  $('#emptyTitle').textContent = '这份文档暂时载入不了';
  $('#emptyHint').textContent = message || '请稍后重试，或改用上传图片。';
}

/* ── 摘要条 ────────────────────────────────────────────────────────── */
function renderSummary() {
  const ledger = state.ledger;
  const summary = (ledger && ledger.summary) || {};
  const budget = (ledger && ledger.budget) || summary.budget || {};
  const routing = (ledger && ledger.meta && ledger.meta.routing) || {};
  const verdict = summary.verdict || (ledger && ledger.verdict) || 'none';
  const meta = VERDICT_META[verdict] || { label: verdict === 'none' ? '未载入' : verdict, cls: 'muted' };

  const badge = $('#verdictBadge');
  badge.setAttribute('data-verdict', verdict);
  $('#verdictText').textContent = meta.label;
  $('#verdictReason').textContent = (routing.reasons && routing.reasons[0]) || '';

  const fields = ledger && ledger.fields ? ledger.fields.length : 0;
  const regions = ledger && ledger.regions ? ledger.regions.length : 0;

  $('#statFields').textContent = summary.field_count !== undefined ? summary.field_count : (fields || '—');
  $('#statRegions').textContent = summary.region_count !== undefined ? summary.region_count : (regions || '—');

  const mean = toNumber(summary.mean_region_confidence);
  $('#statConfidence').textContent = mean === null ? '—' : fmtPercent(mean, 1);

  const loss = toNumber(summary.expected_loss);
  const band = summary.band;
  const lossNote = $('#statLossBand');
  lossNote.className = 'stat-note';
  if (loss === null) {
    $('#statLoss').textContent = '—';
    lossNote.textContent = '';
  } else {
    $('#statLoss').textContent = fmt(loss, 4);
    const bandLabel = band === 'low' ? '低风险' : band === 'medium' ? '中风险' : band === 'high' ? '高风险' : '';
    lossNote.textContent = bandLabel;
    if (band === 'low') lossNote.classList.add('is-ok');
    else if (band === 'medium') lossNote.classList.add('is-warn');
    else if (band === 'high') lossNote.classList.add('is-danger');
  }

  const cost = toNumber(budget.total_cost);
  $('#statCost').textContent = cost === null ? '—' : cost.toFixed(2);

  const human = budget.human_cells !== undefined ? budget.human_cells : summary.review_queue_size;
  $('#statHuman').textContent = human === undefined || human === null ? '—' : human;
  const queueSize = summary.review_queue_size;
  const humanNote = $('#statHumanNote');
  humanNote.className = 'stat-note';
  if (queueSize !== undefined && queueSize !== null) {
    humanNote.textContent = `复核队列 ${queueSize}`;
    if (queueSize > 0) humanNote.classList.add('is-warn');
  } else {
    humanNote.textContent = '';
  }

  $('#exportCsvBtn').disabled = !(ledger && ledger.csv);
}

/* ── 字段台账 ──────────────────────────────────────────────────────── */
function sortedFields() {
  const fields = state.ledger && state.ledger.fields ? state.ledger.fields.slice() : [];
  if (state.sortDir === 'none') return fields;
  const sign = state.sortDir === 'desc' ? -1 : 1;
  fields.sort((a, b) => {
    const left = toNumber(a.confidence);
    const right = toNumber(b.confidence);
    if (left === null && right === null) return 0;
    if (left === null) return 1;
    if (right === null) return -1;
    return sign * (left - right);
  });
  return fields;
}

function toggleSort() {
  state.sortDir = SORT_CYCLE[state.sortDir] || 'desc';
  $('#sortInd').textContent = SORT_ARROW[state.sortDir];
  $('#confHeader').setAttribute('aria-label', `按置信度排序，当前${SORT_LABEL[state.sortDir]}`);
  renderFields();
}

function renderFields() {
  const body = $('#fieldBody');
  const empty = $('#fieldEmpty');
  $('#ledgerCount').textContent = '';

  if (state.loading) {
    body.innerHTML = '';
    empty.hidden = false;
    showLedgerLoading();
    return;
  }

  if (state.error) {
    body.innerHTML = '';
    empty.hidden = false;
    showLedgerError(state.error);
    return;
  }

  if (!state.ledger) {
    body.innerHTML = '';
    empty.hidden = false;
    $('#emptyTitle').textContent = '还没有载入账本';
    $('#emptyHint').textContent = '从上方选一份文档，或点右上角「上传图片」开始核验。';
    return;
  }

  const fields = sortedFields();
  $('#ledgerCount').textContent = fields.length ? `${fields.length} 个字段` : '';

  if (!fields.length) {
    body.innerHTML = '';
    empty.hidden = false;
    $('#emptyTitle').textContent = '这份文档没有抽出字段';
    $('#emptyHint').textContent = '可能是版面里没有匹配到单据模板的字段，换一种单据类型再试。';
    return;
  }

  empty.hidden = true;
  body.innerHTML = fields.map(fieldRowHtml).join('');
}

function fieldRowHtml(field, index) {
  const confidence = toNumber(field.confidence);
  const confCls = confidenceClass(confidence);
  const width = confidence === null ? 0 : Math.round(confidence * 100);
  const sev = severityMeta(field.severity);
  const st = statusMeta(field.status);

  const regionChips = (field.region_ids || [])
    .map((rid) => `<button type="button" class="rid" data-region="${esc(rid)}">${esc(rid)}</button>`)
    .join('');

  return `<tr data-index="${index}" tabindex="0">
    <td class="col-name">${esc(field.name)}</td>
    <td class="col-value">${esc(field.value)}</td>
    <td class="col-conf">
      <div class="conf">
        <div class="conf-bar"><i class="${confCls}" style="width:${width}%"></i></div>
        <span class="conf-num ${confCls}">${fmt(confidence, 3)}</span>
      </div>
    </td>
    <td>${tag(sev.label, sev.cls)}</td>
    <td>${tag(st.label, st.cls)}</td>
    <td class="col-rid">${regionChips || '<span class="muted">无证据</span>'}</td>
  </tr>`;
}

/* ── 决策流水 ──────────────────────────────────────────────────────── */
function renderDecisions() {
  const body = $('#decisionBody');
  const decisions = state.ledger && state.ledger.decisions ? state.ledger.decisions : [];

  if (decisions.length) {
    $('#decColAction').textContent = '动作';
    $('#decColDelta').textContent = '置信度变化';
    $('#decisionEmpty').hidden = true;
    $('#processMeta').textContent = `${decisions.length} 次决策`;
    body.innerHTML = decisions.map(decisionRowHtml).join('');
    return;
  }

  // 账本没有回传 decisions 时，用区域上的审计理由还原一条轨迹。
  const regions = state.ledger && state.ledger.regions ? state.ledger.regions : [];
  const traced = regions.filter((region) => region.note);

  if (!traced.length) {
    $('#decColAction').textContent = '动作';
    $('#decColDelta').textContent = '置信度变化';
    $('#decisionEmpty').hidden = false;
    $('#processMeta').textContent = '';
    body.innerHTML = '';
    return;
  }

  $('#decColAction').textContent = '结果';
  $('#decColDelta').textContent = '最佳候选分';
  $('#decisionEmpty').hidden = true;
  $('#processMeta').textContent = `${traced.length} 条区域轨迹`;
  body.innerHTML = traced.map(regionTraceRowHtml).join('');
}

function decisionRowHtml(decision) {
  const action = actionMeta(decision.action);
  const before = toNumber(decision.confidence_before);
  const after = toNumber(decision.confidence_after);
  let deltaCls = 'same';
  if (before !== null && after !== null) {
    if (after > before) deltaCls = 'up';
    else if (after < before) deltaCls = 'down';
  }
  const cost = toNumber(decision.cost);
  const regionId = decision.region_id || '';
  const chip = findRegion(regionId) ? ` class="rid" data-region="${esc(regionId)}"` : '';

  return `<tr>
    <td><span${chip}>${esc(regionId || '—')}</span></td>
    <td>${tag(action.label, action.cls)}</td>
    <td>
      <span class="delta">
        <span class="from">${fmt(before, 3)}</span>
        <span class="arrow">→</span>
        <span class="to ${deltaCls}">${fmt(after, 3)}</span>
      </span>
    </td>
    <td>${cost === null ? '—' : cost.toFixed(2)}</td>
    <td class="reason-cell"><div class="reason-text">${esc(decision.reason || '')}</div></td>
  </tr>`;
}

function regionTraceRowHtml(region) {
  const action = STATUS_ACTION_META[region.status] || { label: region.status || '未知', cls: 'muted' };
  const cost = toNumber((state.ledger.budget || {}).cost_by_action
    ? (state.ledger.budget.cost_by_action || {})[region.id]
    : null);
  const best = bestCandidateScore(region);
  const chip = ` class="rid" data-region="${esc(region.id)}"`;

  return `<tr>
    <td><span${chip}>${esc(region.id)}</span></td>
    <td>${tag(action.label, action.cls)}</td>
    <td><span class="delta"><span class="to same">${fmt(best, 3)}</span></span></td>
    <td>${cost === null ? '—' : cost.toFixed(2)}</td>
    <td class="reason-cell"><div class="reason-text">${esc(region.note || '')}</div></td>
  </tr>`;
}

function bestCandidateScore(region) {
  const candidates = region.candidates || [];
  let best = null;
  candidates.forEach((candidate) => {
    if (candidate.verdict !== 'scored') return;
    const score = toNumber(candidate.score);
    if (score === null) return;
    if (best === null || score > best) best = score;
  });
  return best;
}

/* ── 成本账单 ──────────────────────────────────────────────────────── */
function renderBudget() {
  const host = $('#budgetBody');
  const ledger = state.ledger;
  if (!ledger) {
    host.innerHTML = '<div class="empty empty-sm"><p class="empty-hint">尚未载入账本。</p></div>';
    return;
  }

  const budget = ledger.budget || {};
  const total = toNumber(budget.total_cost);
  const byRegion = budget.cost_by_action || {};
  const entries = Object.keys(byRegion)
    .map((rid) => ({ rid: rid, cost: toNumber(byRegion[rid]) || 0 }))
    .sort((a, b) => b.cost - a.cost);
  const maxCost = entries.reduce((acc, item) => Math.max(acc, item.cost), 0);

  const rows = [
    ['额外调用次数', budget.extra_calls === undefined ? '—' : budget.extra_calls],
    ['转人工格子', budget.human_cells === undefined ? '—' : budget.human_cells],
    ['残余风险', budget.residual_risk === undefined ? '—' : fmt(budget.residual_risk, 4)],
    ['区域状态', [
      `放行 ${budget.accepted_regions || 0}`,
      `修正 ${budget.repaired_regions || 0}`,
      `升级 ${budget.escalated_regions || 0}`,
      `待人工 ${budget.pending_regions || 0}`,
    ].join(' / ')],
  ];

  let html = `<div class="budget-total">
    <b>${total === null ? '—' : total.toFixed(2)}</b><span>rcu · 本次核验总花费</span>
  </div>`;

  html += '<div class="kv">';
  rows.forEach((row) => {
    html += `<div class="kv-row"><span class="kv-k">${esc(row[0])}</span><span class="kv-v">${esc(row[1])}</span></div>`;
  });
  html += '</div>';

  if (entries.length) {
    html += '<div class="kv-row"><span class="kv-k">按区域花费</span><span class="kv-v muted">rcu</span></div>';
    html += entries
      .map((item) => {
        const width = maxCost > 0 ? Math.round((item.cost / maxCost) * 100) : 0;
        const chip = findRegion(item.rid) ? ` class="rid" data-region="${esc(item.rid)}"` : '';
        return `<div class="cost-row">
          <span${chip}>${esc(item.rid)}</span>
          <span class="cost-bar"><i style="width:${width}%"></i></span>
          <span class="cost-num">${item.cost.toFixed(2)}</span>
        </div>`;
      })
      .join('');
  }

  host.innerHTML = html;

  $$('#budgetBody .rid[data-region]').forEach((chip) => {
    chip.addEventListener('click', () => openRegion(chip.getAttribute('data-region')));
  });
}

/* ── 抽样计划 ──────────────────────────────────────────────────────── */
function renderSampling() {
  const host = $('#samplingBody');
  const routing = state.ledger && state.ledger.meta && state.ledger.meta.routing;
  const sampling = routing && routing.sampling;

  if (!state.ledger) {
    host.innerHTML = '<div class="empty empty-sm"><p class="empty-hint">尚未载入账本。</p></div>';
    return;
  }

  if (!sampling) {
    host.innerHTML = '<div class="empty empty-sm"><p class="empty-hint">本次核验没有生成抽样计划。</p></div>';
    return;
  }

  host.innerHTML = `
    <div class="sample-grid">
      <div class="sample-cell"><span>总体规模</span><b>${esc(sampling.population)}</b></div>
      <div class="sample-cell"><span>抽样数量</span><b>${esc(sampling.sample_size)}</b></div>
      <div class="sample-cell"><span>估计出错率</span><b>${fmtPercent(sampling.estimated_error_rate, 1)}</b></div>
      <div class="sample-cell"><span>检出概率</span><b>${fmtPercent(sampling.detection_probability, 1)}</b></div>
    </div>
    <div class="sample-rationale">${esc(sampling.rationale || '')}</div>
  `;
}

/* ── 问题清单 ──────────────────────────────────────────────────────── */
function renderFindings() {
  const host = $('#findingsBody');
  if (!state.ledger) {
    host.innerHTML = '<div class="empty empty-sm"><p class="empty-hint">尚未载入账本。</p></div>';
    return;
  }
  const findings = state.ledger.findings ? state.ledger.findings.slice() : [];

  if (!findings.length) {
    host.innerHTML = '<div class="empty empty-sm"><p class="empty-hint">没有发现遗留问题。</p></div>';
    return;
  }

  findings.sort((a, b) => {
    const left = SEVERITY_ORDER[a.severity] === undefined ? 9 : SEVERITY_ORDER[a.severity];
    const right = SEVERITY_ORDER[b.severity] === undefined ? 9 : SEVERITY_ORDER[b.severity];
    return left - right;
  });

  host.innerHTML = findings
    .map((finding) => {
      const sev = severityMeta(finding.severity);
      const links = (finding.region_ids || [])
        .filter((rid) => findRegion(rid))
        .map((rid) => `<button type="button" class="rid" data-region="${esc(rid)}">${esc(rid)}</button>`)
        .join('');
      const fields = (finding.field_names || [])
        .map((name) => `<span class="check-pill">${esc(name)}</span>`)
        .join('');
      return `<div class="finding ${esc(sev.cls)}">
        <div class="finding-head">
          ${tag(sev.label, sev.cls)}
          <span class="muted">${esc(finding.kind || '')}</span>
        </div>
        <div class="finding-msg">${esc(finding.message || '')}</div>
        ${finding.suggestion ? `<div class="finding-sugg">建议：${esc(finding.suggestion)}</div>` : ''}
        ${links || fields ? `<div class="finding-links">${links}${fields}</div>` : ''}
      </div>`;
    })
    .join('');

  $$('#findingsBody .rid[data-region]').forEach((chip) => {
    chip.addEventListener('click', () => openRegion(chip.getAttribute('data-region')));
  });
}

/* ── 待复核清单 ────────────────────────────────────────────────────── */
function renderChecklist() {
  const host = $('#checklistBody');
  if (!state.ledger) {
    host.innerHTML = '<div class="empty empty-sm"><p class="empty-hint">尚未载入账本。</p></div>';
    return;
  }
  if (state.checklist === null) {
    host.innerHTML = '<div class="empty empty-sm"><p class="empty-hint">清单加载中…</p></div>';
    return;
  }
  if (!state.checklist.length) {
    host.innerHTML = '<div class="empty empty-sm"><p class="empty-hint">没有必须人工过目的格子。</p></div>';
    return;
  }

  host.innerHTML = state.checklist
    .map((item) => {
      const target = item.target || '';
      const known = findRegion(target) ? ` class="rid" data-region="${esc(target)}"` : '';
      const fields = (item.fields || [])
        .map((fld) => `<span class="check-pill">${esc(fld.name)} = ${esc(fld.value)}（${fmt(fld.confidence, 3)}）</span>`)
        .join('');
      return `<div class="check-item">
        <div class="check-head">
          ${known ? `<span${known}>${esc(target)}</span>` : `<span class="muted">${esc(target)}</span>`}
          ${item.page ? `<span class="muted">第 ${esc(item.page)} 页</span>` : ''}
        </div>
        <div class="check-why">${esc(item.why || '')}</div>
        ${fields ? `<div class="check-fields">${fields}</div>` : ''}
      </div>`;
    })
    .join('');

  $$('#checklistBody .rid[data-region]').forEach((chip) => {
    chip.addEventListener('click', () => openRegion(chip.getAttribute('data-region')));
  });
}

/* ── 详情抽屉 ──────────────────────────────────────────────────────── */
function findRegion(regionId) {
  if (!state.ledger || !state.ledger.regions || !regionId) return null;
  for (const region of state.ledger.regions) {
    if (region.id === regionId) return region;
  }
  return null;
}

function openField(index) {
  const fields = sortedFields();
  const field = fields[index];
  if (!field) return;
  const regionIds = field.region_ids || [];
  openDrawer({ field: field, regionId: regionIds[0] || null });
}

function openRegion(regionId) {
  const region = findRegion(regionId);
  if (!region) return;
  let field = null;
  const fields = state.ledger && state.ledger.fields ? state.ledger.fields : [];
  for (const item of fields) {
    if ((item.region_ids || []).indexOf(regionId) >= 0) {
      field = item;
      break;
    }
  }
  openDrawer({ field: field, regionId: regionId });
}

function openDrawer(target) {
  state.drawer = target;
  renderDrawer();
  $('#drawer').classList.add('is-open');
  $('#drawer').setAttribute('aria-hidden', 'false');
  document.body.classList.add('drawer-open');
}

function closeDrawer() {
  state.drawer = null;
  $('#drawer').classList.remove('is-open');
  $('#drawer').setAttribute('aria-hidden', 'true');
  document.body.classList.remove('drawer-open');
}

function renderDrawer() {
  const target = state.drawer;
  const body = $('#drawerBody');
  if (!target) {
    body.innerHTML = '';
    return;
  }

  const region = target.regionId ? findRegion(target.regionId) : null;
  const field = target.field || null;

  $('#drawerKicker').textContent = field ? '字段详情' : '版面区域';
  $('#drawerTitle').textContent = field ? field.name : (region ? region.id : '—');
  $('#drawerSub').textContent = field
    ? `值：${field.value}`
    : (region ? `第 ${region.page} 页 · ${CATEGORY_META[region.category] || region.category || ''}` : '');

  let html = '';

  if (field) {
    const confidence = toNumber(field.confidence);
    const sev = severityMeta(field.severity);
    const st = statusMeta(field.status);
    html += `<section class="dsection">
      <div class="dsection-title">字段概要</div>
      <div class="meta-grid">
        <div class="meta-cell"><span>字段值</span><b>${esc(field.value)}</b></div>
        <div class="meta-cell"><span>置信度</span><b>${fmt(confidence, 4)}</b></div>
        <div class="meta-cell"><span>风险等级</span>${tag(sev.label, sev.cls)}</div>
        <div class="meta-cell"><span>状态</span>${tag(st.label, st.cls)}</div>
        <div class="meta-cell is-wide"><span>抽取来源</span><b>${esc(field.source || '—')}</b></div>
        ${field.note ? `<div class="meta-cell is-wide"><span>备注</span><b>${esc(field.note)}</b></div>` : ''}
      </div>
    </section>`;
  }

  const regionIds = field && field.region_ids ? field.region_ids : (region ? [region.id] : []);
  if (regionIds.length > 1) {
    html += `<section class="dsection">
      <div class="dsection-title">证据区域</div>
      <div class="region-switch">
        ${regionIds
          .map((rid) => `<button type="button" class="rid${rid === target.regionId ? ' is-active' : ''}" data-region="${esc(rid)}">${esc(rid)}</button>`)
          .join('')}
      </div>
    </section>`;
  }

  if (!region) {
    html += '<div class="empty"><p class="empty-title">这个字段还没有对应的原文区域</p><p class="empty-hint">说明抽取阶段没能在版面上找到它的出处。</p></div>';
    body.innerHTML = html;
    bindDrawerRegionSwitches();
    return;
  }

  const category = CATEGORY_META[region.category] || region.category || '—';
  const bbox = region.bbox || {};
  const st = statusMeta(region.status);

  html += `<section class="dsection">
    <div class="dsection-title">
      <span>原图裁剪 · ${esc(region.id)}</span>
      <span class="muted">${esc(category)} · 第 ${esc(region.page)} 页 · ${esc(st.label)}</span>
    </div>
    <div class="crop-box" id="cropBox"></div>
    <div class="meta-grid" style="margin-top:8px">
      <div class="meta-cell"><span>坐标 x</span><b>${fmt(bbox.x0, 3)} → ${fmt(bbox.x1, 3)}</b></div>
      <div class="meta-cell"><span>坐标 y</span><b>${fmt(bbox.y0, 3)} → ${fmt(bbox.y1, 3)}</b></div>
    </div>
    <button type="button" class="btn btn-ghost locate-btn" id="locateBtn">在原图中定位</button>
  </section>`;

  html += `<section class="dsection">
    <div class="dsection-title">原文</div>
    <div class="raw-text">${esc(region.text || '')}</div>
  </section>`;

  html += `<section class="dsection">
    <div class="dsection-title">识别候选与 Jev 评分</div>
    ${candidatesHtml(region)}
  </section>`;

  html += `<section class="dsection">
    <div class="dsection-title">决策理由</div>
    <div class="note-box">${esc(region.note || '该区域没有单独的决策记录，按现有证据放行。')}</div>
  </section>`;

  body.innerHTML = html;

  loadCrop($('#cropBox'), region.id);
  const locateBtn = $('#locateBtn');
  if (locateBtn) locateBtn.addEventListener('click', () => openLocate(region));
  bindDrawerRegionSwitches();
}

function bindDrawerRegionSwitches() {
  $$('#drawerBody .region-switch .rid[data-region]').forEach((chip) => {
    chip.addEventListener('click', () => {
      if (!state.drawer) return;
      state.drawer.regionId = chip.getAttribute('data-region');
      renderDrawer();
    });
  });
}

function candidatesHtml(region) {
  const candidates = region.candidates || [];
  if (!candidates.length) {
    return '<p class="muted" style="font-size:12.5px">该区域没有候选记录。</p>';
  }
  const bestIndex = bestCandidateIndex(region);

  return `<ul class="cand-list">${candidates
    .map((candidate, index) => {
      const verdict = CANDIDATE_VERDICT_META[candidate.verdict] || { label: candidate.verdict || '未知', cls: 'muted' };
      const scored = candidate.verdict === 'scored';
      const score = scored ? fmt(candidate.score, 3) : '—';
      return `<li class="cand${index === bestIndex ? ' is-best' : ''}">
        <div class="cand-head">
          <span class="cand-origin">${esc(originLabel(candidate.origin))}</span>
          ${tag(verdict.label, verdict.cls)}
          ${index === bestIndex ? tag('当前采用', 'ok') : ''}
        </div>
        <div class="cand-text">${esc(candidate.text || '')}</div>
        <div class="cand-meta">
          <span>Jev 分数 <b>${score}</b></span>
          ${candidate.detail ? `<span class="cand-detail">${esc(candidate.detail)}</span>` : ''}
        </div>
      </li>`;
    })
    .join('')}</ul>`;
}

function bestCandidateIndex(region) {
  const candidates = region.candidates || [];
  let index = -1;
  let best = -Infinity;
  candidates.forEach((candidate, i) => {
    if (candidate.verdict !== 'scored') return;
    const score = toNumber(candidate.score);
    if (score !== null && score > best) {
      best = score;
      index = i;
    }
  });
  if (index >= 0) return index;
  const primary = candidates.findIndex((candidate) => candidate.origin === 'primary');
  if (primary >= 0) return primary;
  return candidates.length ? 0 : -1;
}

/* ── 图片加载（带加载态与失败兜底） ────────────────────────────────── */
function loadCrop(box, regionId) {
  if (!box) return;
  box.innerHTML = '<div class="img-state"><span class="spinner"></span>正在裁剪原图…</div>';
  const image = new Image();
  image.alt = `区域 ${regionId} 的原图裁剪`;
  image.addEventListener('load', () => {
    box.innerHTML = '';
    box.appendChild(image);
  });
  image.addEventListener('error', () => {
    box.innerHTML = '<div class="img-state img-error">裁剪图加载失败<span class="img-hint">原图可能尚未生成，或该区域没有可裁剪的内容。</span></div>';
  });
  image.src = API.crop(state.docId, regionId);
}

/* ── 原图定位 ──────────────────────────────────────────────────────── */
function openLocate(region) {
  const stage = $('#locateStage');
  const foot = $('#locateFoot');
  stage.innerHTML = '<div class="img-state"><span class="spinner"></span>正在载入整页原图…</div>';

  const bbox = region.bbox || {};
  foot.innerHTML = `<span>区域 ${esc(region.id)}</span><span>坐标 ${esc(fmt(bbox.x0, 3))}, ${esc(fmt(bbox.y0, 3))}, ${esc(fmt(bbox.x1, 3))}, ${esc(fmt(bbox.y1, 3))}</span>`;
  $('#locateTitle').textContent = `第 ${region.page} 页 · ${region.id}`;

  const image = new Image();
  image.className = 'locate-img';
  image.alt = '整页原图';
  image.addEventListener('load', () => {
    stage.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'locate-wrap';
    wrap.appendChild(image);

    const box = document.createElement('div');
    box.className = 'locate-box';
    box.setAttribute('data-label', region.id);
    const x0 = clamp01(bbox.x0);
    const y0 = clamp01(bbox.y0);
    const x1 = clamp01(bbox.x1);
    const y1 = clamp01(bbox.y1);
    box.style.left = x0 * 100 + '%';
    box.style.top = y0 * 100 + '%';
    box.style.width = Math.max(0, x1 - x0) * 100 + '%';
    box.style.height = Math.max(0, y1 - y0) * 100 + '%';
    wrap.appendChild(box);

    stage.appendChild(wrap);
  });
  image.addEventListener('error', () => {
    stage.innerHTML = '<div class="img-state img-error">原图加载失败<span class="img-hint">这份文档可能没有保存原图。</span></div>';
  });
  image.src = API.image(state.docId);

  openModal('locateModal');
}

/* ── 模态 ──────────────────────────────────────────────────────────── */
function openModal(id) {
  const modal = document.getElementById(id);
  if (!modal) return;
  modal.hidden = false;
}

function closeModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.hidden = true;
}

/* ── 上传 ──────────────────────────────────────────────────────────── */
async function handleUpload(event) {
  event.preventDefault();
  const input = $('#uploadFile');
  const schema = $('#uploadSchema').value;
  const submit = $('#uploadSubmit');
  const file = input.files && input.files[0];

  if (!file) {
    showToast('请先选择一张图片', 'warn');
    return;
  }

  const form = new FormData();
  form.append('file', file);
  form.append('schema', schema);

  submit.disabled = true;
  submit.textContent = '核验中…';
  $('#uploadHint').textContent = '正在解析版面并跑核验流程，请稍候。';

  try {
    const response = await fetch(API.analyze, { method: 'POST', body: form });
    if (!response.ok) throw new Error(await readError(response));
    const ledger = await response.json();

    const docId = ledger.doc_id || file.name.replace(/\.[^.]+$/, '');
    if (!state.localDocs.some((doc) => doc.doc_id === docId)) {
      state.localDocs.unshift({
        doc_id: docId,
        source: file.name,
        label: file.name,
        schema: schema,
        description: '本地上传',
      });
    }

    closeModal('uploadModal');
    input.value = '';
    $('#uploadHint').textContent = '核验在服务端执行，完成后自动切换到该文档。';

    renderDocLists();
    await selectDocument(docId, ledger);
    showToast(`《${docId}》核验完成`, 'ok');
  } catch (err) {
    $('#uploadHint').textContent = `核验失败：${err.message}`;
    showToast(`核验失败：${err.message}`, 'error');
  } finally {
    submit.disabled = false;
    submit.textContent = '开始核验';
  }
}

/* ── 导出 CSV ──────────────────────────────────────────────────────── */
function exportCsv() {
  const ledger = state.ledger;
  if (!ledger || !ledger.csv) {
    showToast('当前没有可导出的台账', 'warn');
    return;
  }
  const blob = new Blob(['﻿' + ledger.csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${ledger.doc_id || 'ledger'}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/* ── 入口 ──────────────────────────────────────────────────────────── */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

/* CV/ML Atlas — router, renderer, search, favourites, compare, news */
(function () {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* ---------------- storage ---------------- */
  const store = {
    get(k, d) { try { const v = localStorage.getItem('cvml.' + k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('cvml.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } }
  };
  let favModels = store.get('favModels', []);   // ["boardId|model name"]
  let favCards = store.get('favCards', []);     // ["full item id"]
  let customModels = store.get('customModels', []); // [{board, n, d, y, m, params, flops, lat, notes}]
  customModels.forEach(c => { if (!c.d && c.y) c.d = String(c.y); });

  /* ---------------- dates & NEW tags ---------------- */
  const NEW_DAYS = 30;
  const daysAgo = s => s ? (Date.now() - new Date(s).getTime()) / 864e5 : Infinity;
  const isNew = s => daysAgo(s) <= NEW_DAYS;
  const fmtD = d => { if (!d) return '—'; const [y, m] = String(d).split('-'); return m ? new Date(+y, +m - 1, 1).toLocaleString('en', { month: 'short' }) + ' ' + y : y; };
  const newTag = s => isNew(s) ? ` <span class="new" title="Added ${esc(s)}">NEW</span>` : '';
  const bestOf = (b, xs) => (b.lower ? Math.min : Math.max)(...xs.filter(x => typeof x === 'number'));
  const byQuality = b => (x, y) => b.lower ? x.m - y.m : y.m - x.m;

  /* ---------------- tabs ---------------- */
  const builtinHome = { id: 'home', title: 'Home', icon: '⌂', builtin: true, sections: [] };
  const builtinCompare = { id: 'compare', title: 'Favourites & Compare', short: 'Compare', icon: '★', builtin: true, sections: [] };
  const builtinNews = { id: 'news', title: 'AI / ML News', short: 'News', icon: '⚡', builtin: true, sections: [] };
  const builtinZoo = { id: 'arch', title: 'Model Zoo (architectures)', short: 'Model Zoo', icon: '⧉', builtin: true, sections: [] };
  const TABS = [builtinHome, ...CV.tabs, builtinZoo, builtinCompare, builtinNews];
  const TASKS = { det: 'Detection', seg: 'Segmentation', cls: 'Classification & backbones', pose: 'Pose', gen: 'Generative' };
  const archById = Object.fromEntries(CV.archs.map(a => [a.id, a]));
  const tabById = Object.fromEntries(TABS.map(t => [t.id, t]));

  // assign globally unique ids
  CV.tabs.forEach(tab => tab.sections.forEach(sec => {
    sec.fid = `${tab.id}--${sec.id}`;
    sec.items.forEach(it => { it.fid = `${tab.id}--${sec.id}--${it.id}`; it.tab = tab; it.sec = sec; });
  }));

  function boardRows(bid) {
    const b = CV.boards[bid];
    const custom = customModels.filter(c => c.board === bid).map(c => Object.assign({ custom: true, fam: 'Custom' }, c));
    return b.rows.concat(custom);
  }

  /* ---------------- search index ---------------- */
  const stem = w => w.length > 4 && w.endsWith('ies') ? w.slice(0, -3) + 'y'
    : w.length > 4 && w.endsWith('es') && !w.endsWith('ses') ? w.slice(0, -1)
      : w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w;
  const toks = s => String(s || '').toLowerCase()
    .replace(/<[^>]+>/g, ' ').replace(/\\[a-z]+/g, ' ').replace(/[^a-z0-9+\-.]+/g, ' ')
    .split(/\s+/).map(w => w.replace(/^[-.]+|[-.]+$/g, '')).filter(w => w.length > 1).map(stem);

  const INDEX = [];
  CV.tabs.forEach(tab => {
    tab.sections.forEach(sec => {
      INDEX.push({ kind: 'section', tab, title: sec.title, path: `${tab.title}`, fid: sec.fid, T: new Set(toks(sec.title)), G: new Set(toks(sec.tags)), X: new Set(), raw: sec.title.toLowerCase() });
      sec.items.forEach(it => {
        const text = [it.s, it.h, it.tip, (it.f || []).join(' '), it.code].join(' ');
        INDEX.push({ kind: 'item', tab, title: it.t, path: `${tab.title} › ${sec.title}`, fid: it.fid, snippet: it.s || '', T: new Set(toks(it.t)), G: new Set(toks(it.tags)), X: new Set(toks(text)), raw: it.t.toLowerCase() });
      });
    });
    (tab.boards || []).forEach(bid => {
      const b = CV.boards[bid];
      INDEX.push({ kind: 'board', tab, title: b.title, path: `${tab.title} › Leaderboard`, fid: `${tab.id}--lb-${bid}`, T: new Set(toks(b.title + ' leaderboard benchmark sota')), G: new Set(toks(b.metricLabel)), X: new Set(toks(b.rows.map(r => r.n + ' ' + (r.fam || '')).join(' '))), raw: b.title.toLowerCase() });
      b.rows.forEach(r => INDEX.push({ kind: 'model', tab, title: r.n, path: `${tab.title} › ${b.title}`, fid: `${tab.id}--lb-${bid}`, rowKey: r.n, snippet: `${b.metricLabel}: ${r.m} · released ${fmtD(r.d)}`, T: new Set(toks(r.n)), G: new Set(toks(r.fam)), X: new Set(toks(r.notes)), raw: r.n.toLowerCase() }));
    });
  });
  CV.archs.forEach(a => {
    const text = [a.tagline, (a.sections || []).map(s => s.h + ' ' + s.html).join(' '), (a.pipeline || []).map(s => [s.t, s.d, (s.b || []).join(' ')].join(' ')).join(' ')].join(' ');
    INDEX.push({ kind: 'arch', tab: builtinZoo, title: a.name + ' architecture', path: `Model Zoo › ${TASKS[a.task] || a.task}`, fid: a.id, snippet: a.tagline || '',
      T: new Set(toks(a.name + ' ' + (a.aka || ''))), G: new Set(toks((a.tags || '') + ' architecture backbone neck head ' + (TASKS[a.task] || ''))), X: new Set(toks(text)), raw: a.name.toLowerCase() });
  });

  function lev(a, b, max) {
    if (Math.abs(a.length - b.length) > max) return max + 1;
    let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i]; let best = i;
      for (let j = 1; j <= b.length; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
        best = Math.min(best, cur[j]);
      }
      if (best > max) return max + 1;
      prev = cur;
    }
    return prev[b.length];
  }
  function matchSet(q, set) {
    if (set.has(q)) return 3;
    let pre = false, fz = false;
    const maxd = q.length >= 7 ? 2 : q.length >= 4 ? 1 : 0;
    for (const w of set) {
      if (!pre && w.startsWith(q) && q.length >= 2) pre = true;
      else if (!fz && maxd && lev(q, w, maxd) <= maxd) fz = true;
      if (pre) break;
    }
    return pre ? 2 : fz ? 1 : 0;
  }
  function search(query, onlyTab) {
    const q = toks(query);
    if (!q.length) return [];
    const phrase = query.toLowerCase().trim();
    const out = [];
    for (const e of INDEX) {
      if (onlyTab && e.tab.id !== onlyTab) continue;
      let score = 0, hits = 0;
      for (const w of q) {
        const t = matchSet(w, e.T), g = matchSet(w, e.G), x = e.X.size ? matchSet(w, e.X) : 0;
        const s = t * 5 + g * 3 + x * 1;
        if (s > 0) hits++;
        score += s;
      }
      if (!hits) continue;
      if (hits < q.length) score *= 0.25 * hits / q.length; // partial matches rank low
      if (e.raw === phrase) score += 40; else if (e.raw.includes(phrase)) score += 20;
      if (e.kind === 'section') score *= 1.25;
      if (e.kind === 'model') score *= 0.8;
      if (current && e.tab.id === current.id) score *= 1.3;
      out.push({ e, score, full: hits === q.length });
    }
    const full = out.filter(o => o.full);
    return (full.length ? full : out).sort((a, b) => b.score - a.score).slice(0, 40);
  }

  /* ---------------- search UI ---------------- */
  const qEl = $('#q'), resEl = $('#results'), scopeEl = $('#scopeTab');
  let sel = 0, lastRes = [];
  function showResults() {
    const v = qEl.value.trim();
    if (!v) { resEl.classList.add('hidden'); return; }
    const only = scopeEl.checked && current && !current.builtin ? current.id : null;
    lastRes = search(v, only);
    sel = 0;
    if (!lastRes.length) { resEl.innerHTML = `<div class="rnone">No matches for “${esc(v)}”${only ? ' in this tab — untick “this tab only”' : ''}.</div>`; resEl.classList.remove('hidden'); return; }
    resEl.innerHTML = lastRes.map((r, i) => `<a class="ritem ${i === 0 ? 'sel' : ''}" data-i="${i}" href="#/${r.e.tab.id}/${r.e.fid}${r.e.rowKey ? '/' + encodeURIComponent(r.e.rowKey) : ''}">
      <span class="rkind k-${r.e.kind}">${r.e.kind}</span>
      <span class="rtitle">${esc(r.e.title)}</span>
      <span class="rpath">${esc(r.e.path)}</span>
      ${r.e.snippet ? `<span class="rsnip">${esc(r.e.snippet).slice(0, 140)}</span>` : ''}</a>`).join('');
    resEl.classList.remove('hidden');
  }
  function moveSel(d) {
    const items = $$('.ritem', resEl); if (!items.length) return;
    items[sel].classList.remove('sel');
    sel = (sel + d + items.length) % items.length;
    items[sel].classList.add('sel'); items[sel].scrollIntoView({ block: 'nearest' });
  }
  qEl.addEventListener('input', showResults);
  scopeEl.addEventListener('change', showResults);
  qEl.addEventListener('focus', showResults);
  qEl.addEventListener('keydown', ev => {
    if (ev.key === 'ArrowDown') { ev.preventDefault(); moveSel(1); }
    else if (ev.key === 'ArrowUp') { ev.preventDefault(); moveSel(-1); }
    else if (ev.key === 'Enter') { const a = $$('.ritem', resEl)[sel]; if (a) { location.hash = a.getAttribute('href'); resEl.classList.add('hidden'); qEl.blur(); } }
    else if (ev.key === 'Escape') { resEl.classList.add('hidden'); qEl.blur(); }
  });
  resEl.addEventListener('click', ev => { if (ev.target.closest('.ritem')) resEl.classList.add('hidden'); });
  document.addEventListener('click', ev => { if (!ev.target.closest('.searchwrap')) resEl.classList.add('hidden'); });
  document.addEventListener('keydown', ev => {
    if ((ev.key === '/' && !/input|textarea|select/i.test(document.activeElement.tagName)) || (ev.key.toLowerCase() === 'k' && (ev.ctrlKey || ev.metaKey))) { ev.preventDefault(); qEl.focus(); qEl.select(); }
  });

  /* ---------------- theme ---------------- */
  // No stored choice → follow the system / host theme (CSS handles it). The button stores an explicit choice.
  const effectiveTheme = () => document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const savedTheme = store.get('theme', null);
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  $('#themeBtn').onclick = () => { const t = effectiveTheme() === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = t; store.set('theme', t); };

  /* ---------------- mobile section drawer ---------------- */
  const setDrawer = open => document.body.classList.toggle('toc-open', open);
  $('#menuBtn').onclick = () => setDrawer(!document.body.classList.contains('toc-open'));
  $('#toc').addEventListener('click', ev => { if (ev.target.closest('a')) setDrawer(false); });
  $('.layout').addEventListener('click', ev => { if (!ev.target.closest('#toc')) setDrawer(false); });

  /* ---------------- rendering ---------------- */
  const tabsEl = $('#tabs'), tocEl = $('#toc'), mainEl = $('#main');
  const newCount = t => t.sections.reduce((a, s) => a + s.items.filter(i => isNew(i.added)).length, 0)
    + (t.boards || []).reduce((a, bid) => a + CV.boards[bid].rows.filter(r => isNew(r.added)).length, 0);
  tabsEl.innerHTML = TABS.map(t => { const n = t.builtin ? 0 : newCount(t); return `<a href="#/${t.id}" data-tab="${t.id}"><span class="ti">${t.icon || '•'}</span>${esc(t.short || t.title)}${n ? `<span class="nb" title="${n} new in the last ${NEW_DAYS} days">${n}</span>` : ''}</a>`; }).join('');
  let current = null;

  function typeset(el) {
    if (window.renderMathInElement) {
      renderMathInElement(el, { output: window.CV_MATH_OUTPUT || 'htmlAndMathml', delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }], throwOnError: false, ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] });
    } else { setTimeout(() => typeset(el), 150); }
  }

  function renderItem(it) {
    const tags = (it.tags || '').split(',').map(s => s.trim()).filter(Boolean);
    const fav = favCards.includes(it.fid);
    return `<article class="card${it.wide ? ' wide' : ''}" id="${it.fid}">
      <header><h3>${esc(it.t)}${newTag(it.added)}${it.updated && isNew(it.updated) && it.updated !== it.added ? ' <span class="upd">UPDATED</span>' : ''}</h3>
        <button class="star ${fav ? 'on' : ''}" data-card="${it.fid}" title="Bookmark">${fav ? '★' : '☆'}</button>
        <a class="plink" href="#/${it.tab.id}/${it.fid}" title="Link to this card">#</a></header>
      ${tags.length ? `<div class="tags">${tags.map(t => `<span>${esc(t)}</span>`).join('')}</div>` : ''}
      ${it.s ? `<p class="sum">${it.s}</p>` : ''}
      ${(it.f || []).map(f => `<div class="f">$$${esc(f)}$$</div>`).join('')}
      ${it.dg ? `<div class="dgwrap">${CVDiagram.pipeline(it.dg)}</div>` : ''}
      ${it.dgc ? `<div class="dgrow">${it.dgc.map(c => `<div>${CVDiagram.chain(c)}</div>`).join('')}</div>` : ''}
      ${it.lr ? `<div class="widget lrmini" data-widget="lrcurve" data-args="${esc(JSON.stringify(it.lr))}"></div>` : ''}
      ${it.h ? `<div class="body">${it.h}</div>` : ''}
      ${it.widget ? `<div class="widget" data-widget="${esc(it.widget)}" data-args="${esc(JSON.stringify(it.wargs || {}))}"></div>` : ''}
      ${it.tip ? `<div class="tip"><b>Practical</b> ${it.tip}</div>` : ''}
      ${it.code ? `<details class="code"><summary>Implementation</summary><pre><code>${esc(it.code.replace(/^\n/, ''))}</code></pre></details>` : ''}
      ${it.added || it.src ? `<div class="cardfoot">${it.added ? `added ${esc(it.added)}` : ''}${it.updated && it.updated !== it.added ? ` · updated ${esc(it.updated)}` : ''}${it.src ? ` · <a href="${esc(it.src)}" target="_blank" rel="noopener noreferrer">source</a>` : ''}</div>` : ''}
    </article>`;
  }

  // What each leaderboard column means (shown as header tooltips and in the "What do these columns mean?" box).
  const GLOSS = {
    AP: ['COCO AP@[.5:.95]', 'Primary COCO metric: area under the precision–recall curve, averaged over 10 IoU thresholds (0.50–0.95) and all classes.'],
    ap50: ['AP50', 'Average precision at IoU ≥ 0.50 (loose localization, PASCAL-VOC style). Closest single number to “precision across all recall levels”.', 1],
    ap75: ['AP75', 'Average precision at IoU ≥ 0.75 (strict localization).', 1],
    aps: ['APs', 'AP on small objects (area < 32² px).', 1], apm: ['APm', 'AP on medium objects (32²–96² px).', 1], apl: ['APl', 'AP on large objects (> 96² px).', 1],
    ar: ['AR', 'Average recall (max 100 detections / people per image), averaged over IoU/OKS 0.50–0.95. The standard recall figure for COCO.', 1],
    mask: ['Mask AP', 'COCO AP computed with mask IoU instead of box IoU.', 1], mask50: ['Mask AP50', 'Mask AP at IoU ≥ 0.50.', 1], mask75: ['Mask AP75', 'Mask AP at IoU ≥ 0.75.', 1],
    top5: ['Top-5', 'Accuracy when the true class is among the 5 highest-scoring classes. Top-1 = plain accuracy.', 1],
    ms: ['mIoU (ms)', 'mIoU with multi-scale + flip test-time augmentation.', 1],
    params: ['Params (M)', 'Trainable parameters in millions.'], flops: ['GFLOPs', 'Giga floating-point operations per image at the listed input size (some papers report GMACs).'],
    lat: ['Latency (ms)', 'Inference time per image on the stated GPU/runtime; see the board note for hardware and batch size.'],
    coreml: ['iPhone ms', 'CoreML latency on iPhone 12 Pro Max (A14), batch 1.'], res: ['Test res', 'Input resolution at test time.'], ep: ['Epochs', 'Training epochs.'], nfe: ['Sampling steps', 'Network evaluations per generated image.']
  };
  const fmtCell = (k, v) => typeof v === 'number' && (GLOSS[k] || [])[2] && Number.isInteger(v) ? v.toFixed(1) : (v ?? '—');
  function boardGlossary(b) {
    const keys = (b.cols || []).map(c => c.k).filter(k => GLOSS[k]);
    return `<details class="gloss"><summary>What do these columns mean? (precision, recall, accuracy…)</summary><dl>
      <dt>${esc(b.metric)}</dt><dd>${esc(b.metricLabel)}.${b.metric === 'AP' || b.metric === 'Mask AP' ? ' ' + GLOSS.AP[1] : ''}</dd>
      ${keys.map(k => `<dt>${esc(GLOSS[k][0])}</dt><dd>${esc(GLOSS[k][1])}</dd>`).join('')}
      <dt>Precision / recall at one threshold?</dt><dd>Papers do not publish a single precision or recall value, because both depend on the confidence threshold you pick. AP is precision averaged over all recall levels, and AR is the recall figure. To get P, R and F1 at your own threshold, run validation on your data (Ultralytics <code>model.val()</code> prints them; pycocotools gives the full PR curves).</dd>
      <dt>— (dash)</dt><dd>The official paper, repo or docs do not publish that number for this model. Values are never estimated or copied from third-party re-evaluations, because those use different protocols.</dd>
      ${b.task === 'seg' ? '<dt>Accuracy (segmentation)</dt><dd>Pixel accuracy (aAcc) and mean class accuracy (mAcc) are rarely reported for these models; mIoU = TP/(TP+FP+FN) is the standard.</dd>' : ''}
    </dl></details>`;
  }
  function renderBoard(tab, bid) {
    const b = CV.boards[bid];
    const rows = boardRows(bid).slice().sort((x, y) => String(y.d || '').localeCompare(String(x.d || '')) || byQuality(b)(x, y));
    const cols = b.cols || [];
    const nNew = rows.filter(r => isNew(r.added)).length;
    return `<section class="board" id="${tab.id}--lb-${bid}">
      <h2>🏆 ${esc(b.title)}${nNew ? ` <span class="new">${nNew} NEW</span>` : ''} <small>${esc(b.metricLabel)} · newest release first · tap a model for its architecture · click headers to sort · ☆ to favourite</small></h2>
      ${b.note ? `<p class="note">${b.note}</p>` : ''}
      ${boardGlossary(b)}
      <div class="tblwrap"><table class="lb" data-board="${bid}">
        <thead><tr><th></th><th data-k="n">Model</th><th data-k="d" data-dir="desc">Released</th><th data-k="fam">Family</th><th data-k="m" class="num" title="${esc(b.metricLabel)}">${esc(b.metric)}${b.lower ? ' ↓' : ' ↑'}</th>${cols.map(c => `<th data-k="${c.k}" class="num" title="${esc((GLOSS[c.k] || [])[1] || '')}">${esc(c.label)}</th>`).join('')}<th>Notes</th></tr></thead>
        <tbody>${rows.map(r => boardRow(bid, b, r)).join('')}</tbody></table></div>
    </section>`;
  }
  function boardRow(bid, b, r) {
    const key = bid + '|' + r.n, fav = favModels.includes(key), a = CV.archFor(r.n);
    return `<tr data-row="${esc(r.n)}" class="${fav ? 'fav' : ''}${isNew(r.added) ? ' isnew' : ''}">
      <td><button class="star ${fav ? 'on' : ''}" data-model="${esc(key)}">${fav ? '★' : '☆'}</button></td>
      <td><b>${a ? `<a class="mlink" href="#/arch/${a.id}" title="${esc(a.name)} architecture: backbone, neck, head">${esc(r.n)}</a>` : esc(r.n)}</b>${newTag(r.added)}${r.custom ? ' <span class="cst">custom</span>' : ''}${r.src ? ` <a class="src" href="${esc(r.src)}" target="_blank" rel="noopener noreferrer" title="source">↗</a>` : ''}</td>
      <td class="nowrap">${fmtD(r.d)}</td><td>${esc(r.fam || '')}</td><td class="num"><b>${typeof r.m === 'number' && Number.isInteger(r.m) ? r.m.toFixed(1) : r.m ?? '—'}</b></td>
      ${(b.cols || []).map(c => `<td class="num">${fmtCell(c.k, r[c.k])}</td>`).join('')}
      <td class="nt">${esc(r.notes || '')}</td></tr>`;
  }
  function sortTable(th) {
    const table = th.closest('table'), k = th.dataset.k; if (!k) return;
    const bid = table.dataset.board, b = CV.boards[bid];
    const dir = th.dataset.dir === 'desc' ? 'asc' : 'desc';
    $$('th', table).forEach(x => delete x.dataset.dir); th.dataset.dir = dir;
    const rows = boardRows(bid).slice().sort((x, y) => {
      let a = x[k], c = y[k];
      if (a == null) return 1; if (c == null) return -1;
      if (typeof a === 'string') return dir === 'asc' ? a.localeCompare(c) : c.localeCompare(a);
      return dir === 'asc' ? a - c : c - a;
    });
    $('tbody', table).innerHTML = rows.map(r => boardRow(bid, b, r)).join('');
  }

  function renderTab(tab) {
    current = tab;
    $$('#tabs a').forEach(a => a.classList.toggle('active', a.dataset.tab === tab.id));
    document.title = `${tab.title} · CV/ML Atlas`;
    if (tab.id === 'home') return renderHome();
    if (tab.id === 'compare') return renderCompare();
    if (tab.id === 'news') return renderNews();
    if (tab.id === 'arch') return renderZoo();

    tocEl.innerHTML = `<input class="tabfilter" placeholder="Filter this tab…" value="">
      ${(tab.boards || []).map(bid => `<a class="toc-sec" href="#/${tab.id}/${tab.id}--lb-${bid}">🏆 ${esc(CV.boards[bid].title)}</a>`).join('')}
      ${tab.sections.map(sec => `<a class="toc-sec" href="#/${tab.id}/${sec.fid}">${esc(sec.title)}</a>
        ${sec.items.map(it => `<a class="toc-it" data-fid="${it.fid}" href="#/${tab.id}/${it.fid}">${esc(it.t)}</a>`).join('')}`).join('')}`;
    mainEl.innerHTML = `<div class="tabhead"><h1>${tab.icon || ''} ${esc(tab.title)}</h1>${tab.intro ? `<p>${tab.intro}</p>` : ''}</div>
      ${(tab.boards || []).map(bid => renderBoard(tab, bid)).join('')}
      ${tab.sections.map(sec => `<section class="sec" id="${sec.fid}"><h2>${esc(sec.title)}</h2>${sec.intro ? `<p class="secintro">${sec.intro}</p>` : ''}
        <div class="cards">${sec.items.map(renderItem).join('')}</div></section>`).join('')}`;
    typeset(mainEl);
    if (window.CVWidgets) CVWidgets.mountAll(mainEl);
    const tf = $('.tabfilter', tocEl);
    tf.addEventListener('input', () => filterTab(tab, tf.value));
  }

  function filterTab(tab, v) {
    const q = v.trim();
    if (!q) { $$('.card', mainEl).forEach(c => c.classList.remove('hide')); $$('.sec', mainEl).forEach(s => s.classList.remove('hide')); $$('.toc-it', tocEl).forEach(a => a.classList.remove('hide')); return; }
    const keep = new Set(search(q, tab.id).map(r => r.e.fid));
    // a matching section keeps all of its cards
    tab.sections.forEach(sec => { if (keep.has(sec.fid)) sec.items.forEach(it => keep.add(it.fid)); });
    $$('.card', mainEl).forEach(c => c.classList.toggle('hide', !keep.has(c.id)));
    $$('.toc-it', tocEl).forEach(a => a.classList.toggle('hide', !keep.has(a.dataset.fid)));
    $$('.sec', mainEl).forEach(s => s.classList.toggle('hide', !$$('.card:not(.hide)', s).length));
  }

  /* ---------------- home ---------------- */
  function renderHome() {
    tocEl.innerHTML = `<div class="toc-h">Tabs</div>` + TABS.slice(1).map(t => `<a class="toc-sec" href="#/${t.id}">${t.icon || ''} ${esc(t.title)}</a>`).join('');
    const nItems = INDEX.filter(e => e.kind === 'item').length;
    const snap = Object.values(CV.boards).map(b => {
      const newest = boardRows(b.id).slice().sort((a, c) => String(c.d).localeCompare(String(a.d)))[0];
      const recent = boardRows(b.id).filter(r => r.y >= (newest ? newest.y - 1 : 0));
      const top = recent.sort(byQuality(b)).slice(0, 3);
      const tab = CV.tabs.find(t => (t.boards || []).includes(b.id));
      return `<div class="snap"><a href="#/${tab.id}/${tab.id}--lb-${b.id}"><h4>${esc(b.title)}</h4></a><small>${esc(b.metricLabel)} · best released since ${newest ? newest.y - 1 : ''}</small>
        <ol>${top.map(r => `<li><a href="#/${tab.id}/${tab.id}--lb-${b.id}/${encodeURIComponent(r.n)}"><b>${esc(r.n)}</b></a> <span>${r.m}</span> <em>${fmtD(r.d)}</em>${newTag(r.added)}</li>`).join('')}</ol></div>`;
    }).join('');
    // What's new: items + models added in the window, and the changelog
    const fresh = [];
    CV.tabs.forEach(t => t.sections.forEach(s => s.items.forEach(i => { if (isNew(i.added) || isNew(i.updated)) fresh.push({ date: i.updated || i.added, html: `<a href="#/${t.id}/${i.fid}">${esc(i.t)}</a> <small>${esc(t.short || t.title)} › ${esc(s.title)}</small>`, upd: i.updated && i.updated !== i.added }); })));
    Object.values(CV.boards).forEach(b => { const tab = CV.tabs.find(t => (t.boards || []).includes(b.id)); b.rows.forEach(r => { if (isNew(r.added)) fresh.push({ date: r.added, html: `<a href="#/${tab.id}/${tab.id}--lb-${b.id}/${encodeURIComponent(r.n)}">${esc(r.n)}</a> <small>${esc(b.title)} · ${esc(b.metric)} ${r.m} · released ${fmtD(r.d)}</small>` }); }); });
    fresh.sort((a, c) => String(c.date).localeCompare(String(a.date)));
    const log = CV.changelog.slice().sort((a, c) => String(c.date).localeCompare(String(a.date))).slice(0, 12);
    const whatsNew = `<h2>What's new <small class="note">content last updated ${esc(CV.contentUpdated || '—')} · refreshed weekly · NEW = added in the last ${NEW_DAYS} days</small></h2>
      ${fresh.length ? `<ul class="fresh">${fresh.slice(0, 30).map(f => `<li><span class="new">${f.upd ? 'UPDATED' : 'NEW'}</span> <em>${esc(f.date)}</em> ${f.html}</li>`).join('')}</ul>` : '<p class="note">Nothing added in the last 30 days.</p>'}
      ${log.length ? `<details class="log"><summary>Update log</summary><ul>${log.map(c => `<li><em>${esc(c.date)}</em> ${c.link ? `<a href="${esc(c.link)}" target="_blank" rel="noopener noreferrer">${esc(c.text)}</a>` : esc(c.text)}</li>`).join('')}</ul></details>` : ''}`;
    const quick = [['Reduce false positives', 'reduce false positives'], ['Reduce false negatives', 'reduce false negatives'], ['LR schedule explorer', 'lr schedule explorer'], ['Which scheduler when?', 'which scheduler'], ['Encoder vs decoder', 'encoder decoder'], ['Few-shot: what to use', 'few-shot decision guide'], ['Distillation process', 'distillation process'], ['Loss functions', 'loss function'], ['Focal loss', 'focal loss'], ['Optimizers', 'optimizer'], ['LR schedulers', 'learning rate scheduler'], ['Warmup', 'warmup'], ['Batch size ↔ LR', 'linear scaling rule'], ['IoU losses', 'giou ciou'], ['NMS', 'non-maximum suppression'], ['mAP', 'mean average precision'], ['Receptive field', 'receptive field'], ['Attention', 'self-attention'], ['Diffusion', 'ddpm'], ['Flow matching', 'flow matching'], ['LoRA', 'lora'], ['Quantization', 'quantization']];
    mainEl.innerHTML = `<div class="hero"><h1>CV/ML Atlas</h1>
      <p>A one-stop reference for computer-vision engineers: <b>formulas first</b>, crisp explanations, detailed layer layouts, leaderboards with the newest models on top, and practical hyperparameter guidance. ${nItems} topic cards across ${CV.tabs.length} tabs.</p>
      <p class="kbd">Press <kbd>/</kbd> or <kbd>Ctrl</kbd>+<kbd>K</kbd> to search. Tick “this tab only” to search within the current tab. Search tolerates typos (“loss fuctions” works).</p><p class="kbd" id="otaInfo">${otaInfoHtml()}</p></div>
      ${whatsNew}
      <h2>Quick jumps</h2><div class="chips">${quick.map(([l, q]) => `<button class="chip" data-q="${esc(q)}">${esc(l)}</button>`).join('')}</div>
      <h2>Tabs</h2><div class="grid">${CV.tabs.map(t => `<a class="tile" href="#/${t.id}"><span class="big">${t.icon || ''}</span><h3>${esc(t.title)}</h3><p>${t.blurb || ''}</p><small>${t.sections.map(s => esc(s.title)).join(' · ')}</small></a>`).join('')}
        <a class="tile" href="#/arch"><span class="big">⧉</span><h3>Model Zoo</h3><p>${CV.archs.length} architectures explained as backbone → neck → head, with block diagrams, layer tables and training recipes.</p></a>
        <a class="tile" href="#/compare"><span class="big">★</span><h3>Favourites & Compare</h3><p>Star models in any leaderboard, add your own architectures, and plot them against the newest SOTA.</p></a>
        <a class="tile" href="#/news"><span class="big">⚡</span><h3>AI / ML News</h3><p>Live feed from arXiv, Hugging Face, lab blogs and tech press (needs the local server).</p></a></div>
      <h2>Newest SOTA snapshot</h2><div class="grid">${snap}</div>
      ${favCards.length ? `<h2>Your bookmarks</h2><div class="chips">${favCards.map(fid => { const it = findItem(fid); return it ? `<a class="chip" href="#/${it.tab.id}/${fid}">${esc(it.t)} <small>${esc(it.tab.short || it.tab.title)}</small></a>` : ''; }).join('')}</div>` : ''}`;
    $$('.chip[data-q]', mainEl).forEach(b => b.onclick = () => { qEl.value = b.dataset.q; scopeEl.checked = false; qEl.focus(); showResults(); });
  }
  function findItem(fid) { for (const t of CV.tabs) for (const s of t.sections) for (const i of s.items) if (i.fid === fid) return i; return null; }

  /* ---------------- compare ---------------- */
  let cmpBoard = store.get('cmpBoard', 'det-coco'), cmpX = store.get('cmpX', 'params');
  function renderCompare() {
    tocEl.innerHTML = `<div class="toc-h">Leaderboards</div>` + Object.values(CV.boards).map(b => `<a class="toc-sec ${b.id === cmpBoard ? 'active' : ''}" href="#/compare" data-b="${b.id}">${esc(b.title)} <small>(${favModels.filter(k => k.startsWith(b.id + '|')).length}★)</small></a>`).join('');
    $$('[data-b]', tocEl).forEach(a => a.onclick = ev => { ev.preventDefault(); cmpBoard = a.dataset.b; store.set('cmpBoard', cmpBoard); renderCompare(); });
    if (!CV.boards[cmpBoard]) cmpBoard = Object.keys(CV.boards)[0];
    const b = CV.boards[cmpBoard];
    const rows = boardRows(cmpBoard);
    const favs = rows.filter(r => favModels.includes(cmpBoard + '|' + r.n));
    const latest = rows.filter(r => !favs.includes(r) && !r.custom).sort((a, c) => String(c.d).localeCompare(String(a.d)) || byQuality(b)(a, c)).slice(0, 6);
    const xcols = (b.cols || []).filter(c => rows.some(r => typeof r[c.k] === 'number'));
    if (!xcols.find(c => c.k === cmpX)) cmpX = xcols[0] ? xcols[0].k : null;
    const allFav = favModels.map(k => { const [bid, n] = k.split('|'); const r = CV.boards[bid] && boardRows(bid).find(x => x.n === n); return r ? { bid, r } : null; }).filter(Boolean);

    mainEl.innerHTML = `<div class="tabhead"><h1>★ Favourites & Compare</h1><p>Star models in any leaderboard (☆), or add your own architecture below. Your favourites are compared against the newest models on the same benchmark so the comparison is apples-to-apples. Saved in this browser.</p></div>
      <div class="cmpbar"><label>Benchmark <select id="cmpB">${Object.values(CV.boards).map(x => `<option value="${x.id}" ${x.id === cmpBoard ? 'selected' : ''}>${esc(x.title)}</option>`).join('')}</select></label>
        ${xcols.length ? `<label>X axis <select id="cmpX">${xcols.map(c => `<option value="${c.k}" ${c.k === cmpX ? 'selected' : ''}>${esc(c.label)}</option>`).join('')}</select></label>` : ''}
        <span class="legend"><i class="lg fav"></i>favourite <i class="lg new"></i>released ≥ ${newestYear(b) - 1} <i class="lg old"></i>older <i class="lg cst"></i>custom</span></div>
      <div class="chart">${cmpX ? scatter(b, rows, cmpX) : '<p class="note">No numeric cost column for this board.</p>'}</div>
      <h2>Your favourites vs the latest (${esc(b.metricLabel)})</h2>
      ${favs.length ? '' : `<p class="note">No favourites on this benchmark yet — open the <a href="#/${CV.tabs.find(t => (t.boards || []).includes(cmpBoard)).id}/${CV.tabs.find(t => (t.boards || []).includes(cmpBoard)).id}--lb-${cmpBoard}">leaderboard</a> and click ☆.</p>`}
      ${cmpTable(b, favs, latest)}
      <h2>Add a custom architecture</h2>
      <form id="addModel" class="addform">
        <input name="n" required placeholder="Model name (e.g. MyNet-S)">
        <input name="d" type="month" value="${new Date().toISOString().slice(0, 7)}" title="Release month">
        <input name="m" type="number" step="any" required placeholder="${esc(b.metric)}">
        ${(b.cols || []).map(c => `<input name="${c.k}" type="number" step="any" placeholder="${esc(c.label)}">`).join('')}
        <input name="notes" placeholder="Notes (backbone, input size, data…)">
        <button>Add to “${esc(b.title)}” + favourite</button></form><p id="addMsg" class="note" role="status"></p>
      <h2>All favourites</h2>
      ${allFav.length ? `<div class="tblwrap"><table class="lt"><thead><tr><th>Model</th><th>Benchmark</th><th>Score</th><th>Released</th><th></th></tr></thead><tbody>${allFav.map(({ bid, r }) => `<tr><td>${esc(r.n)}</td><td>${esc(CV.boards[bid].title)}</td><td>${r.m}</td><td>${fmtD(r.d)}</td><td><button class="star on" data-model="${esc(bid + '|' + r.n)}">★</button>${r.custom ? ` <button class="del" data-del="${esc(bid + '|' + r.n)}">delete</button>` : ''}</td></tr>`).join('')}</tbody></table></div>` : '<p class="note">None yet.</p>'}
      <h2>Bookmarked cards</h2>
      ${favCards.length ? `<div class="chips">${favCards.map(fid => { const it = findItem(fid); return it ? `<a class="chip" href="#/${it.tab.id}/${fid}">${esc(it.t)} <small>${esc(it.tab.short || it.tab.title)}</small></a>` : ''; }).join('')}</div>` : '<p class="note">Click ☆ on any topic card to bookmark it.</p>'}`;
    $('#cmpB').onchange = e => { cmpBoard = e.target.value; store.set('cmpBoard', cmpBoard); renderCompare(); };
    if ($('#cmpX')) $('#cmpX').onchange = e => { cmpX = e.target.value; store.set('cmpX', cmpX); renderCompare(); };
    $('#addModel').onsubmit = e => {
      e.preventDefault();
      const fd = new FormData(e.target), m = { board: cmpBoard };
      for (const [k, v] of fd.entries()) if (v !== '') m[k] = ['n', 'notes', 'd'].includes(k) ? v : Number(v);
      m.y = +String(m.d || '').slice(0, 4) || new Date().getFullYear();
      if (boardRows(cmpBoard).some(r => r.n === m.n)) { $('#addMsg').textContent = `“${m.n}” already exists on this board — pick another name.`; return; }
      customModels.push(m); store.set('customModels', customModels);
      favModels.push(cmpBoard + '|' + m.n); store.set('favModels', favModels);
      renderCompare();
    };
    $$('[data-del]', mainEl).forEach(btn => btn.onclick = () => {
      const [bid, n] = btn.dataset.del.split('|');
      customModels = customModels.filter(c => !(c.board === bid && c.n === n)); store.set('customModels', customModels);
      favModels = favModels.filter(k => k !== btn.dataset.del); store.set('favModels', favModels);
      renderCompare();
    });
  }
  function cmpTable(b, favs, latest) {
    const cols = b.cols || [];
    const best = bestOf(b, favs.concat(latest).map(r => r.m));
    const row = (r, cls) => `<tr class="${cls}"><td>${cls === 'fav' ? '★' : 'latest'}</td><td><b>${esc(r.n)}</b>${newTag(r.added)}</td><td class="nowrap">${fmtD(r.d)}</td>
      <td class="num"><b>${r.m}</b> <span class="delta">${typeof r.m === 'number' ? (r.m === best ? 'best' : (r.m - best > 0 ? '+' : '') + (r.m - best).toFixed(2)) : ''}</span>
        <div class="bar"><i style="width:${Math.max(2, (b.lower ? best / r.m : r.m / best) * 100)}%"></i></div></td>
      ${cols.map(c => `<td class="num">${r[c.k] ?? '—'}</td>`).join('')}<td class="nt">${esc(r.notes || '')}</td></tr>`;
    return `<div class="tblwrap"><table class="lt cmp"><thead><tr><th></th><th>Model</th><th>Released</th><th class="num">${esc(b.metric)}</th>${cols.map(c => `<th class="num">${esc(c.label)}</th>`).join('')}<th>Notes</th></tr></thead>
      <tbody>${favs.map(r => row(r, 'fav')).join('')}${latest.map(r => row(r, 'latest')).join('')}</tbody></table></div>`;
  }
  function scatter(b, rows, xk) {
    const pts = rows.filter(r => typeof r[xk] === 'number' && typeof r.m === 'number');
    if (!pts.length) return '';
    const W = 860, H = 380, P = { l: 56, r: 20, t: 16, b: 44 };
    const log = xk !== 'lat' && Math.max(...pts.map(p => p[xk])) / Math.max(1e-9, Math.min(...pts.map(p => p[xk]))) > 30;
    const fx = v => log ? Math.log10(v) : v;
    const xs = pts.map(p => fx(p[xk])), ys = pts.map(p => p.m);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const px = (x1 - x0) * 0.06 || 1, py = (y1 - y0) * 0.08 || 1; x0 -= px; x1 += px; y0 -= py; y1 += py;
    const sx = v => P.l + (fx(v) - x0) / (x1 - x0) * (W - P.l - P.r), sy = v => H - P.b - (v - y0) / (y1 - y0) * (H - P.t - P.b);
    const xticks = [], yticks = [];
    for (let i = 0; i <= 5; i++) { const v = x0 + (x1 - x0) * i / 5; xticks.push(log ? Math.pow(10, v) : v); yticks.push(y0 + (y1 - y0) * i / 5); }
    const fmt = v => v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
    const label = (b.cols.find(c => c.k === xk) || {}).label || xk;
    const ny = newestYear(b) - 1;
    const dots = pts.map(p => {
      const fav = favModels.includes(b.id + '|' + p.n), cls = p.custom ? 'cst' : fav ? 'fav' : p.y >= ny ? 'new' : 'old';
      return `<g class="pt ${cls}"><circle cx="${sx(p[xk]).toFixed(1)}" cy="${sy(p.m).toFixed(1)}" r="${cls === 'old' ? 4 : 6}"><title>${esc(p.n)} · ${p.m} · ${label} ${p[xk]}</title></circle>
        ${cls !== 'old' ? `<text x="${(sx(p[xk]) + 8).toFixed(1)}" y="${(sy(p.m) + 4).toFixed(1)}">${esc(p.n)}</text>` : ''}</g>`;
    }).join('');
    return `<svg viewBox="0 0 ${W} ${H}" class="scatter" role="img" aria-label="${esc(b.metric)} vs ${esc(label)}">
      ${yticks.map(v => `<line class="grid" x1="${P.l}" x2="${W - P.r}" y1="${sy(v)}" y2="${sy(v)}"/><text class="ax" x="${P.l - 8}" y="${sy(v) + 4}" text-anchor="end">${fmt(v)}</text>`).join('')}
      ${xticks.map(v => `<text class="ax" x="${sx(v)}" y="${H - P.b + 18}" text-anchor="middle">${fmt(v)}</text>`).join('')}
      <text class="axl" x="${(W + P.l) / 2}" y="${H - 6}" text-anchor="middle">${esc(label)}${log ? ' (log)' : ''} → lower is cheaper</text>
      <text class="axl" transform="translate(14 ${H / 2}) rotate(-90)" text-anchor="middle">${esc(b.metric)} ${b.lower ? '(lower is better)' : '↑'}</text>${dots}</svg>`;
  }
  function newestYear(b) { return Math.max(...boardRows(b.id).filter(r => !r.custom).map(r => r.y || 0)); }

  /* ---------------- news ---------------- */
  let newsCache = null, newsFilter = { src: 'all', cat: 'all', q: '' };
  function renderNews() {
    tocEl.innerHTML = `<div class="toc-h">Filter</div><input class="tabfilter" id="newsQ" placeholder="Filter news…" value="${esc(newsFilter.q)}">
      <div class="toc-h">Category</div>${['all', 'Vision', 'Research', 'Industry', 'Open source'].map(c => `<a class="toc-sec ${newsFilter.cat === c ? 'active' : ''}" data-cat="${c}" href="#/news">${c}</a>`).join('')}
      <div id="srcList"></div>`;
    mainEl.innerHTML = `<div class="tabhead"><h1>⚡ AI / ML News</h1><p>Aggregated live from arXiv (cs.CV, cs.LG), Hugging Face daily papers & blog, lab blogs (Google Research, DeepMind, OpenAI, NVIDIA, PyTorch) and tech press. ${window.CVNews && CVNews.available() ? 'Fetched live on your phone (cached 20 minutes); the last copy stays available offline.' : 'Refreshed every 20 minutes by the local server.'}</p>
      <button id="newsRefresh" class="btn">Refresh now</button> <span id="newsStatus" class="note"></span></div><div id="newsList" class="news">Loading…</div>`;
    $('#newsQ').oninput = e => { newsFilter.q = e.target.value; drawNews(); };
    $$('[data-cat]', tocEl).forEach(a => a.onclick = ev => { ev.preventDefault(); newsFilter.cat = a.dataset.cat; $$('[data-cat]', tocEl).forEach(x => x.classList.toggle('active', x === a)); drawNews(); });
    $('#newsRefresh').onclick = () => loadNews(true);
    loadNews(false);
  }
  async function loadNews(force) {
    const st = $('#newsStatus');
    if (newsCache && !force) return drawNews();
    try {
      if (window.CVNews && CVNews.available()) {
        // Android app: fetch feeds natively on the phone; keep a 20-minute cache across launches
        const cached = store.get('newsCache', null);
        if (!force && cached && Date.now() / 1000 - cached.updated < 1200) newsCache = cached;
        else {
          if (st) st.textContent = 'Fetching live feeds…';
          newsCache = await CVNews.fetchAll();
          if (!newsCache.items.length) throw new Error('offline');
          store.set('newsCache', newsCache);
        }
      } else {
        if (location.protocol === 'file:') throw new Error('file');
        const r = await fetch('api/news' + (force ? '?refresh=1' : ''));
        if (!r.ok) throw new Error(r.status);
        newsCache = await r.json();
      }
      if (st) st.textContent = `Updated ${new Date(newsCache.updated * 1000).toLocaleString()} · ${newsCache.items.length} items · ${newsCache.errors.length ? newsCache.errors.length + ' feeds unavailable' : 'all feeds OK'}`;
      drawNews();
    } catch (e) {
      const fallback = store.get('newsCache', null) || CV.newsSnapshot;
      if (fallback && fallback.items && fallback.items.length) {
        newsCache = fallback;
        if (st) st.textContent = `Saved copy from ${new Date(newsCache.updated * 1000).toLocaleString()} · ${newsCache.items.length} items (${window.CVNews && CVNews.available() ? 'offline — tap Refresh when connected' : 'run server.py for a live feed'})`;
        return drawNews();
      }
      const nl = $('#newsList');
      if (nl) nl.innerHTML = `<div class="tip"><b>Live news needs the local server.</b> Run <code>python server.py</code> in the project folder and open <code>http://localhost:8000</code>. To share with others: <code>ngrok http 8000</code>. (The rest of the app works offline from the file.)</div>`;
    }
  }
  function drawNews() {
    if (!newsCache || current.id !== 'news') return;
    const srcs = [...new Set(newsCache.items.map(i => i.source))].sort();
    $('#srcList').innerHTML = `<div class="toc-h">Source</div>` + ['all', ...srcs].map(s => `<a class="toc-it ${newsFilter.src === s ? 'active' : ''}" data-src="${esc(s)}" href="#/news">${esc(s)} ${s === 'all' ? '' : `<small>${newsCache.items.filter(i => i.source === s).length}</small>`}</a>`).join('');
    $$('[data-src]', tocEl).forEach(a => a.onclick = ev => { ev.preventDefault(); newsFilter.src = a.dataset.src; drawNews(); });
    const q = newsFilter.q.toLowerCase();
    const items = newsCache.items.filter(i => (newsFilter.src === 'all' || i.source === newsFilter.src) && (newsFilter.cat === 'all' || (i.cats || []).includes(newsFilter.cat)) && (!q || (i.title + ' ' + i.summary).toLowerCase().includes(q)));
    $('#newsList').innerHTML = items.length ? items.map(i => `<article class="newsitem"><a href="${esc(i.link)}" target="_blank" rel="noopener noreferrer"><h3>${esc(i.title)}${i.date && (Date.now() / 1000 - i.date) < 3 * 86400 ? ' <span class="new">NEW</span>' : ''}</h3></a>
      <div class="meta"><span class="src">${esc(i.source)}</span> ${i.date ? `<span>${new Date(i.date * 1000).toLocaleDateString()}</span>` : ''} ${(i.cats || []).map(c => `<span class="cat">${esc(c)}</span>`).join('')}</div>
      ${i.summary ? `<p>${esc(i.summary)}</p>` : ''}</article>`).join('') : '<p class="note">Nothing matches.</p>';
  }

  /* ---------------- over-the-air content updates (Android app) ---------------- */
  const fmtVer = v => /^\d{12}$/.test(v || '') ? `${v.slice(0, 4)}-${v.slice(4, 6)}-${v.slice(6, 8)} ${v.slice(8, 10)}:${v.slice(10)}` : (v || '—');
  function showBanner(html, sticky) {
    let b = $('#otaBanner');
    if (!b) { b = document.createElement('div'); b.id = 'otaBanner'; b.className = 'banner'; b.setAttribute('role', 'status'); document.body.appendChild(b); }
    b.innerHTML = html; b.hidden = false;
    clearTimeout(b._t); if (!sticky) b._t = setTimeout(() => { b.hidden = true; }, 5000);
    const r = $('#otaReload', b); if (r) r.onclick = () => location.reload();
    const x = $('#otaClose', b); if (x) x.onclick = () => { b.hidden = true; };
  }
  window.CVUpdate = {
    status(state, ver, msg, manual) {
      if (state === 'updated') showBanner(`<span>New content downloaded (${esc(fmtVer(ver))}).</span> <button id="otaReload" class="btn">Reload</button> <button id="otaClose" class="del">Later</button>`, true);
      else if (manual && state === 'current') showBanner(`You have the latest content (${esc(fmtVer(window.AndroidBridge ? AndroidBridge.contentVersion() : ''))}).`);
      else if (manual) showBanner(`Could not check for updates: ${esc(msg || 'offline')}.`);
      const el = $('#otaInfo'); if (el) el.innerHTML = otaInfoHtml();
    }
  };
  function otaInfoHtml() {
    if (!window.AndroidBridge || !AndroidBridge.contentVersion) return '';
    const on = AndroidBridge.updatesEnabled();
    return `Content version <b>${esc(fmtVer(AndroidBridge.contentVersion()))}</b> · ${on ? 'updates download automatically' : 'automatic updates not configured in this build'} ${on ? '<button id="otaCheck" class="chip">Check for updates</button>' : ''}`;
  }
  document.addEventListener('click', ev => {
    if (ev.target.id === 'otaCheck') { ev.target.textContent = 'Checking…'; AndroidBridge.checkUpdate(); }
  });

  /* ---------------- global clicks ---------------- */
  document.addEventListener('click', ev => {
    const s = ev.target.closest('.star');
    if (s && s.dataset.card) {
      const id = s.dataset.card;
      favCards = favCards.includes(id) ? favCards.filter(x => x !== id) : favCards.concat(id);
      store.set('favCards', favCards); s.classList.toggle('on'); s.textContent = s.classList.contains('on') ? '★' : '☆';
    } else if (s && s.dataset.model) {
      const k = s.dataset.model;
      favModels = favModels.includes(k) ? favModels.filter(x => x !== k) : favModels.concat(k);
      store.set('favModels', favModels);
      if (current.id === 'compare') renderCompare();
      else { s.classList.toggle('on'); s.textContent = s.classList.contains('on') ? '★' : '☆'; s.closest('tr').classList.toggle('fav'); }
    }
    const th = ev.target.closest('table.lb th[data-k]');
    if (th) sortTable(th);
  });

  /* ---------------- router ---------------- */
  /* ---------------- model zoo & architecture pages ---------------- */
  const ROLE = { backbone: ['bb', 'Backbone'], neck: ['nk', 'Neck'], head: ['hd', 'Head'], loss: ['ot', 'Loss / training'], extra: ['ot', 'Other'] };
  function archRows(a) {
    const out = [];
    Object.values(CV.boards).forEach(b => boardRows(b.id).forEach(r => { if (CV.archFor(r.n) === a) out.push({ b, r }); }));
    return out;
  }
  function renderZoo() {
    const byTask = {};
    CV.archs.forEach(a => (byTask[a.task] = byTask[a.task] || []).push(a));
    Object.values(byTask).forEach(l => l.sort((x, y) => String(y.date).localeCompare(String(x.date))));
    tocEl.innerHTML = `<input class="tabfilter" id="zooQ" placeholder="Filter architectures…">` +
      Object.keys(TASKS).filter(t => byTask[t]).map(t => `<a class="toc-sec" href="#/arch" data-jump="zoo-${t}">${esc(TASKS[t])} <small>(${byTask[t].length})</small></a>`).join('');
    mainEl.innerHTML = `<div class="tabhead"><h1>⧉ Model Zoo</h1><p>Every architecture on the leaderboards, explained as <span class="role bb">BACKBONE</span> → <span class="role nk">NECK</span> → <span class="role hd">HEAD</span> with block diagrams, layer tables, losses and training recipes. Newest first. Tap a model name in any leaderboard to land here.</p></div>
      ${Object.keys(TASKS).filter(t => byTask[t]).map(t => `<section class="sec" id="zoo-${t}"><h2>${esc(TASKS[t])}</h2><div class="zoo">${byTask[t].map(a => `<a href="#/arch/${a.id}" data-name="${esc((a.name + ' ' + (a.aka || '') + ' ' + (a.tags || '')).toLowerCase())}"><b>${esc(a.name)}</b><small>${esc(a.org || '')} · ${fmtD(a.date)}</small><span>${esc(a.tagline || '')}</span></a>`).join('')}</div></section>`).join('')}`;
    $$('[data-jump]', tocEl).forEach(l => l.onclick = ev => { ev.preventDefault(); document.getElementById(l.dataset.jump).scrollIntoView({ behavior: 'smooth' }); });
    $('#zooQ').oninput = e => { const q = e.target.value.toLowerCase().trim(); $$('.zoo a', mainEl).forEach(x => x.classList.toggle('hide', q && !x.dataset.name.includes(q))); };
  }
  function renderArch(a) {
    current = builtinZoo;
    $$('#tabs a').forEach(x => x.classList.toggle('active', x.dataset.tab === 'arch'));
    document.title = `${a.name} architecture · CV/ML Atlas`;
    const taskTab = CV.tabs.find(t => t.id === a.task);
    const rows = archRows(a);
    const secs = [['anat', 'Anatomy'], ['diagram', 'Block diagram'], ...(a.sections || []).map((s, i) => ['s' + i, s.h]), ...(a.f && a.f.length ? [['eq', 'Key equations']] : []), ...(rows.length ? [['bench', 'Benchmarks']] : [])];
    const same = CV.archs.filter(x => x.task === a.task && x !== a);
    tocEl.innerHTML = `<div class="toc-h">On this page</div>${secs.map(([id, t]) => `<a class="toc-it" href="#/arch/${a.id}/${id}">${esc(t)}</a>`).join('')}
      <div class="toc-h">${esc(TASKS[a.task] || '')}</div>${same.map(x => `<a class="toc-it" href="#/arch/${x.id}">${esc(x.name)}</a>`).join('')}`;
    const an = a.anatomy || {};
    mainEl.innerHTML = `<a class="crumb" href="#/arch">← Model Zoo</a>${taskTab ? ` · <a class="crumb" href="#/${taskTab.id}">${esc(taskTab.short || taskTab.title)} leaderboards</a>` : ''}
      <div class="archhead"><h1>${esc(a.name)}</h1><div class="archmeta"><span>${esc(a.org || '')}</span><span>Released ${fmtD(a.date)}</span>${a.paper ? `<a href="${esc(a.paper)}" target="_blank" rel="noopener noreferrer">Paper ↗</a>` : ''}${a.code ? `<a href="${esc(a.code)}" target="_blank" rel="noopener noreferrer">Code ↗</a>` : ''}</div></div>
      <p class="sum">${a.tagline || ''}</p>
      <section id="arch-anat" class="anat">${Object.keys(ROLE).filter(k => an[k]).map(k => `<div><b><span class="role ${ROLE[k][0]}">${ROLE[k][1].toUpperCase()}</span></b>${an[k]}</div>`).join('')}</section>
      <section id="arch-diagram" class="archsec"><h2>Block diagram</h2>
        ${a.pipeline ? `<div class="dgwrap">${CVDiagram.pipeline(a.pipeline)}</div>` : ''}
        ${a.blocks && a.blocks.length ? `<h2>Inside the blocks</h2><div class="dgrow">${a.blocks.map(c => `<div>${CVDiagram.chain(c)}</div>`).join('')}</div>` : ''}</section>
      ${(a.sections || []).map((s, i) => `<section id="arch-s${i}" class="archsec"><h2>${esc(s.h)}</h2><div class="body">${s.html}</div></section>`).join('')}
      ${a.f && a.f.length ? `<section id="arch-eq" class="archsec"><h2>Key equations</h2>${a.f.map(f => `<div class="f">$$${esc(f)}$$</div>`).join('')}</section>` : ''}
      ${rows.length ? `<section id="arch-bench" class="archsec"><h2>Benchmarks in this app</h2><div class="tblwrap"><table class="lt"><thead><tr><th>Model</th><th>Benchmark</th><th class="num">Score</th><th>Released</th><th>Details</th></tr></thead><tbody>
        ${rows.map(({ b, r }) => { const tb = CV.tabs.find(t => (t.boards || []).includes(b.id)); return `<tr><td><b>${esc(r.n)}</b></td><td><a href="#/${tb.id}/${tb.id}--lb-${b.id}/${encodeURIComponent(r.n)}">${esc(b.title)}</a></td><td class="num">${esc(b.metric)} ${fmtCell('ap50', r.m)}</td><td class="nowrap">${fmtD(r.d)}</td><td class="nt">${(b.cols || []).filter(c => r[c.k] != null).map(c => `${esc(c.label)} ${fmtCell(c.k, r[c.k])}`).join(' · ')}</td></tr>`; }).join('')}
        </tbody></table></div></section>` : ''}
      ${same.length ? `<h2>Other ${esc((TASKS[a.task] || '').toLowerCase())} architectures</h2><div class="chips">${same.map(x => `<a class="chip" href="#/arch/${x.id}">${esc(x.name)}</a>`).join('')}</div>` : ''}`;
    typeset(mainEl);
  }

  function route() {
    const parts = location.hash.replace(/^#\/?/, '').split('/');
    const tab = tabById[parts[0]] || builtinHome;
    if (tab.id === 'arch') {
      const a = archById[parts[1]];
      if (!a) { if (current !== builtinZoo || mainEl.dataset.arch) { current = null; renderTab(builtinZoo); } mainEl.dataset.arch = ''; window.scrollTo(0, 0); return; }
      if (mainEl.dataset.arch !== a.id) { renderArch(a); mainEl.dataset.arch = a.id; window.scrollTo(0, 0); }
      const t = parts[2] && document.getElementById('arch-' + parts[2]);
      if (t) requestAnimationFrame(() => t.scrollIntoView({ behavior: 'smooth', block: 'start' }));
      return;
    }
    mainEl.dataset.arch = '';
    if (tab !== current || !parts[1]) { renderTab(tab); window.scrollTo(0, 0); }
    const target = parts[1] && document.getElementById(parts[1]);
    if (target) {
      const tf = $('.tabfilter', tocEl);
      if (tf && tf.value) { tf.value = ''; filterTab(tab, ''); }
      const row = parts[2] && target.querySelector(`tr[data-row="${CSS.escape(decodeURIComponent(parts[2]))}"]`);
      const el = row || target;
      requestAnimationFrame(() => {
        el.scrollIntoView({ behavior: 'smooth', block: row ? 'center' : 'start' });
        el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
      });
      $$('.toc a', document).forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#/${tab.id}/${parts[1]}`));
    }
  }
  window.addEventListener('hashchange', route);
  route();
})();

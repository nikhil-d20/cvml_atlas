// Content registry. Data files call CV.tab({...}) and CV.board({...}); weekly updates call CV.update({...}).
// Item schema: { id, t: title, tags: 'comma,separated', s: one-line summary,
//   f: [latex, ...] display formulas, h: html (inline math with \( \) or $$ $$),
//   tip: practical advice html, code: python snippet, added: 'YYYY-MM-DD' (shows NEW for 30 days) }
// Board row schema: { n, d: 'YYYY-MM' release date, fam, m: metric, ...cols, notes, added: 'YYYY-MM-DD' }
window.CV = {
  tabs: [],
  boards: {},
  changelog: [],
  contentUpdated: null,
  newsSnapshot: null,
  tab(def) { this.tabs.push(def); return def; },
  // Architecture detail pages (data/arch_*.js). `match` maps leaderboard row names to the page.
  archs: [],
  arch(def) { this.archs.push(def); return def; },
  archFor(name) { return this.archs.find(a => a.match && a.match.test(name)) || null; },
  board(def) {
    def.rows.forEach(r => { r.y = r.y || +String(r.d || '').slice(0, 4) || null; });
    this.boards[def.id] = def; return def;
  },
  // Weekly update packet (see data/updates.js). Idempotent: existing ids / model names are patched, not duplicated.
  update(u) {
    const date = u.date;
    if (!this.contentUpdated || date > this.contentUpdated) this.contentUpdated = date;
    (u.changelog || []).forEach(c => this.changelog.push(Object.assign({ date }, typeof c === 'string' ? { text: c } : c)));
    (u.items || []).forEach(({ tab, section, sectionTitle, item }) => {
      const t = this.tabs.find(x => x.id === tab);
      if (!t) return console.warn('update: unknown tab', tab);
      let sec = t.sections.find(s => s.id === section);
      if (!sec) { sec = { id: section, title: sectionTitle || section, items: [] }; t.sections.push(sec); }
      const it = Object.assign({ added: date }, item);
      const i = sec.items.findIndex(x => x.id === it.id);
      if (i >= 0) sec.items[i] = Object.assign(sec.items[i], it, { updated: date }); else sec.items.unshift(it);
    });
    (u.models || []).forEach(({ board, row }) => {
      const b = this.boards[board];
      if (!b) return console.warn('update: unknown board', board);
      const r = Object.assign({ added: date }, row);
      r.y = +String(r.d || '').slice(0, 4) || null;
      const i = b.rows.findIndex(x => x.n === r.n);
      if (i >= 0) b.rows[i] = Object.assign(b.rows[i], r); else b.rows.push(r);
    });
  },
  // small table helper: CV.tbl(['a','b'], [[1,2],[3,4]])
  tbl(head, rows, cls) {
    const th = head.map(h => `<th>${h}</th>`).join('');
    const tr = rows.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('');
    return `<div class="tblwrap"><table class="${cls || 'lt'}"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
  }
};

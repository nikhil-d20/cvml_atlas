/* Block-diagram renderer (SVG, theme-aware via CSS classes).
   CVDiagram.pipeline(stages)  — left→right model anatomy: input → backbone → neck → head → output
     stage: { k: 'input'|'backbone'|'neck'|'encoder'|'decoder'|'head'|'output'|'other', t: title, d?: subtitle,
              b?: [block strings], o?: 'outputs passed on' }
   CVDiagram.chain(spec)       — top→bottom block internals with skip / residual connections
     spec: { t: title, k?: color key, ops: [strings], skips?: [[fromIdx, toIdx, label]], note? } */
(function () {
  'use strict';
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const CW = 6.3; // approx px per char at 11px
  function wrap(text, width) {
    const max = Math.max(8, Math.floor(width / CW)), out = [];
    String(text).split('\n').forEach(par => {
      let line = '';
      par.split(' ').forEach(w => {
        if ((line + ' ' + w).trim().length > max && line) { out.push(line); line = w; } else line = (line + ' ' + w).trim();
        while (line.length > max) { out.push(line.slice(0, max)); line = line.slice(max); }
      });
      out.push(line);
    });
    return out;
  }
  const LABEL = { input: 'INPUT', backbone: 'BACKBONE', neck: 'NECK', encoder: 'ENCODER / NECK', decoder: 'DECODER / HEAD', head: 'HEAD', output: 'OUTPUT', other: 'MODULE', loss: 'LOSS / TRAINING' };

  function pipeline(stages) {
    const GAP = 34, PAD = 8, LH = 14;
    const cols = stages.map(s => {
      const w = (s.k === 'input' || s.k === 'output') ? 124 : 186;
      const inner = w - 2 * PAD - 8;
      const title = wrap(s.t, inner + 8), sub = s.d ? wrap(s.d, inner + 8) : [];
      const blocks = (s.b || []).map(b => wrap(b, inner - 4));
      const outs = s.o ? wrap(s.o, inner + 8) : [];
      let h = 22 + title.length * 15 + sub.length * LH + 6;
      blocks.forEach(l => { h += l.length * LH + 10 + 5; });
      if (outs.length) h += outs.length * LH + 10;
      return { s, w, title, sub, blocks, outs, h: h + 6 };
    });
    const H = Math.max(...cols.map(c => c.h)) + 2;
    const W = cols.reduce((a, c) => a + c.w, 0) + GAP * (cols.length - 1) + 4;
    let x = 2, svg = '';
    const arrowY = 34;
    cols.forEach((c, i) => {
      const k = c.s.k || 'other';
      svg += `<g class="dg-${k}"><rect class="dg-box" x="${x}" y="1" width="${c.w}" height="${c.h}" rx="10"/>`;
      svg += `<text class="dg-lab" x="${x + PAD}" y="16">${LABEL[k] || k.toUpperCase()}</text>`;
      let y = 32;
      c.title.forEach(l => { svg += `<text class="dg-t" x="${x + PAD}" y="${y}">${esc(l)}</text>`; y += 15; });
      c.sub.forEach(l => { svg += `<text class="dg-d" x="${x + PAD}" y="${y}">${esc(l)}</text>`; y += LH; });
      y += 4;
      c.blocks.forEach(lines => {
        const bh = lines.length * LH + 8;
        svg += `<rect class="dg-blk" x="${x + PAD}" y="${y}" width="${c.w - 2 * PAD}" height="${bh}" rx="5"/>`;
        lines.forEach((l, j) => { svg += `<text class="dg-b" x="${x + PAD + 6}" y="${y + 14 + j * LH}">${esc(l)}</text>`; });
        y += bh + 5;
      });
      if (c.outs.length) {
        y += 2;
        c.outs.forEach(l => { svg += `<text class="dg-o" x="${x + PAD}" y="${y + 10}">${esc(l)}</text>`; y += LH; });
      }
      svg += '</g>';
      if (i < cols.length - 1) {
        const x1 = x + c.w + 3, x2 = x + c.w + GAP - 4;
        svg += `<line class="dg-arrow" x1="${x1}" y1="${arrowY}" x2="${x2}" y2="${arrowY}" marker-end="url(#dgah)"/>`;
      }
      x += c.w + GAP;
    });
    return frame(W, H + 2, svg, 'Model anatomy: ' + stages.map(s => s.t).join(' → '));
  }

  function chain(spec) {
    const BW = 212, BH0 = 26, GAP = 16, LH = 13, TOP = 26, SKIPW = 60;
    const ops = spec.ops.map(o => wrap(o, BW - 16));
    let y = TOP; const pos = [];
    ops.forEach(l => { const h = Math.max(BH0, l.length * LH + 12); pos.push({ y, h }); y += h + GAP; });
    const H = y - GAP + 8, nSk = (spec.skips || []).length;
    const W = BW + 12 + (nSk ? SKIPW + nSk * 14 : 10);
    const k = spec.k || 'other';
    let svg = `<text class="dg-t" x="4" y="15">${esc(spec.t)}</text>`;
    ops.forEach((lines, i) => {
      const p = pos[i];
      svg += `<g class="dg-${k}"><rect class="dg-blk dg-op" x="4" y="${p.y}" width="${BW}" height="${p.h}" rx="6"/></g>`;
      lines.forEach((l, j) => { svg += `<text class="dg-b" x="${4 + BW / 2}" y="${p.y + p.h / 2 + 4 + (j - (lines.length - 1) / 2) * LH}" text-anchor="middle">${esc(l)}</text>`; });
      if (i < ops.length - 1) svg += `<line class="dg-arrow" x1="${4 + BW / 2}" y1="${p.y + p.h + 1}" x2="${4 + BW / 2}" y2="${pos[i + 1].y - 3}" marker-end="url(#dgah)"/>`;
    });
    (spec.skips || []).forEach(([a, b, lab], i) => {
      const xa = 4 + BW, xo = xa + 18 + i * 14 + (lab ? 10 : 0);
      const ya = pos[a].y + pos[a].h / 2, yb = pos[b].y + pos[b].h / 2;
      svg += `<path class="dg-skip" d="M${xa} ${ya} H${xo} V${yb} H${xa + 4}" marker-end="url(#dgah)"/>`;
      if (lab) svg += `<text class="dg-o" x="${xo + 4}" y="${(ya + yb) / 2 + 4}">${esc(lab)}</text>`;
    });
    if (spec.note) svg += '';
    return frame(W + 40, H, svg, spec.t) + (spec.note ? `<p class="dg-note">${esc(spec.note)}</p>` : '');
  }

  function frame(W, H, body, label) {
    return `<svg class="dg" viewBox="0 0 ${Math.ceil(W)} ${Math.ceil(H)}" width="${Math.ceil(W)}" height="${Math.ceil(H)}" role="img" aria-label="${esc(label)}">
      <defs><marker id="dgah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="dg-ah"/></marker></defs>${body}</svg>`;
  }

  window.CVDiagram = { pipeline, chain };
})();

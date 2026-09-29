/* In-app news aggregator for the Android build.
   The Android shell exposes AndroidBridge.fetch(id, url) (native HTTP, no CORS) and calls back CVNative.done(id, ok, body).
   Mirrors the feed list and categorisation in server.py. */
(function () {
  'use strict';
  const FEEDS = [
    ['arXiv cs.CV', 'https://rss.arxiv.org/rss/cs.CV', 'rss', ['Research', 'Vision'], 30],
    ['arXiv cs.LG', 'https://rss.arxiv.org/rss/cs.LG', 'rss', ['Research'], 15],
    ['HF Daily Papers', 'https://huggingface.co/api/daily_papers', 'hfpapers', ['Research'], 30],
    ['Hugging Face Blog', 'https://huggingface.co/blog/feed.xml', 'rss', ['Open source'], 15],
    ['Google Research', 'https://research.google/blog/rss/', 'rss', ['Research', 'Industry'], 10],
    ['Google DeepMind', 'https://deepmind.google/blog/rss.xml', 'rss', ['Research', 'Industry'], 10],
    ['OpenAI', 'https://openai.com/news/rss.xml', 'rss', ['Industry'], 10],
    ['NVIDIA Developer', 'https://developer.nvidia.com/blog/feed', 'rss', ['Industry'], 10],
    ['PyTorch Blog', 'https://pytorch.org/blog/feed.xml', 'rss', ['Open source'], 8],
    ['MIT Tech Review AI', 'https://www.technologyreview.com/topic/artificial-intelligence/feed', 'rss', ['Industry'], 10],
    ['The Verge AI', 'https://www.theverge.com/rss/ai-artificial-intelligence/index.xml', 'rss', ['Industry'], 10],
    ['Hacker News (AI)', 'https://hn.algolia.com/api/v1/search_by_date?tags=story&query=AI%20model&numericFilters=points%3E80', 'hn', ['Industry'], 15]
  ];
  const VISION_RE = /\b(vision|image|video|visual|segment\w*|detect\w*|pose|keypoint|diffusion|3d|depth|pixel|camera|multimodal|vlm|yolo|detr|sam\b|dino|gaussian splat\w*|nerf|ocr|tracking|generat\w* model|text-to-image|text-to-video)/i;
  const OSS_RE = /\b(open[- ]source|open weights|github|hugging ?face|apache|released the code|weights)\b/i;

  const pending = {};
  let seq = 0;
  window.CVNative = {
    done(id, ok, body) { const p = pending[id]; if (!p) return; delete pending[id]; ok ? p.resolve(body) : p.reject(new Error(body)); }
  };
  function nativeGet(url, timeoutMs) {
    return new Promise((resolve, reject) => {
      const id = 'r' + (++seq);
      pending[id] = { resolve, reject };
      setTimeout(() => { if (pending[id]) { delete pending[id]; reject(new Error('timeout')); } }, timeoutMs || 25000);
      window.AndroidBridge.fetch(id, url);
    });
  }

  const clean = (t, n = 280) => {
    const d = document.createElement('div'); d.innerHTML = String(t || '').replace(/<[^>]+>/g, ' ');
    let s = (d.textContent || '').replace(/\s+/g, ' ').trim().replace(/^arXiv:\S+\s+Announce Type:\s*\w+\s*Abstract:\s*/, '');
    return s.length > n ? s.slice(0, n) + '…' : s;
  };
  const toTs = s => { const t = Date.parse(s || ''); return isNaN(t) ? null : Math.floor(t / 1000); };
  const child = (el, names) => { for (const c of el.children) if (names.includes(c.localName)) return c; return null; };

  function parseRss(txt, limit) {
    const doc = new DOMParser().parseFromString(txt, 'text/xml');
    const els = [...doc.getElementsByTagName('*')].filter(e => e.localName === 'item' || e.localName === 'entry').slice(0, limit);
    return els.map(it => {
      const l = child(it, ['link']), d = child(it, ['pubDate', 'published', 'updated', 'date']), s = child(it, ['description', 'summary', 'content']);
      return { title: clean(child(it, ['title'])?.textContent, 200), link: ((l && (l.textContent || l.getAttribute('href'))) || '').trim(), date: toTs(d?.textContent), summary: clean(s?.textContent) };
    });
  }
  const parseHf = (txt, limit) => JSON.parse(txt).slice(0, limit).map(p => {
    const paper = p.paper || {};
    return { title: clean(paper.title || p.title, 200), link: 'https://huggingface.co/papers/' + (paper.id || ''), date: toTs(p.publishedAt || paper.publishedAt), summary: clean(paper.summary) };
  });
  const parseHn = (txt, limit) => (JSON.parse(txt).hits || []).slice(0, limit).map(h => ({
    title: clean(h.title, 200), link: h.url || `https://news.ycombinator.com/item?id=${h.objectID}`, date: h.created_at_i,
    summary: `${h.points || 0} points · ${h.num_comments || 0} comments on Hacker News`
  }));
  const PARSERS = { rss: parseRss, hfpapers: parseHf, hn: parseHn };

  async function loadFeed([name, url, kind, cats, limit]) {
    const items = PARSERS[kind](await nativeGet(url), limit);
    items.forEach(i => {
      const c = cats.slice(), text = i.title + ' ' + i.summary;
      if (!c.includes('Vision') && VISION_RE.test(text)) c.push('Vision');
      if (!c.includes('Open source') && OSS_RE.test(text)) c.push('Open source');
      Object.assign(i, { source: name, cats: c });
    });
    return items;
  }

  window.CVNews = {
    available: () => !!window.AndroidBridge,
    async fetchAll() {
      const items = [], errors = [];
      const res = await Promise.allSettled(FEEDS.map(loadFeed));
      res.forEach((r, i) => r.status === 'fulfilled' ? items.push(...r.value) : errors.push({ source: FEEDS[i][0], error: String(r.reason && r.reason.message).slice(0, 200) }));
      const seen = new Set(), uniq = [];
      items.forEach(i => { const k = i.title.toLowerCase().replace(/\W+/g, '').slice(0, 80); if (i.title && !seen.has(k)) { seen.add(k); uniq.push(i); } });
      uniq.sort((a, b) => (b.date || 0) - (a.date || 0));
      return { updated: Math.floor(Date.now() / 1000), items: uniq, errors };
    }
  };
})();

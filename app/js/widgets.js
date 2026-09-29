/* Interactive widgets mounted inside topic cards: <div class="widget" data-widget="name" data-args='{…}'>.
   - lrcurve   : static plot of one LR schedule (data-args: {"s": "cosine", …params})
   - lrplot    : interactive LR schedule explorer with PyTorch code
   - prcalc    : precision / recall / F-beta / MCC calculator from TP, FP, FN, TN
   - fpfn      : FP / FN strategy advisor (task + problem + counts → ordered playbook) */
(function () {
  'use strict';
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const store = { get(k, d) { try { const v = localStorage.getItem('cvml.w.' + k); return v ? JSON.parse(v) : d; } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem('cvml.w.' + k, JSON.stringify(v)); } catch (e) { } } };

  /* ---------------- LR schedules (normalized: peak LR = 1, T steps) ---------------- */
  const SCHED = {
    constant: { label: 'Constant', f: () => 1 },
    warmup_constant: { label: 'Linear warmup → constant', f: (t, T, p) => t < p.warm ? (t + 1) / p.warm : 1 },
    step: { label: 'StepLR', f: (t, T, p) => Math.pow(p.gamma, Math.floor(t / p.stepSize)) },
    multistep: { label: 'MultiStepLR', f: (t, T, p) => Math.pow(p.gamma, p.milestones.filter(m => t >= m * T).length) },
    exponential: { label: 'ExponentialLR', f: (t, T, p) => Math.pow(p.expGamma, t) },
    linear: { label: 'Linear decay', f: (t, T, p) => warm(t, p, () => 1 - (t - p.warm) / Math.max(1, T - p.warm) * (1 - p.min)) },
    poly: { label: 'Polynomial (power 0.9)', f: (t, T, p) => warm(t, p, () => p.min + (1 - p.min) * Math.pow(1 - (t - p.warm) / Math.max(1, T - p.warm), p.power)) },
    cosine: { label: 'Cosine annealing', f: (t, T, p) => warm(t, p, () => p.min + 0.5 * (1 - p.min) * (1 + Math.cos(Math.PI * (t - p.warm) / Math.max(1, T - p.warm)))) },
    sgdr: { label: 'Cosine warm restarts (SGDR)', f: (t, T, p) => { let T0 = p.T0 * T, tc = t; while (tc >= T0) { tc -= T0; T0 *= p.Tmult; } return p.min + 0.5 * (1 - p.min) * (1 + Math.cos(Math.PI * tc / T0)); } },
    onecycle: { label: 'OneCycleLR', f: (t, T, p) => { const up = p.pct * T, lo = 1 / p.div, fin = lo / p.finalDiv; const c = (a, b, x) => b + (a - b) * 0.5 * (1 + Math.cos(Math.PI * x)); return t < up ? c(lo, 1, t / up) : c(1, fin, (t - up) / (T - up)); } },
    wsd: { label: 'Warmup-Stable-Decay (WSD)', f: (t, T, p) => { const d0 = T * (1 - p.decayFrac); return t < p.warm ? (t + 1) / p.warm : t < d0 ? 1 : p.min + (1 - p.min) * (1 - (t - d0) / (T - d0)); } },
    invsqrt: { label: 'Inverse square root', f: (t, T, p) => t < p.warm ? (t + 1) / p.warm : Math.sqrt(p.warm / (t + 1)) },
    cyclic: { label: 'CyclicLR (triangular)', f: (t, T, p) => { const cyc = 2 * p.half * T, x = Math.abs(((t % cyc) / (p.half * T)) - 1); return p.base + (1 - p.base) * (1 - x); } },
    plateau: { label: 'ReduceLROnPlateau (example)', f: (t, T) => Math.pow(0.5, [0.35, 0.6, 0.82].filter(m => t >= m * T).length) }
  };
  function warm(t, p, f) { return t < p.warm ? (t + 1) / p.warm : f(); }
  const DEF = { warm: 0, min: 0, gamma: 0.1, stepSize: 30, milestones: [0.5, 0.75], expGamma: 0.97, power: 0.9, T0: 0.2, Tmult: 2, pct: 0.3, div: 25, finalDiv: 1e4, decayFrac: 0.2, base: 0.1, half: 0.1 };

  function plot(name, p, opts) {
    opts = opts || {};
    const T = p.T || 100, W = opts.w || 300, H = opts.h || 120, P = { l: 30, r: 8, t: 8, b: 20 };
    const f = SCHED[name].f, pts = [];
    for (let t = 0; t <= T; t++) pts.push([t, Math.max(0, Math.min(1.05, f(t, T, p)))]);
    const sx = t => P.l + t / T * (W - P.l - P.r), sy = v => H - P.b - v * (H - P.t - P.b);
    const d = pts.map((q, i) => (i ? 'L' : 'M') + sx(q[0]).toFixed(1) + ' ' + sy(q[1]).toFixed(1)).join(' ');
    const area = d + ` L${sx(T)} ${sy(0)} L${sx(0)} ${sy(0)} Z`;
    return `<svg class="lrsvg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(SCHED[name].label)} learning-rate curve">
      <line class="g" x1="${P.l}" x2="${W - P.r}" y1="${sy(0)}" y2="${sy(0)}"/><line class="g" x1="${P.l}" x2="${W - P.r}" y1="${sy(1)}" y2="${sy(1)}"/><line class="g" x1="${P.l}" x2="${W - P.r}" y1="${sy(0.5)}" y2="${sy(0.5)}" stroke-dasharray="2 3"/>
      <text x="${P.l - 4}" y="${sy(1) + 3}" text-anchor="end">η<tspan font-size="7">max</tspan></text><text x="${P.l - 4}" y="${sy(0) + 3}" text-anchor="end">0</text>
      <text x="${P.l}" y="${H - 5}">0</text><text x="${W - P.r}" y="${H - 5}" text-anchor="end">${opts.xlabel || 'training steps →'}</text>
      <path class="a" d="${area}"/><path class="c" d="${d}"/></svg>`;
  }

  function lrcurve(el, args) {
    const p = Object.assign({}, DEF, args);
    el.innerHTML = plot(args.s, p);
  }

  const CODE = {
    constant: 'sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=0)',
    warmup_constant: 'sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters={W})',
    step: 'sched = torch.optim.lr_scheduler.StepLR(opt, step_size={S}, gamma={G})',
    multistep: 'sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[{M1}, {M2}], gamma={G})',
    exponential: 'sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma={EG})',
    linear: 'warm = LinearLR(opt, start_factor=1e-3, total_iters={W})\nmain = LinearLR(opt, start_factor=1.0, end_factor={MIN}, total_iters={T}-{W})\nsched = SequentialLR(opt, [warm, main], milestones=[{W}])',
    poly: 'warm = LinearLR(opt, start_factor=1e-3, total_iters={W})\nmain = PolynomialLR(opt, total_iters={T}-{W}, power={PW})\nsched = SequentialLR(opt, [warm, main], milestones=[{W}])',
    cosine: 'warm = LinearLR(opt, start_factor=1e-3, total_iters={W})\nmain = CosineAnnealingLR(opt, T_max={T}-{W}, eta_min=base_lr*{MIN})\nsched = SequentialLR(opt, [warm, main], milestones=[{W}])',
    sgdr: 'sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0={T0}, T_mult={TM}, eta_min=base_lr*{MIN})',
    onecycle: 'sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=base_lr, total_steps={T}, pct_start={PCT}, div_factor=25, final_div_factor=1e4)',
    wsd: '# warmup → constant → linear cooldown over the last {DF}% of steps\nlam = lambda s: (s+1)/{W} if s < {W} else 1.0 if s < {D0} else max({MIN}, 1-(s-{D0})/({T}-{D0}))\nsched = torch.optim.lr_scheduler.LambdaLR(opt, lam)',
    invsqrt: 'lam = lambda s: (s+1)/{W} if s < {W} else ({W}/(s+1))**0.5\nsched = torch.optim.lr_scheduler.LambdaLR(opt, lam)',
    cyclic: 'sched = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=base_lr*{B}, max_lr=base_lr, step_size_up={HS}, mode="triangular")',
    plateau: 'sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)\n# each epoch: sched.step(val_map)'
  };
  const WHEN = {
    constant: 'Debugging, short fine-tunes of adapters/LoRA, diffusion training (with EMA).',
    warmup_constant: 'Diffusion / DiT training, continual training where the budget is unknown.',
    step: 'Legacy CNN recipes (ResNet 30/60/90); simple and reproducible, but abrupt drops.',
    multistep: 'Detectron-style detection (×0.1 at ~67% and ~89% of iterations), DETR (drop at epoch 200/300).',
    exponential: 'Old EfficientNet/MobileNet TF recipes; smooth decay when you want a fixed per-epoch factor.',
    linear: 'Transformer fine-tuning (BERT/ViT), YOLO (lrf = 0.01), short schedules.',
    poly: 'Semantic segmentation (DeepLab, mmseg: power 0.9), iteration-based training.',
    cosine: 'Default for almost everything with a known budget: classification, ViTs, DETR variants, fine-tuning.',
    sgdr: 'Escaping sharp minima, snapshot ensembles, long training with periodic checkpoints.',
    onecycle: 'Short budgets / fast convergence ("super-convergence"), small datasets, Kaggle-style runs.',
    wsd: 'Large or open-ended pre-training; lets you stop, branch or extend training at any point.',
    invsqrt: 'Original Transformer / NLP recipes; rarely best for vision today.',
    cyclic: 'LR range exploration, some GAN / small-data training; mostly superseded by OneCycle.',
    plateau: 'Unknown budget, noisy small datasets, medical imaging; not reproducible across runs.'
  };

  function lrplot(el) {
    const st = Object.assign({ s: 'cosine', T: 100, warmPct: 5, minPct: 1, gamma: 0.1 }, store.get('lrplot', {}));
    el.innerHTML = `<div class="wform">
      <label>Scheduler <select id="lrS">${Object.entries(SCHED).map(([k, v]) => `<option value="${k}" ${k === st.s ? 'selected' : ''}>${esc(v.label)}</option>`).join('')}</select></label>
      <label>Total steps/epochs <input id="lrT" type="number" min="10" max="100000" value="${st.T}"></label>
      <label>Warmup % <input id="lrW" type="number" min="0" max="50" value="${st.warmPct}"></label>
      <label>Min LR % of max <input id="lrM" type="number" min="0" max="50" step="0.1" value="${st.minPct}"></label>
      <label>γ (step decay) <input id="lrG" type="number" min="0.01" max="0.9" step="0.01" value="${st.gamma}"></label></div>
      <div class="wout"></div>`;
    const draw = () => {
      const s = $('#lrS').value, T = Math.max(10, +$('#lrT').value || 100), W = Math.round(T * (+$('#lrW').value || 0) / 100), MIN = (+$('#lrM').value || 0) / 100, G = +$('#lrG').value || 0.1;
      store.set('lrplot', { s, T, warmPct: +$('#lrW').value, minPct: +$('#lrM').value, gamma: G });
      const p = Object.assign({}, DEF, { T, warm: W, min: MIN, gamma: G, stepSize: Math.max(1, Math.round(T * 0.3)), expGamma: Math.pow(0.01, 1 / T) });
      const d0 = Math.round(T * (1 - DEF.decayFrac));
      const code = CODE[s].replace(/\{T\}/g, T).replace(/\{W\}/g, Math.max(1, W)).replace(/\{MIN\}/g, MIN).replace(/\{G\}/g, G).replace(/\{S\}/g, p.stepSize)
        .replace(/\{M1\}/g, Math.round(T * 0.5)).replace(/\{M2\}/g, Math.round(T * 0.75)).replace(/\{EG\}/g, p.expGamma.toFixed(4)).replace(/\{PW\}/g, 0.9)
        .replace(/\{T0\}/g, Math.round(T * 0.2)).replace(/\{TM\}/g, 2).replace(/\{PCT\}/g, 0.3).replace(/\{DF\}/g, 20).replace(/\{D0\}/g, d0).replace(/\{B\}/g, 0.1).replace(/\{HS\}/g, Math.round(T * 0.1));
      $('.wout', el).innerHTML = plot(s, p, { w: 560, h: 190, xlabel: `step (0 → ${T})` }) +
        `<p class="note"><b>When to use:</b> ${esc(WHEN[s])}</p><pre><code>from torch.optim.lr_scheduler import *\n${esc(code)}\n# call sched.step() once per ${s === 'plateau' ? 'epoch with the val metric' : 'step (or epoch — be consistent with T)'}</code></pre>`;
    };
    const $ = s => el.querySelector(s);
    el.querySelectorAll('select,input').forEach(i => i.addEventListener('input', draw));
    draw();
  }

  /* ---------------- precision / recall calculator ---------------- */
  function prcalc(el) {
    const st = Object.assign({ tp: 820, fp: 140, fn: 180, tn: 8860 }, store.get('prcalc', {}));
    el.innerHTML = `<div class="wform">${['tp', 'fp', 'fn', 'tn'].map(k => `<label>${k.toUpperCase()} <input data-k="${k}" type="number" min="0" value="${st[k]}"></label>`).join('')}</div><div class="wout"></div>
      <p class="note">Detection/segmentation usually have no meaningful TN — leave it 0; accuracy and specificity are then hidden.</p>`;
    const draw = () => {
      const v = {}; el.querySelectorAll('input').forEach(i => { v[i.dataset.k] = Math.max(0, +i.value || 0); });
      store.set('prcalc', v);
      const { tp, fp, fn, tn } = v, P = tp / (tp + fp || 1), R = tp / (tp + fn || 1);
      const Fb = b => (1 + b * b) * P * R / (b * b * P + R || 1);
      const rows = [['Precision', P, 'of predicted positives, how many are correct → high = few FPs'], ['Recall', R, 'of real positives, how many are found → high = few FNs'], ['F1', Fb(1), 'balance of P and R'], ['F0.5', Fb(0.5), 'weights precision higher (FPs are costly)'], ['F2', Fb(2), 'weights recall higher (FNs are costly)']];
      if (tn > 0) {
        const N = tp + fp + fn + tn, mccD = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) || 1;
        rows.push(['Accuracy', (tp + tn) / N, 'misleading when classes are imbalanced'], ['Specificity', tn / (tn + fp || 1), 'true-negative rate'], ['FPR', fp / (fp + tn || 1), 'false alarms per negative'], ['MCC', (tp * tn - fp * fn) / mccD, '−1…1, robust to imbalance']);
      }
      const verdict = fp > fn * 1.5 ? 'False positives dominate → work on precision (see the FP/FN advisor below).' : fn > fp * 1.5 ? 'False negatives dominate → work on recall (see the FP/FN advisor below).' : 'FPs and FNs are balanced → improve the model / data (both), not just the threshold.';
      el.querySelector('.wout').innerHTML = `<table class="lt"><tbody>${rows.map(([n, x, d]) => `<tr><td><b>${n}</b></td><td class="num">${(x * 100).toFixed(1)}%</td><td class="nt">${d}</td></tr>`).join('')}</tbody></table><p class="tip"><b>Reading</b> ${verdict}</p>`;
    };
    el.querySelectorAll('input').forEach(i => i.addEventListener('input', draw));
    draw();
  }

  /* ---------------- FP / FN advisor ---------------- */
  // Each lever: [problem tags, text]. Tags: fp, fn, both, loc (localization). Ordered by typical cost (cheapest first).
  const PLAY = {
    cls: { name: 'Classification (binary / multi-label)', steps: [
      ['Operating point', [['fp', 'Raise the decision threshold (e.g. 0.5 → 0.6–0.8) and pick it on a validation PR curve with an F0.5 or precision@recall target.'], ['fn', 'Lower the threshold (0.5 → 0.2–0.4); choose it for a recall target (e.g. recall ≥ 0.95) on validation.'], ['both', 'Tune per-class thresholds; calibrate with temperature scaling first so scores mean the same thing across classes.']]],
      ['Data', [['fp', 'Add hard negatives: look-alike images the model currently flags. Mine them from production logs.'], ['fn', 'Collect more positives of the missed sub-types (lighting, pose, rare variants); oversample rare positives.'], ['both', 'Audit label noise (confident-learning / cleanlab); inconsistent labels cap both P and R.']]],
      ['Loss & sampling', [['fp', 'Lower the positive class weight / pos_weight in BCE; use label smoothing to curb over-confident positives.'], ['fn', 'Increase pos_weight (≈ #neg/#pos, capped at 10–20) or use focal loss (γ = 2) / asymmetric loss for multi-label.'], ['both', 'Class-balanced sampling (sqrt frequency) plus logit adjustment for long-tailed classes.']]],
      ['Model & input', [['fn', 'Higher input resolution for small cues; stronger pre-trained backbone.'], ['both', 'Bigger / better pre-trained backbone, longer training, stronger augmentation (RandAugment, mixup/cutmix) if overfitting.']]],
      ['Post-processing', [['fp', 'Require k consecutive positive frames (video) or ensemble agreement.'], ['fn', 'TTA (flip / multi-crop) and take the max score.']]]] },
    det: { name: 'Object detection', steps: [
      ['Operating point', [['fp', 'Raise the confidence threshold (YOLO default 0.25 → 0.35–0.5); set it per class if classes differ.'], ['fn', 'Lower the confidence threshold (0.25 → 0.1–0.15) and raise max detections for crowded scenes (300 → 1000).'], ['both', 'Pick the per-class threshold at max F1 (or your cost-weighted F-beta) on validation; do not tune on test.']]],
      ['NMS / duplicates', [['fp', 'Duplicates count as FPs: lower NMS IoU (0.7 → 0.5–0.6) or use an NMS-free model (YOLO26, RT-DETR/D-FINE/RF-DETR).'], ['fn', 'Overlapping objects suppressed? Raise NMS IoU (→ 0.75), use Soft-NMS, or class-aware NMS.']]],
      ['Data', [['fp', 'Add background images with no objects (5–10% of the set) and hard negatives (look-alikes that fire today).'], ['fn', 'Label every instance (missing labels teach the model to ignore objects); add examples of missed sizes, occlusions and lighting.'], ['both', 'Fix box consistency (tight boxes, occlusion policy); relabel ambiguous classes or merge them.']]],
      ['Small / hard objects', [['fn', 'Increase imgsz (640 → 1024/1280) or tile with SAHI (slice 640, overlap 0.2); add a P2 (stride-4) level.'], ['fn', 'Reduce aggressive downscale augmentation; use copy-paste of small instances; small-target-aware assignment (STAL / NWD).'], ['loc', 'Poor boxes (good AP50, low AP75): raise box / DFL loss gain (box 7.5 → 10), train longer, higher resolution, distribution-based box heads (DFL, D-FINE).']]],
      ['Loss & assignment', [['fp', 'Lower cls-loss weight on noisy classes; focal / VFL keeps easy negatives from dominating but check score calibration.'], ['fn', 'More positives per object: raise TAL top-k (10 → 13), ATSS, or one-to-many auxiliary heads; focal loss α toward positives.']]],
      ['Model', [['both', 'Larger model size (n → s → m), better pre-training (Objects365 / DINOv2 backbones), EMA weights, longer schedule.']]],
      ['Post-processing', [['fp', 'Geometry/size filters (min box area, aspect ratio, ROI masks); temporal tracking (require N frames, e.g. ByteTrack).'], ['fn', 'Tracking fills gaps between frames; TTA with WBF for offline use.']]]] },
    seg: { name: 'Segmentation (semantic / instance)', steps: [
      ['Operating point', [['fp', 'Raise the mask / pixel probability threshold (0.5 → 0.6–0.7) for binary segmentation.'], ['fn', 'Lower the pixel threshold (0.5 → 0.3–0.4); for instance seg lower the score threshold.']]],
      ['Loss', [['fp', 'Tversky with α > β (e.g. α = 0.7, β = 0.3) penalizes FPs more; add boundary loss if FPs are halos around objects.'], ['fn', 'Tversky with β > α (β = 0.7) or Focal Tversky (γ = 0.75) for small structures; Dice + CE instead of CE alone; class weights for rare classes.'], ['both', 'Dice + CE combo, OHEM for hard pixels, Lovász fine-tuning for IoU.']]],
      ['Data & sampling', [['fp', 'Add negative crops (images without the class) and look-alike textures.'], ['fn', 'Rare-class-aware crop sampling (mmseg cat_max_ratio 0.75); more annotated thin / small structures.'], ['both', 'Check mask quality at boundaries; SAM-assisted relabeling of sloppy polygons.']]],
      ['Resolution & model', [['fn', 'Higher input or output stride 8; sliding-window inference with overlap for large images.'], ['both', 'Stronger backbone (Next-ViT / ConvNeXt / Swin) + UperNet or Mask2Former.']]],
      ['Post-processing', [['fp', 'Remove connected components below a minimum area; morphological opening; CRF / boundary refinement.'], ['fn', 'Morphological closing / hole filling; flip + multi-scale TTA.']]]] },
    pose: { name: 'Keypoints / pose', steps: [
      ['Operating point', [['fp', 'Raise the person detection threshold and the keypoint confidence threshold (e.g. 0.3 → 0.5) before drawing / using joints.'], ['fn', 'Lower the person detector threshold (top-down accuracy is capped by missed people); lower keypoint confidence threshold.']]],
      ['Data', [['fp', 'Add images with people-like objects (mannequins, posters) as negatives.'], ['fn', 'More occluded / truncated / unusual poses; half-body augmentation; correct flip_idx.']]],
      ['Model & decoding', [['fn', 'Higher input (256×192 → 384×288), HRNet/ViTPose backbones, flip test.'], ['loc', 'Imprecise joints: DARK / UDP decoding, SimCC split ratio 2–3, RLE loss, larger heatmaps.']]],
      ['Post-processing', [['fp', 'Temporal smoothing (One-Euro filter) and bone-length sanity checks.'], ['fn', 'Tracking to carry keypoints across short occlusions.']]]] },
    anomaly: { name: 'Anomaly / defect detection', steps: [
      ['Operating point', [['fp', 'Raise the anomaly score threshold; set it from the distribution of scores on held-out normal images (e.g. 99.5th percentile).'], ['fn', 'Lower the threshold to hit a recall target on known defects; accept more manual review.']]],
      ['Data', [['fp', 'Add more normal variation (lighting, part tolerances) to the memory bank / training set — FPs are usually unseen normal variation.'], ['fn', 'Collect real defect examples and switch from one-class to supervised or semi-supervised (defects as positives).']]],
      ['Model', [['both', 'Higher resolution / tiling for tiny defects; better feature extractor (WideResNet → DINOv2); PatchCore coreset ratio ↑.']]]] }
  };

  function fpfn(el) {
    const st = Object.assign({ task: 'det', prob: 'fp', tp: '', fp: '', fn: '' }, store.get('fpfn', {}));
    el.innerHTML = `<div class="wform">
      <label>Task <select id="ffT">${Object.entries(PLAY).map(([k, v]) => `<option value="${k}" ${k === st.task ? 'selected' : ''}>${esc(v.name)}</option>`).join('')}</select></label>
      <label>Main problem <select id="ffP">
        <option value="fp" ${st.prob === 'fp' ? 'selected' : ''}>Too many false positives (false alarms)</option>
        <option value="fn" ${st.prob === 'fn' ? 'selected' : ''}>Too many false negatives (misses)</option>
        <option value="both" ${st.prob === 'both' ? 'selected' : ''}>Both / overall accuracy too low</option>
        <option value="loc" ${st.prob === 'loc' ? 'selected' : ''}>Found but imprecise (boxes / keypoints / edges)</option></select></label>
      <label>TP (optional) <input id="ffTP" type="number" min="0" value="${st.tp}"></label>
      <label>FP <input id="ffFP" type="number" min="0" value="${st.fp}"></label>
      <label>FN <input id="ffFN" type="number" min="0" value="${st.fn}"></label></div><div class="wout"></div>`;
    const $ = s => el.querySelector(s);
    const draw = () => {
      const task = $('#ffT').value; let prob = $('#ffP').value;
      const tp = +$('#ffTP').value || 0, fp = +$('#ffFP').value || 0, fn = +$('#ffFN').value || 0;
      store.set('fpfn', { task, prob, tp: $('#ffTP').value, fp: $('#ffFP').value, fn: $('#ffFN').value });
      let head = '';
      if (tp + fp + fn > 0) {
        const P = tp / (tp + fp || 1), R = tp / (tp + fn || 1), F1 = 2 * P * R / (P + R || 1);
        const auto = fp > 1.5 * fn ? 'fp' : fn > 1.5 * fp ? 'fn' : 'both';
        head = `<p class="note">Precision <b>${(P * 100).toFixed(1)}%</b> · Recall <b>${(R * 100).toFixed(1)}%</b> · F1 <b>${(F1 * 100).toFixed(1)}%</b> — your counts suggest <b>${{ fp: 'an FP problem', fn: 'an FN problem', both: 'a balanced error mix' }[auto]}</b>${auto !== prob && prob !== 'loc' ? ` (you selected “${prob.toUpperCase()}”)` : ''}.</p>`;
      }
      const want = prob === 'both' ? ['both', 'fp', 'fn'] : [prob, 'both'];
      const steps = PLAY[task].steps.map(([name, levers]) => [name, levers.filter(([tag]) => want.includes(tag))]).filter(([, l]) => l.length);
      const warn = prob === 'fp' ? 'Lowering FPs usually costs recall. Watch FN count on the same validation set after every change.' : prob === 'fn' ? 'Lowering FNs usually costs precision. Watch FP count on the same validation set after every change.' : prob === 'both' ? 'When both are high, thresholds only trade one for the other — the fix is data quality, capacity or resolution.' : 'Localization errors show up as good AP50 but low AP75 / high MPJPE — measure those, not just mAP50.';
      $('.wout').innerHTML = head + `<ol class="play">${steps.map(([name, l]) => `<li><b>${esc(name)}</b><ul>${l.map(([tag, t]) => `<li><span class="ptag p-${tag}">${tag.toUpperCase()}</span> ${esc(t)}</li>`).join('')}</ul></li>`).join('')}</ol>
        <p class="tip"><b>Rule</b> ${esc(warn)} Change one lever at a time, keep a fixed validation set, and compare PR curves, not single numbers.</p>`;
    };
    el.querySelectorAll('select,input').forEach(i => i.addEventListener('input', draw));
    draw();
  }

  const W = { lrcurve, lrplot, prcalc, fpfn };
  window.CVWidgets = {
    mountAll(root) {
      root.querySelectorAll('.widget[data-widget]').forEach(el => {
        if (el.dataset.mounted) return;
        el.dataset.mounted = '1';
        const fn = W[el.dataset.widget];
        if (fn) { try { fn(el, JSON.parse(el.dataset.args || '{}')); } catch (e) { el.textContent = 'Widget error: ' + e.message; } }
      });
    }
  };
})();

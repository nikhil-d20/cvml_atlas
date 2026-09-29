// Detection architecture pages. P(k, title, subtitle, blocks, outputs) builds one pipeline stage.
(function () {
const R = String.raw, T = CV.tbl;
const P = (k, t, d, b, o) => ({ k, t, d, b, o });
const IMG = s => P('input', 'Image', s || '640×640×3 (letterbox)');

// ---------- shared YOLO building blocks ----------
const C2F = { t: 'C2f block (YOLOv8)', k: 'backbone', ops: ['Conv 1×1 (c → 2c′)', 'Split → [a, b]', 'Bottleneck(b) ×n (3×3 → 3×3, +skip)', 'Concat [a, b, y1 … yn]', 'Conv 1×1 → c_out'], skips: [[1, 3, 'a, b pass through']] };
const C3K2 = { t: 'C3k2 block (YOLO11 / YOLO26)', k: 'backbone', ops: ['Conv 1×1 (c → 2c′)', 'Split → [a, b]', 'n × (Bottleneck or C3k: CSP with 3×3 bottlenecks)', 'Concat [a, b, y1 … yn]', 'Conv 1×1 → c_out'], skips: [[1, 3, 'concat']] };
const SPPF = { t: 'SPPF (spatial pyramid pooling – fast)', k: 'backbone', ops: ['Conv 1×1 (c → c/2)', 'MaxPool 5×5 → y1', 'MaxPool 5×5 → y2', 'MaxPool 5×5 → y3', 'Concat [x, y1, y2, y3] → Conv 1×1'], skips: [[0, 4, 'x']], note: 'Three chained 5×5 pools = receptive fields 5, 9, 13 (same as SPP) at lower cost.' };
const C2PSA = { t: 'C2PSA (YOLO11 attention)', k: 'backbone', ops: ['Conv 1×1, split → [a, b]', 'PSA: multi-head self-attention on b (+skip)', 'FFN 1×1 convs (+skip)', 'Concat [a, b] → Conv 1×1'], skips: [[0, 3, 'a']] };
const DECOUPLED = { t: 'Decoupled anchor-free head (per level)', k: 'head', ops: ['Feature P_l', 'Box branch: 3×3 conv ×2 → 1×1 → 4 × reg_max bins (DFL)', 'Cls branch: 3×3 conv ×2 → 1×1 → K sigmoid scores', 'Decode: distances (l, t, r, b) × stride → xyxy', 'NMS (IoU 0.7) at inference'] };

CV.arch({
  id: 'yolo26', name: 'YOLO26', task: 'det', match: /^YOLO26/, date: '2026-01', org: 'Ultralytics',
  code: 'https://github.com/ultralytics/ultralytics', paper: 'https://docs.ultralytics.com/models/yolo26/', tags: 'yolo26,ultralytics,nms-free,end-to-end,edge,musgd,stal,progloss',
  tagline: 'Edge-first YOLO: end-to-end (no NMS) inference, DFL removed for simpler export, new loss balancing and small-target assignment, trained with the MuSGD optimizer. One family covers detect, segment, pose, OBB and classify.',
  anatomy: {
    backbone: 'CSP-style conv backbone in the YOLO11 lineage: strided 3×3 convs + C3k2 blocks, SPPF and a C2PSA attention block at stride 32.',
    neck: 'PAN-FPN: top-down upsample + concat + C3k2, then bottom-up strided conv + concat + C3k2 → P3/8, P4/16, P5/32.',
    head: 'End-to-end one-to-one detection head (NMS-free) with direct box regression (no DFL bins). Task heads for masks, keypoints, oriented boxes.',
    loss: 'ProgLoss (progressive balancing of loss terms during training) + STAL (small-target-aware label assignment); MuSGD = SGD combined with Muon-style orthogonalized updates.'
  },
  pipeline: [IMG(), P('backbone', 'CSP backbone', 'C3k2 + SPPF + C2PSA', ['Conv 3×3 s2 → Conv 3×3 s2 (÷4)', 'C3k2 → Conv s2 → C3k2 (P3/8)', 'Conv s2 → C3k2 (P4/16)', 'Conv s2 → C3k2 → SPPF → C2PSA (P5/32)'], 'P3, P4, P5'),
    P('neck', 'PAN-FPN', 'top-down + bottom-up', ['Upsample P5, concat P4 → C3k2', 'Upsample, concat P3 → C3k2 → out P3', 'Conv s2, concat → C3k2 → out P4', 'Conv s2, concat → C3k2 → out P5']),
    P('head', 'One-to-one head (NMS-free)', 'direct box regression, no DFL', ['Box: 4 values per point (no bins)', 'Cls: K sigmoid scores', 'Top-k selection instead of NMS', 'Optional: mask / pose / OBB branches']),
    P('output', 'Detections', 'final boxes, no post-processing')],
  blocks: [C3K2, SPPF, C2PSA],
  sections: [
    { h: 'What changed vs YOLO11', html: T(['Change', 'Why it matters'], [['End-to-end one-to-one head (no NMS)', 'Removes NMS latency and threshold tuning; export to TensorRT / CoreML / TFLite is a single graph.'], ['DFL removed', 'Plain box regression is simpler to export and faster on CPUs / NPUs.'], ['ProgLoss', 'Re-weights loss terms over training for more stable convergence.'], ['STAL', 'Label assignment that keeps enough positives for tiny objects (edge / aerial use-cases).'], ['MuSGD optimizer', 'Hybrid of SGD and Muon (orthogonalized momentum) borrowed from LLM training for faster, stabler convergence.'], ['CPU speed focus', 'Ultralytics reports up to ~43% faster CPU (ONNX) inference for the nano model vs YOLO11n.']]) },
    { h: 'Scaling (n → x)', html: T(['Model', 'COCO AP', 'Params (M)', 'GFLOPs', 'T4 TRT ms', 'CPU ONNX ms'], [['YOLO26n', '40.9', '2.4', '5.5', '1.7', '38.9'], ['YOLO26s', '48.6', '9.5', '20.9', '2.5', '87.2'], ['YOLO26m', '53.1', '20.4', '68.4', '4.7', '220.0'], ['YOLO26l', '55.0', '24.8', '86.8', '6.2', '286.2'], ['YOLO26x', '57.5', '55.7', '194.4', '11.8', '525.8']]) + '<p>Sizes change depth multiple (number of C3k2 repeats) and width multiple (channels), as in YOLO11.</p>' },
    { h: 'Training & deployment', html: '<ul><li>Ultralytics API: <code>YOLO("yolo26s.pt").train(data=..., epochs=100–300, imgsz=640)</code>; defaults: mosaic (off last 10 epochs), HSV, flip, scale 0.5.</li><li>Export: <code>model.export(format="onnx" | "engine" | "coreml" | "tflite")</code> — no NMS plugin needed.</li><li>License: AGPL-3.0 (enterprise license for closed-source use).</li></ul>' }
  ]
});

CV.arch({
  id: 'yolov12', name: 'YOLOv12', task: 'det', match: /^YOLOv12/, date: '2025-02', org: 'Univ. at Buffalo & UCAS (Tian, Ye, Doermann)',
  paper: 'https://arxiv.org/abs/2502.12524', code: 'https://github.com/sunsmarterjie/yolov12', tags: 'yolov12,area attention,r-elan,attention-centric yolo',
  tagline: 'Attention-centric YOLO: area attention (A²) and residual ELAN blocks bring self-attention into a real-time detector without losing speed.',
  anatomy: { backbone: 'Conv stem + C3k2 stages at high resolution, then A2C2f (area-attention) blocks at strides 16 and 32. No SPPF / PSA.', neck: 'PAN-FPN built from A2C2f / C3k2 blocks.', head: 'YOLO11/YOLOv8-style decoupled anchor-free head with DFL; NMS at inference.', loss: 'BCE (TAL-weighted) + CIoU + DFL.' },
  pipeline: [IMG(), P('backbone', 'Conv + R-ELAN backbone', 'attention only in deep stages', ['Conv s2 ×2 (÷4)', 'C3k2 (P2) → Conv s2 → C3k2 (P3/8)', 'Conv s2 → A2C2f ×4 (P4/16)', 'Conv s2 → A2C2f ×4 (P5/32)'], 'P3, P4, P5'),
    P('neck', 'PAN-FPN', 'A2C2f in top-down path', ['Up + concat → A2C2f', 'Up + concat → C3k2 (P3 out)', 'Conv s2 + concat → A2C2f (P4 out)', 'Conv s2 + concat → C3k2 (P5 out)']),
    P('head', 'Decoupled head + DFL', '', ['Cls + box branches per level', 'NMS']), P('output', 'Detections')],
  blocks: [{ t: 'Area attention (A²)', k: 'backbone', ops: ['Feature map H×W×C', 'Split into l = 4 areas (rows or columns)', 'Self-attention inside each area (FlashAttention)', '+ 7×7 depthwise conv "position perceiver" (no positional encoding)', 'MLP (ratio 1.2) → output'], skips: [[0, 4, '+']], note: 'Cost drops from (HW)² to (HW)²/l while keeping a large receptive field.' },
    { t: 'R-ELAN block', k: 'backbone', ops: ['Conv 1×1 transition', 'Area-attention blocks (stacked)', 'Concat features', 'Conv 1×1', '× scale 0.01, residual add'], skips: [[0, 4, 'residual']] }],
  sections: [{ h: 'Key ideas', html: '<ul><li><b>Area attention</b>: simple reshape-based partition instead of windows → linear-ish cost, no complex ops.</li><li><b>R-ELAN</b>: residual shortcut with a small scaling factor stabilizes the attention-heavy large models.</li><li>Removes positional encoding; FlashAttention required for the reported speed (Turing+/Ampere GPUs).</li><li>Improves AP by +1.0–1.2 over YOLO11 at similar latency (N: 40.6 vs 39.4).</li></ul>' }]
});

CV.arch({
  id: 'yolo11', name: 'YOLO11', task: 'det', match: /^YOLO11/, date: '2024-09', org: 'Ultralytics',
  code: 'https://github.com/ultralytics/ultralytics', paper: 'https://docs.ultralytics.com/models/yolo11/', tags: 'yolo11,c3k2,c2psa,ultralytics',
  tagline: 'Refined YOLOv8: C3k2 blocks and a C2PSA attention block give higher AP with fewer parameters. Same anchor-free decoupled head with DFL and NMS.',
  anatomy: { backbone: 'Conv(64,3,2) → Conv(128,3,2) → C3k2 → Conv s2 → C3k2 → Conv s2 → C3k2 → Conv s2 → C3k2 → SPPF → C2PSA.', neck: 'PAN-FPN with C3k2 blocks.', head: 'Detect: decoupled cls/box branches, DFL (reg_max 16), depthwise convs in cls branch; NMS.', loss: 'BCE (TAL, α=0.5, β=6, top-k 10) × 0.5 + CIoU × 7.5 + DFL × 1.5.' },
  pipeline: [IMG(), P('backbone', 'YOLO11 backbone', '', ['Conv 3×3 s2 (64) → Conv 3×3 s2 (128)', 'C3k2 (256) → Conv s2 (256)', 'C3k2 (512) → P3/8', 'Conv s2 → C3k2 (512) → P4/16', 'Conv s2 → C3k2 (1024) → SPPF → C2PSA → P5/32'], 'P3, P4, P5'),
    P('neck', 'PAN-FPN', '', ['Up ×2, concat P4 → C3k2 (512)', 'Up ×2, concat P3 → C3k2 (256) → out P3', 'Conv s2, concat → C3k2 (512) → out P4', 'Conv s2, concat → C3k2 (1024) → out P5']),
    P('head', 'Detect head', '8400 anchor points @640', ['Box: DFL 4×16 bins', 'Cls: K scores', 'NMS IoU 0.7']), P('output', 'Detections')],
  blocks: [C3K2, SPPF, C2PSA, DECOUPLED],
  sections: [{ h: 'Model scaling', html: T(['Size', 'depth mult.', 'width mult.', 'max channels', 'AP', 'Params'], [['n', '0.50', '0.25', '1024', '39.5', '2.6M'], ['s', '0.50', '0.50', '1024', '47.0', '9.4M'], ['m', '0.50', '1.00', '512', '51.5', '20.1M'], ['l', '1.00', '1.00', '512', '53.4', '25.3M'], ['x', '1.00', '1.50', '512', '54.7', '56.9M']]) },
    { h: 'Training recipe (Ultralytics defaults)', html: T(['hyp', 'value'], [['optimizer', 'SGD lr0 0.01 momentum 0.937 wd 5e-4 (auto → AdamW for short runs)'], ['schedule', 'linear to lrf 0.01, 3 warmup epochs'], ['aug', 'mosaic 1.0 (close last 10), HSV (0.015,0.7,0.4), fliplr 0.5, scale 0.5, translate 0.1'], ['epochs (COCO)', '600 (from scratch)']]) }]
});

CV.arch({
  id: 'yolov10', name: 'YOLOv10', task: 'det', match: /^YOLOv10/, date: '2024-05', org: 'Tsinghua University',
  paper: 'https://arxiv.org/abs/2405.14458', code: 'https://github.com/THU-MIG/yolov10', tags: 'yolov10,nms-free,dual assignment,one-to-one,cib,psa',
  tagline: 'First NMS-free YOLO: consistent dual assignments train a one-to-many head (rich supervision) and a one-to-one head (used at inference).',
  anatomy: { backbone: 'YOLOv8-style CSP backbone with efficiency changes: spatial-channel decoupled downsampling, compact inverted blocks (CIB) in deep stages, large-kernel DW convs, partial self-attention (PSA) at P5.', neck: 'PAN-FPN.', head: 'Two heads sharing the neck: one-to-many (top-10 TAL, training only) and one-to-one (top-1, inference). Lightweight cls branch.', loss: 'Same loss on both heads with a consistent matching metric (score^α · IoU^β) so the 1-to-1 head learns the best 1-to-many sample.' },
  pipeline: [IMG(), P('backbone', 'Efficiency-driven CSP backbone', '', ['Conv stem', 'C2f stages (P3)', 'SCDown: PW conv then DW s2 conv', 'C2fCIB stages (P4, P5)', 'SPPF → PSA'], 'P3–P5'), P('neck', 'PAN-FPN', '', ['C2f / C2fCIB fusion blocks']),
    P('head', 'Dual heads', '', ['One-to-many head (train only)', 'One-to-one head (train + inference)', 'No NMS']), P('output', 'Detections', 'top-k by score')],
  blocks: [{ t: 'Compact Inverted Block (CIB)', k: 'backbone', ops: ['DW 3×3', 'PW 1×1 (expand)', 'DW 3×3 (or large-kernel 7×7)', 'PW 1×1 (project)', 'DW 3×3'], skips: [[0, 4, '+']] }, { t: 'Consistent dual assignment', k: 'head', ops: ['Shared neck features', 'Head A: one-to-many (top-k = 10)', 'Head B: one-to-one (top-1)', 'Same metric m = s^α · IoU^β for both', 'Inference uses Head B only'] }],
  f: [R`m(\alpha,\beta)=s\cdot p^{\alpha}\cdot\mathrm{IoU}(\hat b,b)^{\beta},\;\;\alpha_{o2o}=\alpha_{o2m},\;\beta_{o2o}=\beta_{o2m}`],
  sections: [{ h: 'Why NMS-free matters', html: '<p>NMS latency grows with the number of boxes and needs threshold tuning per dataset. With a one-to-one head each object produces one high-scoring box, so post-processing is a top-k. YOLOv10-S is 1.8× faster than RT-DETR-R18 at similar AP (paper).</p>' }]
});

CV.arch({
  id: 'yolov9', name: 'YOLOv9', task: 'det', match: /^YOLOv9/, date: '2024-02', org: 'Academia Sinica (Wang, Liao)',
  paper: 'https://arxiv.org/abs/2402.13616', code: 'https://github.com/WongKinYiu/yolov9', tags: 'yolov9,gelan,pgi,programmable gradient information',
  tagline: 'GELAN backbone plus Programmable Gradient Information: an auxiliary reversible branch supplies reliable gradients during training and is removed at inference.',
  anatomy: { backbone: 'GELAN (generalized ELAN): CSP split + stacked computational blocks (RepNCSP) with dense concatenation.', neck: 'GELAN-based PAN with SPPELAN.', head: 'Decoupled anchor-free head with DFL (v8-style); NMS.', loss: 'BCE + CIoU + DFL on main and auxiliary branches (PGI).' },
  pipeline: [IMG(), P('backbone', 'GELAN', '', ['Conv stem', 'RepNCSPELAN4 stages', 'ADown downsampling'], 'P3–P5'), P('neck', 'GELAN PAN + SPPELAN'), P('head', 'Main head + auxiliary reversible branch', 'aux branch only in training', ['Multi-level auxiliary information', 'Decoupled DFL head']), P('output', 'Detections')],
  blocks: [{ t: 'GELAN block (RepNCSPELAN4)', k: 'backbone', ops: ['Conv 1×1 → split [a, b]', 'b → RepNCSP → Conv 3×3 → c', 'c → RepNCSP → Conv 3×3 → d', 'Concat [a, b, c, d] → Conv 1×1'], skips: [[0, 3, 'a, b']] }],
  sections: [{ h: 'Key idea: information bottleneck', html: '<p>Deep features lose input information layer by layer, so gradients to early layers become unreliable. PGI adds an auxiliary branch that keeps complete information during training. Because it is dropped at inference, it costs no inference time.</p>' }]
});

CV.arch({
  id: 'yolov8', name: 'YOLOv8', task: 'det', match: /^YOLOv8/, date: '2023-01', org: 'Ultralytics',
  code: 'https://github.com/ultralytics/ultralytics', paper: 'https://docs.ultralytics.com/models/yolov8/', tags: 'yolov8,c2f,anchor-free,dfl,tal',
  tagline: 'Anchor-free YOLO with C2f blocks, decoupled head, Task-Aligned assignment and Distribution Focal Loss — the template for YOLO11/12/26.',
  anatomy: { backbone: 'CSPDarknet with C2f blocks: Conv64 s2, Conv128 s2, C2f×3, Conv256 s2, C2f×6, Conv512 s2, C2f×6, Conv1024 s2, C2f×3, SPPF.', neck: 'PAN-FPN with C2f (no 1×1 lateral convs, direct concat).', head: 'Decoupled anchor-free head, DFL reg_max 16, sigmoid classes (no objectness).', loss: 'BCE (TAL) 0.5 + CIoU 7.5 + DFL 1.5.' },
  pipeline: [IMG(), P('backbone', 'CSPDarknet (C2f)', '', ['Conv s2 (64) → Conv s2 (128) → C2f ×3', 'Conv s2 (256) → C2f ×6 → P3', 'Conv s2 (512) → C2f ×6 → P4', 'Conv s2 (1024) → C2f ×3 → SPPF → P5'], 'P3, P4, P5'),
    P('neck', 'PAN-FPN (C2f)', '', ['Up + concat P4 → C2f', 'Up + concat P3 → C2f (P3 out)', 'Conv s2 + concat → C2f (P4 out)', 'Conv s2 + concat → C2f (P5 out)']), P('head', 'Decoupled head', '', ['Box: 4×16 DFL bins', 'Cls: K sigmoid', 'NMS']), P('output', 'Detections')],
  blocks: [C2F, SPPF, DECOUPLED],
  f: [R`t=s^{\alpha}\,u^{\beta}\;\;(\alpha=0.5,\beta=6,\;\text{top-}k=10)`, R`L=7.5\,L_{CIoU}+0.5\,L_{BCE}+1.5\,L_{DFL}`],
  sections: [{ h: 'Scaling', html: T(['Size', 'depth', 'width', 'max ch', 'AP', 'Params'], [['n', '0.33', '0.25', '1024', '37.3', '3.2M'], ['s', '0.33', '0.50', '1024', '44.9', '11.2M'], ['m', '0.67', '0.75', '768', '50.2', '25.9M'], ['l', '1.00', '1.00', '512', '52.9', '43.7M'], ['x', '1.00', '1.25', '512', '53.9', '68.2M']]) }]
});

CV.arch({
  id: 'yolov5', name: 'YOLOv5', task: 'det', match: /^YOLOv5/, date: '2020-06', org: 'Ultralytics',
  code: 'https://github.com/ultralytics/yolov5', tags: 'yolov5,anchor-based,c3,objectness,coupled head',
  tagline: 'The PyTorch YOLO that set the engineering standard: CSP backbone with C3 blocks, SPPF, PANet and an anchor-based coupled head with objectness.',
  anatomy: { backbone: 'CSPDarknet53 with C3 blocks (v6.0+: 6×6 s2 stem conv replacing Focus).', neck: 'SPPF + PANet.', head: 'Coupled anchor-based head: 1×1 conv per level → 3 anchors × (4 + 1 + K). Auto-anchor (k-means + GA) at train start.', loss: 'BCE cls + BCE objectness (level weights 4.0 / 1.0 / 0.4) + CIoU box.' },
  pipeline: [IMG(), P('backbone', 'CSPDarknet53 (C3)', '', ['Conv 6×6 s2 → Conv s2', 'C3 ×3 → Conv s2', 'C3 ×6 (P3) → Conv s2', 'C3 ×9 (P4) → Conv s2', 'C3 ×3 → SPPF (P5)']), P('neck', 'PANet'), P('head', 'Coupled anchor head', '3 anchors per cell per level', ['(x, y, w, h, obj, K classes) × 3']), P('output', 'Detections', 'after NMS')],
  blocks: [{ t: 'C3 block', k: 'backbone', ops: ['Conv 1×1 → branch a', 'Conv 1×1 → branch b → Bottleneck ×n', 'Concat [a, b]', 'Conv 1×1'], skips: [[0, 2, 'a']] }, SPPF],
  f: [R`b_{xy}=2\sigma(t_{xy})-0.5+c_{xy},\;\;b_{wh}=p_{wh}\,(2\sigma(t_{wh}))^2`]
});

CV.arch({
  id: 'yolox', name: 'YOLOX', task: 'det', match: /^YOLOX/, date: '2021-07', org: 'Megvii',
  paper: 'https://arxiv.org/abs/2107.08430', code: 'https://github.com/Megvii-BaseDetection/YOLOX', tags: 'yolox,anchor-free,simota,decoupled head',
  tagline: 'Made YOLO anchor-free with a decoupled head and SimOTA label assignment; the bridge between YOLOv5 and YOLOv8.',
  anatomy: { backbone: 'Modified CSPDarknet (YOLOv5-style).', neck: 'PAFPN.', head: 'Decoupled: 1×1 conv to 256, then two parallel 2×(3×3 conv) branches: cls, and reg + IoU-objectness.', loss: 'BCE cls + BCE obj + IoU loss; SimOTA dynamic-k assignment.' },
  pipeline: [IMG(), P('backbone', 'CSPDarknet'), P('neck', 'PAFPN'), P('head', 'Decoupled anchor-free head', '', ['cls branch', 'reg + obj branch', 'SimOTA in training']), P('output', 'Detections', 'NMS')],
  f: [R`\text{SimOTA: } c_{ij}=L^{cls}_{ij}+\lambda L^{reg}_{ij},\;\;k_i=\Big\lfloor\sum_{\text{top-}10}\mathrm{IoU}_{ij}\Big\rfloor`],
  sections: [{ h: 'Training notes', html: '<p>300 epochs, strong mosaic + mixup turned off for the last 15 epochs (the origin of "close mosaic"), SGD 0.01×bs/64, cosine, EMA.</p>' }]
});

CV.arch({
  id: 'rtmdet', name: 'RTMDet', task: 'det', match: /^RTMDet/, date: '2022-12', org: 'OpenMMLab (Shanghai AI Lab)',
  paper: 'https://arxiv.org/abs/2212.07784', code: 'https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet', tags: 'rtmdet,cspnext,dynamic soft label,mmdetection',
  tagline: 'Throughput-oriented one-stage detector: large-kernel CSPNeXt, shared-weight head and dynamic soft label assignment. Also powers RTMPose and RTMO.',
  anatomy: { backbone: 'CSPNeXt: CSP blocks with 5×5 depthwise convs, channel attention.', neck: 'CSPNeXt-PAFPN.', head: 'Head conv weights shared across levels, separate BN per level; anchor-free.', loss: 'QFL + GIoU; dynamic soft label assigner (soft IoU cost + center prior).' },
  pipeline: [IMG(), P('backbone', 'CSPNeXt'), P('neck', 'CSPNeXt-PAFPN'), P('head', 'Shared-conv head', 'separate BN per level'), P('output', 'Detections', 'NMS')],
  f: [R`C=\lambda_1C_{cls}+\lambda_2C_{reg}+\lambda_3C_{center},\;\;C_{reg}=-\log(\mathrm{IoU})`]
});

const HYBRID = { t: 'Efficient hybrid encoder', k: 'neck', ops: ['S5 (stride 32) tokens + 2-D sin-cos PE', 'AIFI: 1 transformer encoder layer on S5 only', 'CCFF: top-down fusion with S4, S3 (RepC3 blocks)', 'CCFF: bottom-up fusion', 'Flatten 3 levels → memory for the decoder'], note: 'Attention only on the smallest map (intra-scale); cross-scale mixing with cheap convs → most of the encoder cost disappears.' };
const DEC = { t: 'DETR decoder layer (deformable)', k: 'head', ops: ['Queries (content + box anchor)', 'Self-attention between queries', 'Multi-scale deformable cross-attention (few sampled points / level)', 'FFN', 'Box refinement Δ + class logits (aux loss per layer)'], skips: [[0, 1, '+'], [1, 2, '+']] };

CV.arch({
  id: 'rtdetr', name: 'RT-DETR / RT-DETRv2', task: 'det', match: /^RT-DETR/, date: '2023-04', org: 'Baidu',
  paper: 'https://arxiv.org/abs/2304.08069', code: 'https://github.com/lyuwenyu/RT-DETR', tags: 'rt-detr,rtdetrv2,hybrid encoder,aifi,ccff,real-time detr,query selection',
  tagline: 'First real-time DETR: an efficient hybrid encoder plus IoU-aware query selection beats YOLOs of its time with no NMS. v2 (2024-07) adds a bag of freebies and deployment-friendly sampling.',
  anatomy: { backbone: 'ResNet-18/34/50/101 (ResNet-D) or HGNetv2 (L/X) → S3, S4, S5.', neck: 'Efficient hybrid encoder: AIFI (self-attention on S5) + CCFF (conv cross-scale fusion).', head: 'IoU-aware query selection (top-300 encoder tokens) → 6-layer deformable transformer decoder (3 for R18) → class + box per query.', loss: 'Hungarian matching; VFL (IoU-aware) + L1 (5) + GIoU (2); auxiliary losses per decoder layer; denoising queries.' },
  pipeline: [IMG('640×640×3'), P('backbone', 'ResNet / HGNetv2', '', ['Stages 1–4', 'Take S3 (÷8), S4 (÷16), S5 (÷32)'], 'S3–S5'),
    P('encoder', 'Hybrid encoder', 'AIFI + CCFF', ['AIFI: transformer layer on S5', 'CCFF: PAN-like conv fusion', 'Encoder heads score all tokens'], '3-level memory'),
    P('decoder', 'Query selection + decoder', '300 queries, no NMS', ['Top-300 tokens by IoU-aware score → initial queries', '6 × deformable decoder layers', 'Iterative box refinement']), P('output', 'Detections', '300 (class, box); keep score > thr')],
  blocks: [HYBRID, DEC],
  f: [R`\mathcal L(\hat y,y)=\mathcal L_{box}(\hat b,b)+\mathcal L_{cls}(\hat c,\hat b,y,b)`, R`\text{VFL target} = \mathrm{IoU}(\hat b,b)\;\text{(IoU-aware classification)}`],
  sections: [{ h: 'Speed/accuracy knob', html: '<p>Because every decoder layer is supervised, you can drop the last decoder layers at inference to trade AP for speed without retraining.</p>' },
    { h: 'RT-DETRv2 changes', html: '<ul><li>Different number of sampling points per scale in deformable attention.</li><li>Discrete sampling operator replaces <code>grid_sample</code> for easier deployment.</li><li>Dynamic data augmentation (strong early, weak late) and scale-adaptive hyperparameters (backbone LR per size).</li></ul>' },
    { h: 'Training recipe', html: T(['hyp', 'value'], [['optimizer', 'AdamW lr 1e-4, backbone lr 1e-5, wd 1e-4'], ['batch / epochs', '16 / 72 (6× schedule)'], ['aug', 'photometric distort, zoom-out, IoU crop, flip, multi-scale 480–800'], ['EMA', '0.9999'], ['queries / DN groups', '300 / denoising on']]) }]
});

CV.arch({
  id: 'dfine', name: 'D-FINE', task: 'det', match: /^D-FINE/, date: '2024-10', org: 'USTC',
  paper: 'https://arxiv.org/abs/2410.13842', code: 'https://github.com/Peterande/D-FINE', tags: 'd-fine,fine-grained distribution refinement,fdr,go-lsd,localization distillation',
  tagline: 'Redefines DETR box regression as iterative refinement of per-edge probability distributions (FDR) with self-distillation from deep to shallow layers (GO-LSD).',
  anatomy: { backbone: 'HGNetv2 (B0–B5 for N → X).', neck: 'RT-DETR hybrid encoder (lighter for small sizes).', head: 'Deformable decoder whose box output is a distribution over offsets for each of the 4 edges, refined layer by layer.', loss: 'VFL + L1 + GIoU + FGL (fine-grained localization) + DDF (decoupled distillation focal) for GO-LSD.' },
  pipeline: [IMG(), P('backbone', 'HGNetv2'), P('encoder', 'Hybrid encoder (AIFI + CCFF)'), P('decoder', 'Decoder with FDR', '', ['Layer 1: initial box', 'Layers 2–L: refine edge distributions (residual)', 'Final layer teaches earlier layers (GO-LSD)']), P('output', 'Detections', 'no NMS')],
  blocks: [{ t: 'Fine-grained Distribution Refinement', k: 'head', ops: ['Initial box b⁰ from first decoder layer', 'Layer l predicts logits over N bins per edge', 'Edge offset = Σ W(n) · Pr_l(n) (non-uniform W)', 'b^l = b⁰ + offsets scaled by box size', 'Final-layer distributions distill into earlier layers'] }],
  f: [R`\mathbf{b}^{l}=\mathbf{b}^{0}+\{H,H,W,W\}\cdot\sum_{n=0}^{N}W(n)\,\Pr{}^{l}(n)`, R`\mathcal L_{FGL}=\sum_l \mathrm{IoU}\cdot\big(\omega_\leftarrow\mathrm{CE}(\Pr(n),n_\leftarrow)+\omega_\rightarrow\mathrm{CE}(\Pr(n),n_\rightarrow)\big)`],
  sections: [{ h: 'Why distributions?', html: '<p>Regressing box coordinates directly treats every edge as a fixed Dirac value. Distributions express uncertainty (occluded or blurry edges), and residual refinement of distributions converges to more precise boxes: +AP75 in particular.</p>' }]
});

CV.arch({
  id: 'deim', name: 'DEIM', task: 'det', match: /^DEIM-/, date: '2024-12', org: 'Intellindust AI Lab',
  paper: 'https://arxiv.org/abs/2412.04234', code: 'https://github.com/ShihuaHuang95/DEIM', tags: 'deim,dense o2o,matchability-aware loss,mal,fast convergence detr',
  tagline: 'A training framework (not a new network) that makes one-to-one DETR training converge ~2× faster: Dense O2O matching plus a Matchability-Aware Loss. Applied to D-FINE and RT-DETRv2.',
  anatomy: { backbone: 'Same as the base model (HGNetv2 for DEIM-D-FINE, ResNet for DEIM-RT-DETRv2).', neck: 'Hybrid encoder (unchanged).', head: 'Unchanged decoder.', loss: 'MAL replaces VFL; Dense O2O increases positives per image by mosaic/mixup-style image composition.' },
  pipeline: [IMG('mosaic-composed training image'), P('backbone', 'HGNetv2 / ResNet'), P('encoder', 'Hybrid encoder'), P('decoder', 'One-to-one decoder', 'more targets per image → more positives'), P('loss', 'MAL', 'matchability-aware focal loss'), P('output', 'Detections')],
  f: [R`\mathrm{MAL}(p,q,y)=\begin{cases}-\big(q^{\gamma}\log p+(1-q^{\gamma})\log(1-p)\big) & y=1\\ -p^{\gamma}\log(1-p) & y=0\end{cases},\;\;q=\mathrm{IoU}`],
  sections: [{ h: 'Why it works', html: '<p>One-to-one matching gives each object exactly one positive query, which is sparse supervision early in training. Composing images increases the number of objects (and therefore positives) per sample without adding one-to-many heads. MAL fixes VFL\'s weak gradients for low-IoU matches.</p>' }]
});

CV.arch({
  id: 'deimv2', name: 'DEIMv2', task: 'det', match: /^DEIMv2/, date: '2025-09', org: 'Intellindust AI Lab',
  code: 'https://github.com/Intellindust-AI-Lab/DEIMv2', tags: 'deimv2,dinov3,spatial tuning adapter,real-time detr',
  tagline: 'DEIM with DINOv3 features: a Spatial Tuning Adapter turns the single-scale ViT output into a multi-scale pyramid. Eight sizes from Atto (0.5M) to X (50M).',
  anatomy: { backbone: 'S–X: DINOv3-pretrained / distilled ViT; Atto–N: pruned HGNetv2 for ultra-light budgets.', neck: 'Spatial Tuning Adapter (STA) builds multi-scale features from the ViT map and adds fine detail; lightweight encoder.', head: 'DEIM-style one-to-one deformable decoder (NMS-free).', loss: 'DEIM training (Dense O2O + MAL).' },
  pipeline: [IMG(), P('backbone', 'DINOv3 ViT (S–X) or HGNetv2 (Atto–N)', '', ['Semantic, single-scale features']), P('neck', 'Spatial Tuning Adapter', '', ['Multi-scale pyramid from ViT map', 'Fine-grained detail branch']), P('decoder', 'DEIM decoder'), P('output', 'Detections', 'no NMS')],
  sections: [{ h: 'When to use', html: '<p>Strong accuracy per parameter at the small end (S reaches 50.9 AP with 9.7M params). The Atto, Femto and Pico sizes target microcontroller-class and mobile budgets.</p>' }]
});

CV.arch({
  id: 'rfdetr', name: 'RF-DETR', task: 'det', match: /^RF-DETR/, date: '2025-03', org: 'Roboflow',
  paper: 'https://arxiv.org/abs/2511.09554', code: 'https://github.com/roboflow/rf-detr', tags: 'rf-detr,dinov2,nas,weight-sharing,lw-detr,real-time transformer,segmentation',
  tagline: 'DINOv2 backbone + lightweight deformable decoder, with weight-sharing neural architecture search producing a Pareto family (N → 2XL). First real-time detector over 60 COCO AP; strong transfer to new domains (RF100-VL).',
  anatomy: { backbone: 'DINOv2 ViT with interleaved windowed and global attention blocks (patch size and windows are NAS knobs).', neck: 'Multi-scale projector (C2f-style) producing feature maps from ViT layers — no heavy transformer encoder (LW-DETR design).', head: 'Deformable cross-attention decoder with two-stage query selection; decoder depth and query count are searchable and can be cut at inference. Seg variant adds a mask head (query embeddings · pixel embeddings).', loss: 'Hungarian matching, IoU-aware classification loss + L1 + GIoU; group DETR style parallel query groups for faster convergence.' },
  pipeline: [IMG('square input, size is a NAS knob'), P('backbone', 'DINOv2 ViT', 'windowed + global attention', ['Patch embedding (NAS: patch size)', 'Transformer blocks, windowed attention in most', 'Global attention in a few blocks'], 'intermediate token maps'),
    P('neck', 'Multi-scale projector', 'LW-DETR style, no encoder', ['C2f-style projection', 'Up/down-sample to 2–3 scales']),
    P('decoder', 'Deformable decoder', 'two-stage query selection', ['Top-k proposals → queries', 'L decoder layers (NAS)', 'Class + box per query (+ mask)']), P('output', 'Detections / masks', 'no NMS')],
  blocks: [DEC],
  sections: [{ h: 'Weight-sharing NAS', html: '<p>One set of weights is trained while randomly sampling configurations: input resolution, patch size, number of decoder layers, number of queries, and number of attention windows. Afterwards, the Pareto-optimal configurations for a target latency are evaluated without retraining, and these become the N/S/M/L/XL/2XL checkpoints. The same search can be run on your own dataset.</p>' },
    { h: 'Fine-tuning', html: '<pre><code>from rfdetr import RFDETRMedium\nmodel = RFDETRMedium()\nmodel.train(dataset_dir="coco_format_dir", epochs=50, batch_size=8, lr=1e-4)</code></pre><p>Licenses: N–L Apache-2.0; XL/2XL under Roboflow PML.</p>' }]
});

CV.arch({
  id: 'detr', name: 'DETR', task: 'det', match: /^DETR \(/, date: '2020-05', org: 'Facebook AI (Meta)',
  paper: 'https://arxiv.org/abs/2005.12872', code: 'https://github.com/facebookresearch/detr', tags: 'detr,set prediction,hungarian,object queries,transformer detection',
  tagline: 'Detection as direct set prediction: CNN + transformer encoder-decoder + bipartite matching. No anchors, no NMS.',
  anatomy: { backbone: 'ResNet-50/101, C5 only (2048 × H/32 × W/32) → 1×1 conv to d = 256.', neck: '6-layer transformer encoder over all C5 pixels with 2-D sine positional encodings.', head: '6-layer transformer decoder with N = 100 learned object queries → FFN (class incl. "no object") + 3-layer MLP (box).', loss: 'Hungarian matching; CE (no-object weight 0.1) + L1 (5) + GIoU (2); auxiliary loss after every decoder layer.' },
  pipeline: [IMG('~800×1333'), P('backbone', 'ResNet-50', '', ['C5: 2048 × H/32 × W/32', '1×1 conv → 256']), P('encoder', 'Transformer encoder ×6', 'global self-attention', ['+ sine PE at every layer']),
    P('decoder', 'Transformer decoder ×6', '100 object queries', ['Self-attn among queries', 'Cross-attn to encoder memory', 'FFN']), P('head', 'Prediction FFNs', '', ['Linear → K+1 classes', 'MLP → (cx, cy, w, h)']), P('output', '100 predictions', 'drop "no object"')],
  f: [R`\hat\sigma=\arg\min_{\sigma\in\mathfrak S_N}\sum_i\mathcal L_{match}(y_i,\hat y_{\sigma(i)})`],
  sections: [{ h: 'Limitations → successors', html: '<ul><li>500 epochs to converge (queries have no spatial prior) → Deformable DETR, DAB, DN, DINO.</li><li>Poor small objects (single C5 scale) → multi-scale deformable attention.</li><li>O(HW²) encoder cost → deformable / hybrid encoders (RT-DETR).</li></ul>' }]
});

CV.arch({
  id: 'deformable-detr', name: 'Deformable DETR', task: 'det', match: /^Deformable DETR/, date: '2020-10', org: 'SenseTime',
  paper: 'https://arxiv.org/abs/2010.04159', code: 'https://github.com/fundamentalvision/Deformable-DETR', tags: 'deformable detr,multi-scale deformable attention,iterative refinement,two-stage',
  tagline: 'Replaces dense attention with multi-scale deformable attention (each query samples a few points on each level): 10× fewer epochs and much better small objects.',
  anatomy: { backbone: 'ResNet-50 C3–C5 + an extra stride-64 level (4 levels, 256 ch).', neck: '6-layer deformable encoder (8 heads × 4 levels × 4 points).', head: '6-layer decoder with deformable cross-attention; iterative bounding-box refinement; two-stage variant uses encoder proposals as queries (300).', loss: 'Focal loss cls + L1 + GIoU, Hungarian.' },
  pipeline: [IMG(), P('backbone', 'ResNet-50', '', ['C3, C4, C5 + extra conv (÷64)']), P('encoder', 'Deformable encoder ×6', 'MSDeformAttn'), P('decoder', 'Deformable decoder ×6', '300 queries', ['Iterative box refinement', 'Two-stage: encoder proposals']), P('output', 'Detections')],
  f: [R`\text{MSDeformAttn}(z_q,\hat p_q,\{x^l\})=\sum_{m=1}^{M}W_m\Big[\sum_{l=1}^{L}\sum_{k=1}^{K}A_{mlqk}\,W'_m\,x^l\big(\phi_l(\hat p_q)+\Delta p_{mlqk}\big)\Big]`]
});

CV.arch({
  id: 'dino-det', name: 'DINO (DETR with Improved deNoising anchOr boxes)', task: 'det', match: /^DINO(-4scale)? \(/, date: '2022-03', org: 'IDEA Research',
  paper: 'https://arxiv.org/abs/2203.03605', code: 'https://github.com/IDEA-Research/DINO', tags: 'dino detector,contrastive denoising,mixed query selection,look forward twice',
  tagline: 'The DETR that first topped COCO: contrastive denoising, mixed query selection and look-forward-twice box refinement on top of Deformable DETR.',
  anatomy: { backbone: 'ResNet-50 (4-scale / 5-scale) or Swin-L.', neck: '6-layer deformable encoder.', head: '6-layer deformable decoder, 900 queries; positional queries initialized from top-K encoder features, content queries learnable (mixed selection).', loss: 'Focal + L1 + GIoU; denoising loss on noised GT queries (positive and negative).' },
  pipeline: [IMG(), P('backbone', 'ResNet-50 / Swin-L'), P('encoder', 'Deformable encoder ×6'), P('decoder', 'Decoder ×6', '900 matching + denoising queries', ['Mixed query selection', 'Contrastive DN groups', 'Look forward twice']), P('output', 'Detections')],
  blocks: [{ t: 'Contrastive denoising (CDN)', k: 'head', ops: ['GT boxes + labels', 'Positive: small noise (< λ1) → must reconstruct GT', 'Negative: larger noise (λ1–λ2) → must predict "no object"', 'Attention mask hides GT info from matching queries'] }]
});

CV.arch({
  id: 'codetr', name: 'Co-DETR', task: 'det', match: /^Co-DETR/, date: '2022-11', org: 'SenseTime',
  paper: 'https://arxiv.org/abs/2211.12860', code: 'https://github.com/Sense-X/Co-DETR', tags: 'co-detr,collaborative hybrid assignment,one-to-many auxiliary heads',
  tagline: 'Adds auxiliary one-to-many heads (ATSS, Faster R-CNN) during training to supervise the encoder densely, then feeds their positives as extra decoder queries. Inference is plain DETR.',
  anatomy: { backbone: 'ResNet / Swin-L / ViT-L (with Objects365 pre-training for the 66 AP result).', neck: 'Deformable / DINO encoder + FPN-like multi-scale adapter for auxiliary heads.', head: 'DINO-style decoder (inference) + auxiliary ATSS / Faster R-CNN heads (training only).', loss: 'DETR loss + auxiliary head losses + customized positive-query losses.' },
  pipeline: [IMG(), P('backbone', 'ViT-L / Swin-L'), P('encoder', 'Encoder'), P('head', 'Aux one-to-many heads', 'training only', ['ATSS head', 'Faster R-CNN head']), P('decoder', 'DETR decoder', 'inference path'), P('output', 'Detections')]
});

CV.arch({
  id: 'grounding-dino', name: 'Grounding DINO', task: 'det', match: /^Grounding DINO/, date: '2023-03', org: 'IDEA Research',
  paper: 'https://arxiv.org/abs/2303.05499', code: 'https://github.com/IDEA-Research/GroundingDINO', tags: 'grounding dino,open-vocabulary,zero-shot detection,text prompt,phrase grounding',
  tagline: 'Open-set detector: detects anything described in text by fusing language and vision in the neck, query selection and decoder.',
  anatomy: { backbone: 'Image: Swin-T/L. Text: BERT-base.', neck: 'Feature enhancer: image↔text cross-attention + self-attention (6 layers).', head: 'Language-guided query selection (900) → cross-modality decoder (image cross-attn + text cross-attn) → box + similarity to each text token.', loss: 'Contrastive loss between queries and text tokens + L1 + GIoU.' },
  pipeline: [P('input', 'Image + text prompt', '"person . dog . red car ."'), P('backbone', 'Swin (image) + BERT (text)'), P('neck', 'Feature enhancer', 'bi-directional cross-attention'), P('decoder', 'Cross-modality decoder', 'language-guided queries'), P('output', 'Boxes + phrases', 'threshold on text similarity')],
  sections: [{ h: 'Practical', html: '<p>Box threshold ~0.35, text threshold ~0.25. Excellent auto-labeler: combine with SAM for masks (Grounded-SAM), then distill into a fast closed-set detector.</p>' }]
});

CV.arch({
  id: 'vitdet', name: 'ViTDet', task: 'det', match: /^ViTDet/, date: '2022-03', org: 'Meta AI (FAIR)',
  paper: 'https://arxiv.org/abs/2203.16527', code: 'https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet', tags: 'vitdet,plain vit backbone,simple feature pyramid,mae',
  tagline: 'Shows a plain, non-hierarchical ViT (MAE pre-trained) works as a detection backbone with just a simple feature pyramid and window attention.',
  anatomy: { backbone: 'Plain ViT-B/L/H, patch 16, window attention (14×14) in most blocks + 4 global-attention blocks.', neck: 'Simple feature pyramid from the last stride-16 map: deconv ×2 (÷4), deconv (÷8), identity (÷16), max-pool (÷32).', head: 'Mask R-CNN or Cascade Mask R-CNN.', loss: 'Standard R-CNN losses; LSJ augmentation, 100 epochs.' },
  pipeline: [IMG('1024×1024 (LSJ)'), P('backbone', 'Plain ViT (MAE)', 'single scale ÷16'), P('neck', 'Simple feature pyramid', '', ['2 deconvs → ÷4', '1 deconv → ÷8', 'identity → ÷16', 'max-pool → ÷32']), P('head', 'Cascade Mask R-CNN'), P('output', 'Boxes + masks')]
});

CV.arch({
  id: 'rcnn', name: 'Faster / Mask / Cascade R-CNN', task: 'det', match: /R-CNN/, date: '2015-06', org: 'Microsoft Research / FAIR',
  paper: 'https://arxiv.org/abs/1506.01497', code: 'https://github.com/facebookresearch/detectron2', tags: 'faster r-cnn,mask r-cnn,cascade r-cnn,two-stage,rpn,roialign',
  tagline: 'Two-stage detection: a Region Proposal Network proposes boxes, RoIAlign crops features, and a second head classifies and refines them. Mask R-CNN (2017) adds a mask branch; Cascade R-CNN adds stages with rising IoU thresholds.',
  anatomy: { backbone: 'ResNet-50/101 (or any) C2–C5.', neck: 'FPN → P2–P6, 256 channels.', head: 'RPN (3×3 conv → objectness + deltas, 3 anchors/location) → RoIAlign 7×7 → 2×FC-1024 → class + box; mask head: 4× conv3×3 + deconv on 14×14 RoIs → 28×28 masks.', loss: 'RPN: BCE + Smooth L1; RoI: CE + Smooth L1; mask: per-pixel BCE.' },
  pipeline: [IMG('short side 800'), P('backbone', 'ResNet-50', '', ['C2–C5']), P('neck', 'FPN', '', ['P2–P6, 256 ch']), P('head', 'Stage 1: RPN', '', ['~2000 proposals (NMS 0.7)']), P('head', 'Stage 2: RoI heads', '', ['RoIAlign 7×7 (box) / 14×14 (mask)', 'Box head: 2 FC', 'Mask head: FCN → 28×28']), P('output', 'Boxes (+ masks)', 'NMS 0.5')],
  f: [R`L=L_{cls}^{rpn}+L_{box}^{rpn}+L_{cls}^{roi}+L_{box}^{roi}\;(+L_{mask})`]
});

CV.arch({
  id: 'retinanet', name: 'RetinaNet', task: 'det', match: /^RetinaNet/, date: '2017-08', org: 'FAIR',
  paper: 'https://arxiv.org/abs/1708.02002', code: 'https://github.com/facebookresearch/detectron2', tags: 'retinanet,focal loss,one-stage,anchors',
  tagline: 'One-stage detector that matched two-stage accuracy by introducing focal loss to handle the extreme foreground/background imbalance.',
  anatomy: { backbone: 'ResNet C3–C5.', neck: 'FPN P3–P7.', head: 'Two subnets shared across levels: 4× (3×3 conv 256 + ReLU) → cls (K·A sigmoid) and box (4·A); A = 9 anchors (3 scales × 3 ratios).', loss: 'Focal loss (γ=2, α=0.25) + Smooth L1; anchors IoU ≥ 0.5 positive, < 0.4 negative.' },
  pipeline: [IMG(), P('backbone', 'ResNet'), P('neck', 'FPN P3–P7'), P('head', 'Class + box subnets', '9 anchors per location'), P('output', 'Detections', 'NMS')],
  f: [R`\mathrm{FL}(p_t)=-\alpha_t(1-p_t)^{\gamma}\log p_t,\;\;b_{cls}=-\log\tfrac{1-\pi}{\pi},\;\pi=0.01`]
});
})();

// Weekly content updates. Appended by the weekly updater agent (see UPDATING.md) — newest packet at the bottom.
// Each packet: CV.update({ date, changelog: [...], items: [{tab, section, sectionTitle?, item}], models: [{board, row}] })
// Anything with an `added` date in the last 30 days shows a NEW tag in the app.

CV.update({
  date: '2026-09-29',
  changelog: [
    { text: 'Initial release: 10 knowledge tabs, 7 leaderboards with release dates, favourites & compare, live news.' }
  ]
});

CV.update({
  date: '2026-09-29',
  changelog: [
    { text: 'Model Zoo: 76 architecture pages with backbone → neck → head block diagrams; tap any leaderboard model name.' },
    { text: 'New Foundations section: Model Anatomy (stem, backbone, neck, head per task).' },
    { text: 'Leaderboards: AP50 / AP75 / APs / APm / APl, top-5, AR columns from official papers + a column glossary.' },
    { text: 'Next-ViT S/B/L added (ImageNet, ADE20K UperNet) plus two same-protocol backbone boards (Semantic FPN, Mask R-CNN 1×).', link: 'https://github.com/bytedance/Next-ViT' }
  ],
  models: ['Next-ViT-S', 'Next-ViT-B', 'Next-ViT-L'].map(n => ({ board: 'cls-in1k', row: { n, d: '2022-07' } }))
    .concat(['Next-ViT-S + UperNet', 'Next-ViT-B + UperNet', 'Next-ViT-L + UperNet'].map(n => ({ board: 'seg-ade', row: { n, d: '2022-07' } })))
    .concat(['Next-ViT-S', 'Next-ViT-B', 'Next-ViT-L'].map(n => ({ board: 'seg-ade-fpn', row: { n, d: '2022-07' } })))
    .concat(['Next-ViT-S', 'Next-ViT-B', 'Next-ViT-L'].map(n => ({ board: 'seg-maskrcnn', row: { n, d: '2022-07' } })))
    .concat([{ board: 'det-rt', row: { n: 'D-FINE-M', d: '2024-10' } }]),
  items: ['parts', 'pertask', 'fmap', 'backbones', 'necks', 'heads'].map(id => ({ tab: 'core', section: 'anatomy', item: { id } }))
});

CV.update({
  date: '2026-09-30',
  changelog: [
    { text: 'FP / FN Playbook: interactive strategy advisor, P/R/F-beta calculator, lever cheat-sheet, error analysis and data-collection guide (Practical Guide tab).' },
    { text: 'LR schedulers rewritten: 16 schedulers with plotted curves, when-to-use guide and an interactive schedule explorer with PyTorch code.' },
    { text: 'New Few-shot & Data-efficient Learning tab (metric learning, meta-learning, CLIP adapters, few-shot detection & segmentation).' },
    { text: 'Knowledge Distillation section (process, logit / feature / relation / self / semi-supervised KD) and Practical Training Concepts.' },
    { text: 'Encoders & Decoders section in Foundations.' }
  ],
  items: [].concat(
    ['advisor', 'calc', 'cost', 'levers', 'analysis', 'workflow', 'inputs'].map(id => ({ tab: 'hp', section: 'errors', item: { id } })),
    ['why', 'constant', 'warmup', 'step', 'multistep', 'exponential', 'linear', 'poly', 'cosine', 'sgdr', 'onecycle', 'wsd', 'invsqrt', 'cyclic', 'plateau', 'llrd', 'schedcmp', 'explorer'].map(id => ({ tab: 'train', section: 'sched', item: { id } })),
    ['process', 'logit', 'feature', 'relation', 'self', 'semi', 'foundation', 'gap', 'other', 'kdhp'].map(id => ({ tab: 'train', section: 'kd', item: { id } })),
    ['units', 'splits', 'preproc', 'freeze', 'ckpt', 'curves', 'noise', 'threshold', 'throughput'].map(id => ({ tab: 'train', section: 'practice', item: { id } })),
    ['what', 'kinds', 'seg', 'det', 'ae', 'gen', 'choose'].map(id => ({ tab: 'core', section: 'encdec', item: { id } })),
    ['terms', 'guide', 'bench'].map(id => ({ tab: 'fewshot', section: 'setup', item: { id } })),
    ['proto', 'siamese', 'cosine'].map(id => ({ tab: 'fewshot', section: 'metric', item: { id } })),
    [{ tab: 'fewshot', section: 'meta', item: { id: 'maml' } }],
    ['probe', 'clip', 'peft'].map(id => ({ tab: 'fewshot', section: 'transfer', item: { id } })),
    ['tfa', 'modern'].map(id => ({ tab: 'fewshot', section: 'fsod', item: { id } })),
    ['map', 'sam'].map(id => ({ tab: 'fewshot', section: 'fsseg', item: { id } })),
    ['aug', 'semi', 'recipe'].map(id => ({ tab: 'fewshot', section: 'data', item: { id } })))
});

CV.update({
  date: '2026-10-19',
  changelog: [{ text: 'YOLO27 (preview, not yet released) added to Real-time detection with preliminary COCO numbers from the official docs, plus an architecture page.', link: 'https://docs.ultralytics.com/models/yolo27/' }],
  models: [
    ['YOLO27n (preview)', 42.6, 3.0, 7.2, '0.44'], ['YOLO27s (preview)', 50.1, 11.8, 28.2, '0.64'],
    ['YOLO27m (preview)', 56.1, 22.8, 65.0, '1.01'], ['YOLO27l (preview)', 60.5, 72.3, 165.3, '1.86']
  ].map(([n, m, params, flops, rtx]) => ({ board: 'det-rt', row: {
    n, d: '2026-10', fam: 'YOLO (Ultralytics)', m, params, flops,
    notes: `ANNOUNCED, NOT RELEASED (no weights yet) — preliminary docs numbers; ${rtx} ms on RTX PRO 6000 TensorRT11 (not T4)${n.startsWith('YOLO27l') ? '; 61.4 mAP @800' : ''}${/27[ns]/.test(n) ? '; NMS by default, nms=False for NMS-free' : '; NMS-free query/attention design'}`,
    src: 'https://docs.ultralytics.com/models/yolo27/' } })),
  items: [{ tab: 'det', section: 'arch', item: { id: 'yolo', h: CV.tbl(['Version', 'Date', 'Key changes'], [['YOLOv1', '2015-06', 'S×S grid regression, single pass'], ['YOLOv2/9000', '2016-12', 'anchors (k-means), BN, high-res fine-tune'], ['YOLOv3', '2018-04', 'Darknet-53, 3 scales, logistic cls'], ['YOLOv4', '2020-04', 'CSPDarknet, SPP, PAN, Mosaic, CIoU, Mish'], ['YOLOv5', '2020-06', 'PyTorch, auto-anchor, SiLU, SPPF, strong engineering'], ['YOLOX', '2021-07', 'anchor-free, decoupled head, SimOTA'], ['YOLOv6 / v7', '2022-06/07', 'RepVGG re-param / E-ELAN, aux heads'], ['YOLOv8', '2023-01', 'anchor-free, C2f, decoupled head, TAL, DFL'], ['YOLOv9', '2024-02', 'GELAN, programmable gradient information (PGI)'], ['YOLOv10', '2024-05', 'NMS-free dual assignment, efficiency-driven design'], ['YOLO11', '2024-09', 'C3k2, C2PSA attention block'], ['YOLOv12', '2025-02', 'area attention (A2), R-ELAN, FlashAttention'], ['YOLO26', '2026-01', 'end-to-end NMS-free, DFL removed, ProgLoss + STAL (small targets), MuSGD optimizer; faster CPU inference'], ['<a href="#/arch/yolo27">YOLO27</a> (preview)', '2026-10 (announced)', 'N/S: conv with fine + coarse maps only and a training-only foreground alignment branch; M: transformer decoder over object queries; L: UltraViT backbone; NMS-free M/L. Not yet released.']]) } }]
});

CV.update({
  date: '2026-10-19',
  changelog: [{ text: 'New AutoML for Computer Vision tab: what to automate, search algorithms (TPE, GP-BO, ASHA, PBT, NAS), 20+ tools with repo links and verified status, recipes (Ultralytics tuner, Optuna 5, Ray Tune, AutoGluon, threshold search), best practices and a budget estimator.' }],
  items: [].concat(
    ['map', 'when', 'cost'].map(id => ({ tab: 'automl', section: 'what', item: { id } })),
    ['basic', 'tpe', 'gp', 'cmaes', 'mf', 'pbt', 'mo', 'nas'].map(id => ({ tab: 'automl', section: 'algos', item: { id } })),
    ['hpo', 'cvauto', 'agents', 'cloud', 'legacy'].map(id => ({ tab: 'automl', section: 'tools', item: { id } })),
    ['yolo', 'optuna', 'multi', 'thresh', 'autogluon', 'ray'].map(id => ({ tab: 'automl', section: 'recipes', item: { id } })),
    ['space', 'practices', 'pitfalls'].map(id => ({ tab: 'automl', section: 'best', item: { id } })))
});

CV.update({
  date: '2026-10-03',
  changelog: [{ text: 'Added HSMNet (multi-resolution steganalysis classifier, 2025) to Classification and the Model Zoo, plus a card on steganalysis classifiers.', link: 'https://www.sciencedirect.com/science/article/abs/pii/S0020025525009600' }],
  items: [{
    tab: 'cls', section: 'arch', item: {
      id: 'steganalysis', t: 'Steganalysis classifiers (HSMNet, SRNet, Yedroudj-Net, …)',
      tags: 'hsmnet,hsm net,steganalysis,stego detection,srnet,yedroudj-net,zhu-net,xu-net,ye-net,cover stego,binary classification,srm filters,hybrid dilated convolution',
      s: 'Binary classifiers that decide whether an image hides a secret message (cover vs stego). The signal is a tiny noise residual, so the architecture rules are almost the opposite of ImageNet classifiers.',
      f: [String.raw`P_E=\min_{P_{FA}}\tfrac12\big(P_{FA}+P_{MD}\big)`],
      h: CV.tbl(['Model', 'Year', 'Key idea'], [
        ['Xu-Net', '2016', 'fixed high-pass filter preprocessing, absolute-value layer, TanH, BN'],
        ['Ye-Net', '2017', '30 SRM high-pass filters as first layer + truncated linear unit (TLU)'],
        ['Yedroudj-Net', '2018', 'SRM filters + TLU + BN, compact and strong baseline'],
        ['SRNet', '2018', 'learns its own preprocessing; no pooling in the first 7 layers; residual blocks — widely used reference'],
        ['Zhu-Net', '2019', 'separable convolutions + spatial pyramid pooling for arbitrary input sizes'],
        ['<a href="#/arch/hsmnet">HSMNet</a>', '2025', 'multi-resolution: vanilla-conv local branch + hybrid-dilated-conv global branch, SE attention, hierarchical fusion']]) +
        String.raw`<p><b>Design rules (vs ImageNet nets):</b> suppress image content and amplify noise residuals (high-pass / SRM filters or learned equivalents); avoid early pooling or strided downsampling that averages away the signal; never resize the input; use global pooling at the end so inputs of any size work. Benchmarks: BOSSbase / BOWS2 (spatial domain) and ALASKA #2; algorithms: WOW, S-UNIWARD, HILL, MiPOD; payloads in bits per pixel (bpp).</p>`,
      tip: 'Treat it like a fine-grained binary classifier with a pair constraint: put each cover and its stego version in the same batch. Use only flips and 90° rotations as augmentation, and report error per embedding algorithm and payload.',
      src: 'https://www.sciencedirect.com/science/article/abs/pii/S0020025525009600'
    }
  }]
});

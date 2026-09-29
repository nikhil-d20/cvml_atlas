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

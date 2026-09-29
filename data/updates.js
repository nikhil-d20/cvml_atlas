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

// Pose architecture pages.
(function () {
const R = String.raw, T = CV.tbl;
const P = (k, t, d, b, o) => ({ k, t, d, b, o });
const CROP = P('input', 'Person crop', '256×192 from a person detector');

CV.arch({
  id: 'hrnet', name: 'HRNet', task: 'pose', match: /^HRNet/, date: '2019-02', org: 'Microsoft Research Asia',
  paper: 'https://arxiv.org/abs/1902.09212', code: 'https://github.com/open-mmlab/mmpose', tags: 'hrnet,high resolution,multi-branch,heatmap,top-down',
  tagline: 'Keeps a high-resolution branch through the whole network and repeatedly fuses it with lower-resolution branches: precise and semantic features without an encoder-decoder.',
  anatomy: { backbone: 'Stem (2 × 3×3 s2) → 4 stages; branches at ÷4, ÷8, ÷16, ÷32 added stage by stage; W32 = 32 channels in the high-res branch (32/64/128/256).', neck: 'Multi-resolution fusion units after every module (upsample 1×1+nearest, downsample 3×3 s2).', head: '1×1 conv on the ÷4 branch → K heatmaps (64×48 for 256×192 input).', loss: 'Target-weighted MSE on Gaussian heatmaps (σ = 2).' },
  pipeline: [CROP, P('backbone', 'HRNet-W32', '', ['Stem ÷4', 'Stage 1: 1 branch (bottlenecks)', 'Stage 2: 2 branches', 'Stage 3: 3 branches (4 modules)', 'Stage 4: 4 branches (3 modules)'], 'high-res ÷4 branch'),
    P('neck', 'Repeated multi-scale fusion', 'inside the backbone'), P('head', 'Heatmap head', '', ['1×1 conv → 17 heatmaps', 'Flip test + DARK / UDP decoding']), P('output', '17 keypoints')],
  blocks: [{ t: 'Exchange (fusion) unit, 3 branches', k: 'backbone', ops: ['Branch ÷4, ÷8, ÷16 features', 'Higher → lower: 3×3 s2 conv(s)', 'Lower → higher: 1×1 conv + nearest upsample', 'Sum at each resolution → ReLU'] }]
});

CV.arch({
  id: 'simplebaseline', name: 'SimpleBaseline', task: 'pose', match: /^SimpleBaseline/, date: '2018-04', org: 'Microsoft Research Asia',
  paper: 'https://arxiv.org/abs/1804.06208', tags: 'simplebaseline,deconvolution head,resnet pose',
  tagline: 'ResNet + three deconvolution layers → heatmaps. Showed a simple design is enough for strong top-down pose.',
  anatomy: { backbone: 'ResNet-50/101/152, C5 (÷32).', neck: '3 × (4×4 deconv, 256 ch, stride 2, BN, ReLU) → ÷4.', head: '1×1 conv → K heatmaps.', loss: 'Heatmap MSE.' },
  pipeline: [CROP, P('backbone', 'ResNet', '', [], 'C5 ÷32'), P('neck', 'Deconv upsampler', '', ['Deconv ×3 → ÷4']), P('head', '1×1 conv → heatmaps'), P('output', 'Keypoints')]
});

CV.arch({
  id: 'vitpose', name: 'ViTPose', task: 'pose', match: /^ViTPose/, date: '2022-04', org: 'Univ. of Sydney / JD Explore',
  paper: 'https://arxiv.org/abs/2204.12484', code: 'https://github.com/ViTAE-Transformer/ViTPose', tags: 'vitpose,plain vit,mae,pose transformer',
  tagline: 'Plain MAE-pretrained ViT + a very light decoder is enough for state-of-the-art pose; scales from 100M to 1B parameters.',
  anatomy: { backbone: 'Plain ViT-B/L/H/G, patch 16, on a 256×192 crop → 16×12 tokens.', neck: 'Classic decoder: 2 × (deconv 4×4 s2 + BN + ReLU) or simple: bilinear ×4 + ReLU + 3×3 conv.', head: '1×1 / 3×3 conv → K heatmaps at ÷4.', loss: 'Heatmap MSE; ViTPose++ adds task-specific MoE FFNs for multi-dataset training.' },
  pipeline: [CROP, P('backbone', 'Plain ViT (MAE)', '', ['Patch 16 → 16×12 tokens', 'L transformer blocks']), P('neck', 'Light decoder', '', ['Deconv ×2 or bilinear ×4']), P('head', 'Heatmap conv'), P('output', 'Keypoints')]
});

CV.arch({
  id: 'rtmpose', name: 'RTMPose', task: 'pose', match: /^RTMPose/, date: '2023-03', org: 'OpenMMLab (Shanghai AI Lab)',
  paper: 'https://arxiv.org/abs/2303.07399', code: 'https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose', tags: 'rtmpose,simcc,gau,cspnext,real-time pose',
  tagline: 'Real-time top-down pose with SimCC coordinate classification and a Gated Attention Unit head; > 90 FPS on CPU for the m model at 75.8 AP.',
  anatomy: { backbone: 'CSPNeXt (from RTMDet), sizes t / s / m / l / x.', neck: 'None.', head: '7×7 conv → flatten → FC → GAU (gated attention over keypoint tokens) → two linear classifiers for x and y (split ratio 2).', loss: 'KL divergence to Gaussian-smoothed 1-D labels (SimCC); two-stage augmentation; EMA.' },
  pipeline: [CROP, P('backbone', 'CSPNeXt'), P('head', 'SimCC head', '', ['7×7 conv → K tokens', 'FC → GAU', 'Linear → x logits (W·2)', 'Linear → y logits (H·2)']), P('output', 'Keypoints', 'argmax / 2')],
  f: [R`\mathcal L=\sum_k D_{KL}\big(\mathcal N_{1D}(x_k)\,\|\,\mathrm{softmax}(z^x_k/\tau)\big)+D_{KL}\big(\mathcal N_{1D}(y_k)\,\|\,\mathrm{softmax}(z^y_k/\tau)\big)`]
});

CV.arch({
  id: 'rtmo', name: 'RTMO', task: 'pose', match: /^RTMO/, date: '2023-12', org: 'Tsinghua / OpenMMLab',
  paper: 'https://arxiv.org/abs/2312.07526', code: 'https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo', tags: 'rtmo,one-stage pose,dual 1d heatmap',
  tagline: 'One-stage multi-person pose: YOLO-style detector predicts per-instance dual 1-D heatmaps for coordinates, closing most of the gap to top-down methods at constant cost.',
  anatomy: { backbone: 'CSPDarknet.', neck: 'Hybrid encoder / PAN.', head: 'Per-anchor box + dual 1-D heatmap keypoint head (dynamic coordinate classifier).', loss: 'Box losses + MLE-based coordinate classification loss + OKS.' },
  pipeline: [P('input', 'Full image', '640×640'), P('backbone', 'CSPDarknet'), P('neck', 'PAN / hybrid encoder'), P('head', 'Box + 1-D heatmap head'), P('output', 'People × 17 keypoints')]
});

CV.arch({
  id: 'higherhrnet', name: 'HigherHRNet', task: 'pose', match: /^HigherHRNet/, date: '2019-08', org: 'UIUC / Microsoft',
  paper: 'https://arxiv.org/abs/1908.10357', tags: 'higherhrnet,bottom-up,associative embedding',
  tagline: 'Bottom-up pose: HRNet plus a deconvolution to ÷2 resolution for small people, with associative-embedding tags to group keypoints into persons.',
  anatomy: { backbone: 'HRNet-W32/W48.', neck: 'Deconv module → ÷2 high-resolution heatmaps.', head: 'Heatmaps (K) + tag maps (K) at ÷4 and ÷2; heatmap aggregation across scales.', loss: 'Heatmap MSE + associative embedding (pull/push) loss.' },
  pipeline: [P('input', 'Full image', '512 / 640'), P('backbone', 'HRNet'), P('neck', 'Deconv to ÷2'), P('head', 'Heatmaps + tags'), P('output', 'Grouped people')]
});

CV.arch({
  id: 'openpose', name: 'OpenPose', task: 'pose', match: /^OpenPose/, date: '2016-11', org: 'Carnegie Mellon University',
  paper: 'https://arxiv.org/abs/1812.08008', code: 'https://github.com/CMU-Perceptual-Computing-Lab/openpose', tags: 'openpose,part affinity fields,bottom-up,multi-person',
  tagline: 'Bottom-up multi-person pose with Part Affinity Fields: 2-D vector fields that encode limb direction, used to link keypoints between people.',
  anatomy: { backbone: 'VGG-19 first 10 layers.', neck: 'Multi-stage refinement CNN (PAF stages then heatmap stages).', head: 'Confidence maps S (K) + PAFs L (2 × limbs).', loss: 'L2 on maps at every stage (intermediate supervision).' },
  pipeline: [P('input', 'Full image'), P('backbone', 'VGG-19 (10 layers)'), P('neck', 'Iterative stages', '', ['PAF stages', 'Heatmap stages']), P('head', 'Heatmaps + PAFs'), P('output', 'People', 'bipartite matching of limbs')]
});
})();

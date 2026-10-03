// Backbone / classification architecture pages. These backbones also appear in detection & segmentation boards.
(function () {
const R = String.raw, T = CV.tbl;
const P = (k, t, d, b, o) => ({ k, t, d, b, o });
const IMG = s => P('input', 'Image', s || '224×224×3');
const CLSHEAD = P('head', 'Classifier head', '', ['Global avg pool (or CLS token)', 'Linear → 1000']);
const DENSE = 'As a detection / segmentation backbone, stages give C2–C5 (strides 4–32) to FPN, UperNet, Semantic FPN or Mask R-CNN; the classifier head is dropped.';

CV.arch({
  id: 'nextvit', name: 'Next-ViT', aka: 'nextvit next vit small base large', task: 'cls', match: /^Next-ViT/, date: '2022-07', org: 'ByteDance',
  paper: 'https://arxiv.org/abs/2207.05501', code: 'https://github.com/bytedance/Next-ViT', tags: 'next-vit,ncb,ntb,mhca,e-mhsa,hybrid backbone,tensorrt,coreml,deployment,segmentation backbone',
  tagline: 'A hybrid CNN-Transformer backbone designed for real deployment (TensorRT and CoreML). It interleaves fast convolution blocks with efficient transformer blocks so it keeps ResNet-like latency with transformer-level accuracy. It comes in three sizes: Small, Base and Large (there is no "Medium").',
  anatomy: {
    backbone: 'Stem (4 convs, ÷4) → 4 stages. Each stage stacks Next Convolution Blocks (NCB) and ends its pattern with a Next Transformer Block (NTB), following the Next Hybrid Strategy (NHS). S/B/L differ mainly in how many times the stage-3 pattern repeats.',
    neck: 'Classification: none. Segmentation: Semantic FPN or UperNet (PPM + FPN). Detection / instance seg: FPN in Mask R-CNN.',
    head: 'Classification: BN → global avg pool → Linear. Segmentation / detection: the decoder head of the chosen framework.',
    loss: 'Classification CE with DeiT-style recipe; downstream uses the framework losses (CE for Semantic FPN / UperNet; R-CNN losses for Mask R-CNN).'
  },
  pipeline: [IMG('224² (cls) · 512² (ADE20K) · 1333×800 (COCO)'),
    P('backbone', 'Next-ViT backbone', 'S / B / L', ['Stem: 4 × conv (÷4)', 'Stage 1: NCB × N (÷4) → C2', 'Stage 2: NCB × N + NTB (÷8) → C3', 'Stage 3: (NCB × N + NTB) × L (÷16) → C4', 'Stage 4: NCB × N + NTB (÷32) → C5'], 'C2–C5'),
    P('neck', 'Task neck', '', ['Cls: none', 'Seg: Semantic FPN / UperNet', 'Det: FPN (Mask R-CNN)']),
    P('head', 'Task head', '', ['Cls: GAP → Linear 1000', 'Seg: 1×1 conv → 150 classes (ADE20K)', 'Det: RPN + RoI box/mask heads']), P('output', 'Labels / masks / boxes')],
  blocks: [
    { t: 'NCB — Next Convolution Block', k: 'backbone', ops: ['Input x', 'MHCA: group 3×3 conv (32 ch per head) → BN → ReLU → 1×1 projection', 'Residual add', 'MLP: 1×1 conv (expand) → ReLU → 1×1 conv', 'Residual add'], skips: [[0, 2, '+'], [2, 4, '+']], note: 'Multi-Head Convolutional Attention = attention-like token mixing implemented with grouped convolution; fully TensorRT / CoreML friendly (BN + ReLU, no LayerNorm).' },
    { t: 'NTB — Next Transformer Block', k: 'backbone', ops: ['Input x → 1×1 conv (shrink to r·C channels)', 'E-MHSA: self-attention with spatially reduced K, V (avg-pool, ratio s)', 'MHCA on the remaining (1−r)·C channels', 'Concat [global, local] features', 'MLP (+ residual)'], skips: [[0, 3, 'local path'], [3, 4, '+']], note: 'Captures low-frequency (global) and high-frequency (local) information in one block; shrink ratio r = 0.75 in the paper.' },
    { t: 'Next Hybrid Strategy (per stage)', k: 'other', ops: ['NCB', 'NCB', '… (N NCBs)', 'NTB', 'repeat pattern L times (stage 3)'], note: 'Transformer blocks are placed at the end of each pattern, not at the start or in separate stages.' }
  ],
  f: [R`\mathrm{MHCA}(z)=\mathrm{Concat}\big(\mathrm{CA}_1(z_1),\dots,\mathrm{CA}_h(z_h)\big)W^{P},\;\;\mathrm{CA}(z)=\mathrm{ReLU}\big(\mathrm{BN}(\mathrm{GConv}_{3\times3}(z))\big)`,
      R`\text{E-MHSA}: \mathrm{Attn}\big(Q,\;\mathrm{Pool}_s(K),\;\mathrm{Pool}_s(V)\big),\;\;\text{cost }O\!\left(\tfrac{N^2}{s^2}\right)`],
  sections: [
    { h: 'Results (official)', html: T(['Size', 'ImageNet top-1 (224)', 'Params', 'GFLOPs', 'T4 TensorRT ms (bs 8)', 'iPhone CoreML ms'], [['Next-ViT-S', '82.5', '31.7M', '5.8', '7.7', '3.5'], ['Next-ViT-B', '83.2', '44.8M', '8.3', '10.5', '4.5'], ['Next-ViT-L', '83.6', '57.8M', '10.8', '13.0', '5.5']]) +
        T(['Size', 'ADE20K Semantic FPN mIoU', 'ADE20K UperNet mIoU (ss / ms)', 'COCO Mask R-CNN 1× box / mask AP', '3× box / mask AP'], [['S', '46.5', '48.1 / 49.0', '45.9 / 41.8', '48.0 / 43.2'], ['B', '48.6', '50.4 / 51.1', '47.2 / 42.8', '49.5 / 44.4'], ['L', '49.1', '50.1 / 50.8', '48.0 / 43.2', '50.2 / 44.8']]) +
        '<p>With large-scale ImageNet-1K-6M pre-training, top-1 rises to 84.8 / 85.1 / 85.4 at 224. See the Segmentation tab leaderboards <b>Efficient segmentation backbones · ADE20K (Semantic FPN)</b> and <b>Backbones … (Mask R-CNN 1×)</b> for side-by-side comparisons on the same protocol.</p>' },
    { h: 'Using Next-ViT for segmentation', html: '<ol><li>Pick a decoder: <b>Semantic FPN</b> (light, 80k iters) for speed, or <b>UperNet</b> (160k iters) for accuracy.</li><li>Take C2–C5 from the four stages as decoder inputs; initialize from the ImageNet checkpoint.</li><li>Recipe (paper, mmsegmentation): AdamW, lr 1e-4 (Semantic FPN) / 6e-5 (UperNet), weight decay 0.01, poly schedule, crop 512×512, batch 16–32.</li><li>Deploy: export to ONNX → TensorRT FP16. BN + ReLU everywhere means convs fuse cleanly, which gives the latency advantage over LayerNorm/GELU-based ViTs.</li></ol>' },
    { h: 'Why it is fast on real hardware', html: '<ul><li>Avoids LayerNorm, GELU and window shuffles that TensorRT / CoreML handle poorly; uses BN, ReLU and grouped conv.</li><li>Attention only in NTBs with spatial reduction, so the quadratic cost stays small at high resolution.</li><li>Paper claim: Next-ViT-S beats ResNet-101 by 5.5 mAP (COCO) and 7.7 mIoU (ADE20K) at similar TensorRT latency.</li></ul>' }]
});

CV.arch({
  id: 'resnet', name: 'ResNet', task: 'cls', match: /^ResNet/, date: '2015-12', org: 'Microsoft Research',
  paper: 'https://arxiv.org/abs/1512.03385', code: 'https://github.com/pytorch/vision', tags: 'resnet,residual,bottleneck,resnet50,resnet101',
  tagline: 'Residual learning (y = F(x) + x) made 100+ layer networks trainable; still the reference backbone for detection and segmentation.',
  anatomy: { backbone: 'Stem 7×7 s2 conv + 3×3 max-pool (÷4) → 4 stages of bottleneck blocks [3, 4, 6, 3] (R50) / [3, 4, 23, 3] (R101); channels 256 → 2048.', neck: 'Classification: none. ' + DENSE, head: 'GAP → FC 1000.', loss: 'CE (modern recipes: BCE + mixup/cutmix, 600 epochs → 80.4%).' },
  pipeline: [IMG(), P('backbone', 'ResNet-50', '', ['Stem 7×7 s2 + maxpool → 64 × 56²', 'conv2_x: 3 bottlenecks → 256 × 56² (C2)', 'conv3_x: 4 → 512 × 28² (C3)', 'conv4_x: 6 → 1024 × 14² (C4)', 'conv5_x: 3 → 2048 × 7² (C5)'], 'C2–C5'), CLSHEAD, P('output', 'Class probs')],
  blocks: [{ t: 'Bottleneck block', k: 'backbone', ops: ['1×1 conv, C/4 → BN → ReLU', '3×3 conv, C/4 (stride here in v1.5) → BN → ReLU', '1×1 conv, C → BN (γ init 0)', 'Add shortcut → ReLU'], skips: [[0, 3, 'identity / 1×1 proj']] }]
});

CV.arch({
  id: 'vit', name: 'Vision Transformer (ViT)', task: 'cls', match: /^ViT-/, date: '2020-10', org: 'Google Research',
  paper: 'https://arxiv.org/abs/2010.11929', code: 'https://github.com/google-research/vision_transformer', tags: 'vit,patch embedding,cls token,transformer encoder',
  tagline: 'Split the image into 16×16 patches, treat them as tokens, and run a standard transformer encoder. Scales with data better than any CNN.',
  anatomy: { backbone: 'Patch embedding (conv 16×16 s16) + CLS token + learned position embeddings → L pre-norm transformer blocks (B: 12 × 768, L: 24 × 1024).', neck: 'None (single scale ÷16). Dense tasks add a simple pyramid (ViTDet) or ViT-Adapter.', head: 'LayerNorm(CLS) → Linear (pre-training: MLP).', loss: 'CE; pre-trained on IN-21k / JFT, then fine-tuned at higher resolution (PE interpolation).' },
  pipeline: [IMG(), P('backbone', 'ViT encoder', '', ['Patchify 16×16 → 196 tokens × D', 'Prepend CLS, add position embeddings', 'L × (LN → MHSA → + → LN → MLP → +)'], 'CLS token + patch tokens'), P('head', 'Head', '', ['LayerNorm → Linear']), P('output', 'Class probs')],
  blocks: [{ t: 'Transformer block (pre-norm)', k: 'backbone', ops: ['LayerNorm', 'Multi-head self-attention', 'LayerNorm', 'MLP: Linear 4D → GELU → Linear D'], skips: [[0, 1, '+'], [2, 3, '+']] }]
});

CV.arch({ id: 'deit', name: 'DeiT', task: 'cls', match: /^DeiT/, date: '2020-12', org: 'Meta AI', paper: 'https://arxiv.org/abs/2012.12877', code: 'https://github.com/facebookresearch/deit', tags: 'deit,distillation token,data-efficient',
  tagline: 'Same architecture as ViT, trained on ImageNet-1k only using strong augmentation, regularization and a distillation token learning from a CNN teacher.',
  anatomy: { backbone: 'ViT-Ti/S/B (+ distillation token).', neck: 'None.', head: 'Two heads: CLS → labels, DIST → teacher predictions; averaged at test.', loss: 'CE(labels) + CE(hard teacher labels).' },
  pipeline: [IMG(), P('backbone', 'ViT + DIST token'), P('head', 'CLS head + distillation head'), P('output', 'Class probs')] });

CV.arch({
  id: 'swin', name: 'Swin Transformer', task: 'cls', match: /^Swin/, date: '2021-03', org: 'Microsoft Research Asia',
  paper: 'https://arxiv.org/abs/2103.14030', code: 'https://github.com/microsoft/Swin-Transformer', tags: 'swin,shifted window,hierarchical transformer,patch merging',
  tagline: 'Hierarchical transformer with attention inside local windows that shift between layers: linear cost in image size and multi-scale maps like a CNN, which made it the go-to detection and segmentation backbone of 2021–22.',
  anatomy: { backbone: '4×4 patch partition → 4 stages (C, 2C, 4C, 8C; Swin-T C=96, depths 2-2-6-2) with patch merging (2×2 concat → linear) between stages.', neck: DENSE, head: 'GAP → Linear.', loss: 'CE with DeiT-style recipe.' },
  pipeline: [IMG(), P('backbone', 'Swin-T', '', ['Patch partition 4×4 + linear (96)', 'Stage 1: 2 blocks, ÷4', 'Merge → Stage 2: 2 blocks (192), ÷8', 'Merge → Stage 3: 6 blocks (384), ÷16', 'Merge → Stage 4: 2 blocks (768), ÷32'], 'C2–C5'), CLSHEAD, P('output', 'Class probs')],
  blocks: [{ t: 'Two consecutive Swin blocks', k: 'backbone', ops: ['LN → W-MSA (7×7 windows) → + residual', 'LN → MLP → + residual', 'LN → SW-MSA (windows shifted by 3) → + residual', 'LN → MLP → + residual'], note: 'Shifting the windows by half a window every other block lets information cross window borders.' }]
});

CV.arch({
  id: 'convnext', name: 'ConvNeXt (v1 / v2)', task: 'cls', match: /^ConvNeXt/, date: '2022-01', org: 'Meta AI / UC Berkeley',
  paper: 'https://arxiv.org/abs/2201.03545', code: 'https://github.com/facebookresearch/ConvNeXt', tags: 'convnext,convnext v2,grn,fcmae,modern convnet',
  tagline: 'A ResNet modernized step by step with transformer design choices (patchify stem, 7×7 depthwise, inverted bottleneck, LN, GELU) — matches Swin at the same FLOPs with pure convolutions.',
  anatomy: { backbone: 'Stem 4×4 s4 conv + LN → 4 stages [3, 3, 9, 3] (T) with separate downsampling (LN + 2×2 s2 conv); channels 96 → 768 (T).', neck: DENSE, head: 'GAP → LN → Linear.', loss: 'CE; v2 pre-trains with FCMAE (masked autoencoder with sparse convs).' },
  pipeline: [IMG(), P('backbone', 'ConvNeXt-T', '', ['Stem 4×4 s4 (96)', 'Stage 1 ×3 (96)', 'Stage 2 ×3 (192)', 'Stage 3 ×9 (384)', 'Stage 4 ×3 (768)'], 'C2–C5'), CLSHEAD, P('output', 'Class probs')],
  blocks: [{ t: 'ConvNeXt block', k: 'backbone', ops: ['DW conv 7×7', 'LayerNorm', '1×1 conv (4C) → GELU', '(v2: GRN)', '1×1 conv (C) → LayerScale → DropPath'], skips: [[0, 4, '+']] }]
});

CV.arch({ id: 'efficientnet', name: 'EfficientNet / EfficientNetV2', task: 'cls', match: /^EfficientNet/, date: '2019-05', org: 'Google Brain', paper: 'https://arxiv.org/abs/1905.11946', tags: 'efficientnet,compound scaling,mbconv,fused-mbconv',
  tagline: 'NAS-designed MBConv network scaled jointly in depth, width and resolution (compound scaling). V2 adds Fused-MBConv and progressive learning.',
  anatomy: { backbone: 'B0: 3×3 stem → 16 MBConv blocks (k3/k5, expansion 6, SE) in 7 stages → 1×1 conv 1280.', neck: DENSE + ' (EfficientDet uses BiFPN.)', head: 'GAP → dropout → Linear.', loss: 'CE + label smoothing, RMSProp, AutoAugment, stochastic depth.' },
  pipeline: [IMG('224 (B0) … 600 (B7)'), P('backbone', 'MBConv stages', '', ['Stem 3×3 s2 (32)', 'MBConv1 k3 (16)', 'MBConv6 k3/k5 ×15', 'Conv 1×1 (1280)']), CLSHEAD, P('output', 'Class probs')],
  blocks: [{ t: 'MBConv (inverted residual + SE)', k: 'backbone', ops: ['1×1 expand ×6 → BN → SiLU', 'DW k×k → BN → SiLU', 'Squeeze-Excite (0.25)', '1×1 project → BN (linear)'], skips: [[0, 3, '+ if same shape']] }] });

CV.arch({ id: 'mobilenet', name: 'MobileNet v2 / v3 / v4', task: 'cls', match: /^MobileNet/, date: '2018-01', org: 'Google', paper: 'https://arxiv.org/abs/2404.10518', tags: 'mobilenet,inverted residual,uib,edge,mobile',
  tagline: 'Mobile backbones built on depthwise-separable convs: v2 inverted residuals, v3 NAS + SE + h-swish, v4 (2024) Universal Inverted Bottleneck and Mobile MQA attention that is Pareto-optimal across CPUs, GPUs, DSPs and EdgeTPUs.',
  anatomy: { backbone: 'Stacked inverted-residual / UIB blocks; v4 hybrids add Mobile MQA attention in late stages.', neck: 'For detection: SSDLite / FPN-lite; for segmentation: LR-ASPP.', head: 'GAP → 1×1 conv (1280) → Linear.', loss: 'CE.' },
  pipeline: [IMG(), P('backbone', 'Inverted-residual / UIB stages'), CLSHEAD, P('output', 'Class probs')],
  blocks: [{ t: 'Universal Inverted Bottleneck (v4)', k: 'backbone', ops: ['Optional DW conv (before expansion)', '1×1 expand', 'Optional DW conv (after expansion)', '1×1 project'], skips: [[0, 3, '+']], note: 'Switching the two optional DWs gives IB (MobileNetV2), ConvNeXt-like, ExtraDW and FFN variants — NAS picks per layer.' }] });

CV.arch({ id: 'densenet', name: 'DenseNet', task: 'cls', match: /^DenseNet/, date: '2016-08', org: 'Cornell / Tsinghua', paper: 'https://arxiv.org/abs/1608.06993', tags: 'densenet,dense connectivity,growth rate',
  tagline: 'Each layer receives the concatenated outputs of all previous layers in its block (growth rate k = 32): strong feature reuse, few parameters.',
  anatomy: { backbone: '4 dense blocks [6, 12, 24, 16] (121) of BN-ReLU-1×1-BN-ReLU-3×3 layers, joined by transition layers (1×1 conv + 2×2 avg-pool, compression 0.5).', neck: 'None.', head: 'GAP → Linear.', loss: 'CE.' },
  pipeline: [IMG(), P('backbone', 'Dense blocks + transitions'), CLSHEAD, P('output', 'Class probs')] });

CV.arch({ id: 'vgg', name: 'VGG', task: 'cls', match: /^VGG/, date: '2014-09', org: 'Oxford VGG', paper: 'https://arxiv.org/abs/1409.1556', tags: 'vgg,3x3 convs,deep plain network',
  tagline: 'Showed depth matters: only 3×3 convs and 2×2 max-pools, 16–19 layers. Large (138M) because of the three FC layers.',
  anatomy: { backbone: '5 blocks of 2–3 × (3×3 conv + ReLU), max-pool after each; channels 64 → 512.', neck: 'None.', head: 'FC 4096 → FC 4096 → FC 1000 (with dropout).', loss: 'CE.' },
  pipeline: [IMG(), P('backbone', '13 conv layers', '', ['64 ×2, pool', '128 ×2, pool', '256 ×3, pool', '512 ×3, pool', '512 ×3, pool']), P('head', '3 FC layers'), P('output', 'Class probs')] });

CV.arch({ id: 'alexnet', name: 'AlexNet', task: 'cls', match: /^AlexNet/, date: '2012-09', org: 'Univ. of Toronto', paper: 'https://papers.nips.cc/paper/4824', tags: 'alexnet,imagenet 2012,relu,dropout',
  tagline: 'The 2012 ImageNet winner that started the deep-learning era: 5 conv + 3 FC layers, ReLU, dropout, data augmentation, trained on two GPUs.',
  anatomy: { backbone: 'Conv 11×11 s4 (96) → pool → 5×5 (256) → pool → 3×3 (384) → 3×3 (384) → 3×3 (256) → pool.', neck: 'None.', head: 'FC 4096 → FC 4096 → FC 1000.', loss: 'CE, SGD momentum 0.9, LR 0.01 ÷10 on plateau.' },
  pipeline: [IMG('227×227×3'), P('backbone', '5 conv layers'), P('head', '3 FC layers'), P('output', 'Class probs')] });

CV.arch({ id: 'coatnet', name: 'CoAtNet', task: 'cls', match: /^CoAtNet/, date: '2021-06', org: 'Google Brain', paper: 'https://arxiv.org/abs/2106.04803', tags: 'coatnet,hybrid,relative attention',
  tagline: 'Vertical hybrid: MBConv stages at high resolution, relative-attention transformer stages at low resolution (C-C-T-T layout).',
  anatomy: { backbone: 'S0 stem → S1, S2: MBConv → S3, S4: transformer blocks with relative attention.', neck: 'None.', head: 'GAP → Linear.', loss: 'CE; JFT-3B pre-training for 90.9%.' },
  pipeline: [IMG(), P('backbone', 'C-C-T-T stages'), CLSHEAD, P('output', 'Class probs')] });

CV.arch({ id: 'eva02', name: 'EVA-02', task: 'cls', match: /^EVA-02/, date: '2023-03', org: 'BAAI', paper: 'https://arxiv.org/abs/2303.11331', tags: 'eva-02,masked image modeling,clip teacher,swiglu,rope',
  tagline: 'Plain ViT with SwiGLU, 2-D RoPE and sub-LN, pre-trained by masked image modeling to reconstruct EVA-CLIP features — top ImageNet accuracy (90.0) at 304M params.',
  anatomy: { backbone: 'ViT-L/14 with SwiGLU FFN, 2-D RoPE, sub-LN.', neck: 'None (ViTDet-style pyramid for detection).', head: 'Linear.', loss: 'MIM (cosine to CLIP features) then CE fine-tuning.' },
  pipeline: [IMG('448²'), P('backbone', 'EVA-02 ViT-L'), P('head', 'Linear'), P('output', 'Class probs')] });

CV.arch({ id: 'beit3', name: 'BEiT-3', task: 'cls', match: /BEiT-3/, date: '2022-08', org: 'Microsoft', paper: 'https://arxiv.org/abs/2208.10442', tags: 'beit-3,multiway transformer,masked data modeling',
  tagline: 'Multiway transformer (shared attention, modality-specific FFN experts) pre-trained with masked "language" modeling on images, text and image-text pairs.',
  anatomy: { backbone: '1.9B multiway transformer (vision, language, vision-language experts).', neck: 'ViT-Adapter for dense tasks.', head: 'Task heads (Linear / Mask2Former / ...).', loss: 'Masked data modeling, then task fine-tuning.' },
  pipeline: [IMG(), P('backbone', 'Multiway transformer'), P('head', 'Task head'), P('output', 'Prediction')] });

CV.arch({ id: 'dinov2', name: 'DINOv2 / DINOv3', task: 'cls', match: /^DINOv[23]/, date: '2023-04', org: 'Meta AI', paper: 'https://arxiv.org/abs/2304.07193', code: 'https://github.com/facebookresearch/dinov3', tags: 'dinov2,dinov3,self-supervised,foundation,frozen features,gram anchoring',
  tagline: 'Self-supervised ViT foundation models whose frozen features work out of the box for classification, segmentation, depth and retrieval. DINOv3 (2025-08) scales to a 7B teacher with Gram anchoring to keep dense features clean.',
  anatomy: { backbone: 'ViT-S/B/L/g (DINOv2, patch 14, registers) · ViT up to 7B + distilled smaller ViTs/ConvNeXts (DINOv3, patch 16).', neck: 'Frozen features → linear probe / ViT-Adapter / DPT (depth) / Mask2Former.', head: 'Linear classifier on CLS (+ avg patch tokens) for the leaderboard numbers.', loss: 'Self-distillation: DINO (CLS) + iBOT (masked patches) + KoLeo; DINOv3 adds Gram anchoring.' },
  pipeline: [IMG(), P('backbone', 'Self-supervised ViT', 'frozen', ['Patch tokens + CLS (+ register tokens)']), P('head', 'Linear probe', '', ['Linear on [CLS, mean(patch)]']), P('output', 'Class probs')],
  f: [R`\mathcal L=-\sum p_t\log p_s\;(\text{DINO})+\mathcal L_{iBOT}+\lambda\,\mathcal L_{KoLeo},\;\;\mathcal L_{Gram}=\big\|X_SX_S^{\top}-X_GX_G^{\top}\big\|_F^2`] });

CV.arch({ id: 'vmamba', name: 'VMamba', task: 'cls', match: /^VMamba/, date: '2024-01', org: 'UCAS / Huawei', paper: 'https://arxiv.org/abs/2401.10166', tags: 'vmamba,state space model,ss2d,cross-scan',
  tagline: 'Hierarchical state-space backbone: 2-D selective scan (SS2D) scans the image along four directions to give each token a global receptive field in linear time.',
  anatomy: { backbone: 'Stem ÷4 → 4 stages of VSS blocks (SS2D + FFN) with downsampling.', neck: DENSE, head: 'GAP → Linear.', loss: 'CE.' },
  pipeline: [IMG(), P('backbone', 'VSS stages', 'SS2D cross-scan'), CLSHEAD, P('output', 'Class probs')] });

CV.arch({ id: 'internimage', name: 'InternImage', task: 'cls', match: /^InternImage/, date: '2022-11', org: 'Shanghai AI Lab', paper: 'https://arxiv.org/abs/2211.05778', code: 'https://github.com/OpenGVLab/InternImage', tags: 'internimage,dcnv3,deformable convolution,large-scale cnn',
  tagline: 'Large-scale CNN built on DCNv3 (deformable conv with shared weights across groups and softmax modulation): a CNN that scales to 1B+ parameters and led COCO/ADE20K in 2022–23.',
  anatomy: { backbone: 'Stem ÷4 → 4 stages of DCNv3 blocks (LN, FFN, layer scale).', neck: 'FPN / DINO encoder (det), UperNet / Mask2Former (seg).', head: 'Task heads.', loss: 'CE / task losses.' },
  pipeline: [IMG(), P('backbone', 'DCNv3 stages'), P('head', 'Task head'), P('output', 'Prediction')],
  f: [R`y(p_0)=\sum_{g=1}^{G}\sum_{k=1}^{K}w_g\,m_{gk}\,x_g(p_0+p_k+\Delta p_{gk}),\;\;\textstyle\sum_k m_{gk}=1`] });

// Steganalysis (binary cover-vs-stego image classification)
CV.arch({
  id: 'hsmnet', name: 'HSMNet (steganalysis)', aka: 'hsm net hsmnet hsm-net steganalysis', task: 'cls', match: /^HSMNet/i, date: '2025', org: 'Information Sciences (Elsevier), 2025',
  paper: 'https://www.sciencedirect.com/science/article/abs/pii/S0020025525009600', tags: 'hsmnet,steganalysis,hybrid dilated convolution,squeeze-excitation,se attention,multi-resolution,cover source mismatch,stego detection,binary classification',
  tagline: 'Multi-resolution grayscale image steganalysis network. It decides whether an image is a clean cover or a stego image hiding a message, and it works across image resolutions. Two branches, local (vanilla conv) and global (hybrid dilated conv), use squeeze-and-excitation channel attention and are fused hierarchically.',
  anatomy: {
    backbone: 'Preprocessing module with a hybrid dilated convolution block that enlarges the receptive field, plus SE channel attention; then a two-branch feature extractor. One branch uses vanilla convolutions for local (fine texture / noise-residual) features; the other uses hybrid dilated convolutions for global features of multi-resolution images.',
    neck: 'Hierarchical fusion strategy that combines the local and global branch features; SE blocks in the preprocessing and feature-extraction stages re-weight channels toward texture-rich regions, where adaptive steganography embeds most of the payload.',
    head: 'Classification module: global pooling over the fused features → fully-connected layers → 2 classes (cover / stego). Global pooling is what lets one network accept images of different resolutions.',
    loss: 'Binary classification (cover vs stego), evaluated by detection accuracy / error rate per steganographic algorithm and payload.'
  },
  pipeline: [
    { k: 'input', t: 'Grayscale image', d: 'any resolution (multi-resolution), cover or stego' },
    { k: 'backbone', t: 'Preprocessing module', d: 'per the paper', b: ['Hybrid dilated convolution block (larger receptive field)', 'SE channel attention'] },
    { k: 'backbone', t: 'Two-branch feature extraction', b: ['Local branch: vanilla convolutions → local features', 'Global branch: hybrid dilated convolutions → global features', 'SE blocks in both stages'] },
    { k: 'neck', t: 'Hierarchical fusion', d: 'combine local + global representations' },
    { k: 'head', t: 'Classifier', b: ['Global pooling (resolution independent)', 'FC layers → 2 logits', 'cover / stego'] },
    { k: 'output', t: 'Decision', d: 'P(stego)' }],
  blocks: [
    { t: 'Hybrid dilated convolution (concept)', k: 'backbone', ops: ['3×3 conv, dilation r₁ (e.g. 1)', '3×3 conv, dilation r₂ (e.g. 2)', '3×3 conv, dilation r₃ (e.g. 5)', 'Receptive field grows fast, no gridding holes'], note: 'Rates are chosen without a common factor (e.g. 1-2-5) so that stacked dilated kernels cover every pixel. The illustrative rates are the standard HDC choice; the paper\'s exact rates are in the full text.' },
    { t: 'Squeeze-and-Excitation block', k: 'backbone', ops: ['Feature map C×H×W', 'Global average pool → C', 'FC (C/r) → ReLU → FC (C) → sigmoid', 'Scale channels of the input'], skips: [[0, 3, 'x']] }],
  f: [R`\text{HDC receptive field: } r_{eff}=1+\sum_i (k-1)\,d_i\;\;(k=3,\;d=1,2,5\Rightarrow 17)`, R`\text{SE: } s=\sigma\big(W_2\,\delta(W_1\,\mathrm{GAP}(x))\big),\;\;y=s\odot x`, R`P_E=\min_{P_{FA}}\tfrac12\big(P_{FA}+P_{MD}\big)\;\;(\text{steganalysis detection error})`],
  sections: [
    { h: 'What problem it solves', html: '<p>Most CNN steganalyzers (Xu-Net, Ye-Net, Yedroudj-Net, SRNet, Zhu-Net) are trained and tested on fixed-size images, typically 256×256 or 512×512 from BOSSbase. Real images come at many resolutions, which causes <b>cover-source mismatch</b> and an accuracy drop. HSMNet targets <b>multi-resolution</b> grayscale steganalysis: the global dilated branch captures image-wide statistics, and the local branch keeps the fine noise-residual cues that embedding leaves behind.</p>' },
    { h: 'Evaluation (as reported)', html: '<p>The authors evaluate on <b>BOSSbase</b> and <b>ALASKA #2</b> and report better detection than current multi-resolution steganalysis methods. Per-algorithm accuracy tables, payloads and the exact layer configuration are in the full paper (paywalled), so they are not reproduced here. This app only lists numbers read from an official source.</p>' },
    { h: 'Practical notes for training steganalyzers', html: '<ul><li><b>Never resize or JPEG-recompress</b> the inputs; interpolation destroys the embedding signal. Use crops or resolution-agnostic pooling instead, which is the point of HSMNet\'s design.</li><li>Train cover and its matching stego image in the <b>same mini-batch</b> (pair constraint); this is standard practice and speeds convergence.</li><li>Augment only with label-preserving, signal-preserving ops: flips and 90° rotations. Avoid color or blur augmentation.</li><li>Curriculum on payload: start at a high payload (e.g. 0.4 bpp), then fine-tune to lower payloads (0.2, 0.1 bpp).</li><li>Report detection error P<sub>E</sub> or accuracy per embedding algorithm (WOW, S-UNIWARD, HILL, …) and per payload. One averaged number hides the hard cases.</li></ul>' }]
});

// Efficient backbones compared in the Next-ViT tables
CV.arch({ id: 'pvtv2', name: 'PVT v2', task: 'cls', match: /^PVTv2/, date: '2021-06', org: 'Nanjing Univ. / HKU', paper: 'https://arxiv.org/abs/2106.13797', tags: 'pvt,pyramid vision transformer,spatial reduction attention',
  tagline: 'Pyramid Vision Transformer: four-stage hierarchical ViT with spatial-reduction attention; v2 adds overlapping patch embedding, conv FFN and linear SRA.',
  anatomy: { backbone: 'Overlapping patch embed → 4 stages (÷4 … ÷32), SRA reduces K/V resolution.', neck: DENSE, head: 'GAP → Linear.', loss: 'CE.' }, pipeline: [IMG(), P('backbone', 'PVT v2 stages'), CLSHEAD, P('output', 'Class probs')] });
CV.arch({ id: 'twins', name: 'Twins (SVT)', task: 'cls', match: /^Twins/, date: '2021-04', org: 'Meituan', paper: 'https://arxiv.org/abs/2104.13840', tags: 'twins,spatially separable attention',
  tagline: 'Alternates locally-grouped self-attention (windows) and global sub-sampled attention in each stage.',
  anatomy: { backbone: '4 stages with LSA + GSA pairs, conditional position encoding.', neck: DENSE, head: 'GAP → Linear.', loss: 'CE.' }, pipeline: [IMG(), P('backbone', 'LSA / GSA stages'), CLSHEAD, P('output', 'Class probs')] });
CV.arch({ id: 'poolformer', name: 'PoolFormer (MetaFormer)', task: 'cls', match: /^PoolFormer/, date: '2021-11', org: 'Sea AI Lab', paper: 'https://arxiv.org/abs/2111.11418', tags: 'poolformer,metaformer,pooling token mixer',
  tagline: 'Replaces attention with simple average pooling as the token mixer to show the MetaFormer structure (norm → mixer → norm → MLP) matters most.',
  anatomy: { backbone: '4 stages of MetaFormer blocks with pooling token mixer.', neck: DENSE, head: 'GAP → Linear.', loss: 'CE.' }, pipeline: [IMG(), P('backbone', 'MetaFormer stages', 'pool mixer'), CLSHEAD, P('output', 'Class probs')] });
CV.arch({ id: 'efficientformer', name: 'EfficientFormer', task: 'cls', match: /^EfficientFormer/, date: '2022-06', org: 'Snap / Northeastern', paper: 'https://arxiv.org/abs/2206.01191', tags: 'efficientformer,mobile vit,latency-driven',
  tagline: 'Latency-driven hybrid for phones: 4-D (conv, pooling mixer) blocks in early stages and 3-D (attention) blocks only in the last stage.',
  anatomy: { backbone: 'Conv stem → MB4D blocks → MB3D (MHSA) in last stage.', neck: DENSE, head: 'GAP → Linear (+ distillation head).', loss: 'CE + distillation.' }, pipeline: [IMG(), P('backbone', 'MB4D + MB3D'), CLSHEAD, P('output', 'Class probs')] });
CV.arch({ id: 'trtvit', name: 'TRT-ViT', task: 'cls', match: /^TRT-ViT/, date: '2022-05', org: 'ByteDance', paper: 'https://arxiv.org/abs/2205.09579', tags: 'trt-vit,tensorrt oriented,hybrid',
  tagline: 'TensorRT-oriented hybrid (precursor to Next-ViT from the same team): design rules chosen by measured TensorRT latency rather than FLOPs.',
  anatomy: { backbone: 'Conv stages early, mixed conv-transformer blocks late.', neck: DENSE, head: 'GAP → Linear.', loss: 'CE.' }, pipeline: [IMG(), P('backbone', 'Conv + transformer stages'), CLSHEAD, P('output', 'Class probs')] });
CV.arch({ id: 'uniformer', name: 'UniFormer', task: 'cls', match: /^UniFormer/, date: '2022-01', org: 'CAS / Shanghai AI Lab', paper: 'https://arxiv.org/abs/2201.09450', tags: 'uniformer,local global relation aggregator',
  tagline: 'Unifies convolution and attention as "relation aggregators": local (conv-like) MHRA in shallow stages, global (attention) MHRA in deep stages.',
  anatomy: { backbone: 'Stages 1–2 local MHRA (DW conv), stages 3–4 global MHRA (self-attention), dynamic position embedding.', neck: DENSE, head: 'GAP → Linear.', loss: 'CE.' }, pipeline: [IMG(), P('backbone', 'Local → global MHRA'), CLSHEAD, P('output', 'Class probs')] });
})();

// Segmentation architecture pages (decoders / full models). Backbones used by segmenters live in arch_cls.js.
(function () {
const R = String.raw, T = CV.tbl;
const P = (k, t, d, b, o) => ({ k, t, d, b, o });

CV.arch({
  id: 'segformer', name: 'SegFormer', task: 'seg', match: /^SegFormer/, date: '2021-05', org: 'NVIDIA / HKU',
  paper: 'https://arxiv.org/abs/2105.15203', code: 'https://github.com/NVlabs/SegFormer', tags: 'segformer,mix transformer,mit,all-mlp decoder,efficient attention',
  tagline: 'Hierarchical Mix Transformer encoder with no positional encoding and a tiny all-MLP decoder. Simple, robust to resolution changes, and efficient from B0 (3.8M) to B5 (85M).',
  anatomy: { backbone: 'MiT-B0…B5: 4 stages with overlapping patch embeddings (7×7 s4, then 3×3 s2), efficient self-attention (sequence reduction R = 8/4/2/1), Mix-FFN (3×3 DW conv inside FFN).', neck: 'All-MLP decoder: per-stage linear to C, upsample all to ¼, concat, linear fuse.', head: 'Linear → K classes at ¼ resolution → bilinear ×4.', loss: 'Pixel CE.' },
  pipeline: [P('input', 'Image', '512×512 (ADE) / 1024×1024 (Cityscapes)'), P('backbone', 'Mix Transformer (MiT)', '', ['Stage 1: OPE 7×7 s4 → ÷4', 'Stage 2: OPE 3×3 s2 → ÷8', 'Stage 3 → ÷16', 'Stage 4 → ÷32', 'Each: N × (Efficient attn + Mix-FFN)'], 'F1–F4'),
    P('neck', 'All-MLP decoder', '', ['Linear C_i → C (256 / 768)', 'Upsample to ÷4, concat', 'Linear fuse 4C → C']), P('head', 'Classifier', '', ['Linear C → K', 'Upsample ×4']), P('output', 'Class map', 'H×W')],
  blocks: [{ t: 'MiT block', k: 'backbone', ops: ['LayerNorm', 'Efficient self-attention (K, V reduced by R)', 'LayerNorm', 'Mix-FFN: Linear → DWConv 3×3 → GELU → Linear'], skips: [[0, 1, '+'], [2, 3, '+']] }],
  f: [R`\hat K=\mathrm{Linear}(C\cdot R,C)\big(\mathrm{Reshape}(\tfrac{N}{R},C\cdot R)(K)\big),\;\;O\big(\tfrac{N^2}{R}\big)`],
  sections: [{ h: 'Variants', html: T(['Model', 'Channels (4 stages)', 'Depths', 'Params', 'ADE20K mIoU'], [['B0', '32, 64, 160, 256', '2, 2, 2, 2', '3.8M', '37.4'], ['B1', '64, 128, 320, 512', '2, 2, 2, 2', '13.7M', '42.2'], ['B2', '64, 128, 320, 512', '3, 4, 6, 3', '27.5M', '46.5'], ['B3', '64, 128, 320, 512', '3, 4, 18, 3', '47.3M', '49.4'], ['B5', '64, 128, 320, 512', '3, 6, 40, 3', '84.7M', '51.0']]) },
    { h: 'Training recipe', html: '<p>AdamW lr 6e-5 (head ×10), wd 0.01, poly schedule (power 1.0), 160k iterations, batch 16, random resize 0.5–2.0, crop 512, flip. ImageNet-1k pre-trained encoder.</p>' }]
});

CV.arch({
  id: 'mask2former', name: 'Mask2Former / MaskFormer', task: 'seg', match: /^Mask2?Former/, date: '2021-12', org: 'Meta AI (FAIR) / UIUC',
  paper: 'https://arxiv.org/abs/2112.01527', code: 'https://github.com/facebookresearch/Mask2Former', tags: 'mask2former,maskformer,masked attention,universal segmentation,mask classification,panoptic',
  tagline: 'One architecture for semantic, instance and panoptic segmentation: predict N (class, mask) pairs. Masked cross-attention restricts each query to its own predicted region.',
  anatomy: { backbone: 'ResNet / Swin / ConvNeXt / ViT (with adapter) → C2–C5.', neck: 'Pixel decoder: 6-layer multi-scale deformable attention (MSDeformAttn) over ÷8, ÷16, ÷32, then FPN-style upsample to ÷4 per-pixel embeddings.', head: 'Transformer decoder: 9 layers (3 rounds over ÷32, ÷16, ÷8 features), 100–200 queries; order per layer: masked cross-attention → self-attention → FFN. Each query → class logits + mask embedding · per-pixel embeddings.', loss: 'Hungarian matching; CE (2.0, no-object 0.1) + mask BCE (5.0) + Dice (5.0) computed on 12,544 importance-sampled points.' },
  pipeline: [P('input', 'Image', 'LSJ 1024² (COCO) / 512² (ADE)'), P('backbone', 'Backbone', 'Swin-L / R50 / ConvNeXt', [], 'C2–C5'),
    P('neck', 'Pixel decoder', 'MSDeformAttn ×6', ['Multi-scale features ÷8/16/32 → transformer decoder', 'Per-pixel embeddings at ÷4']),
    P('decoder', 'Transformer decoder', '9 layers, N queries', ['Masked cross-attention (mask from previous layer)', 'Self-attention', 'FFN', 'Round-robin over scales']),
    P('head', 'Mask classification', '', ['Class: Linear → K+1', 'Mask: MLP(query) · pixel embeddings → sigmoid']), P('output', 'N masks + classes', 'semantic / instance / panoptic inference')],
  blocks: [{ t: 'Mask2Former decoder layer', k: 'head', ops: ['Queries X_{l-1}', 'Masked cross-attention: attend only where M_{l-1} > 0.5', 'Self-attention among queries', 'FFN', 'Predict class + mask M_l (fed to next layer)'], skips: [[0, 1, '+']] }],
  f: [R`X_l=\mathrm{softmax}(\mathcal M_{l-1}+Q_lK_l^{\top})V_l+X_{l-1}`, R`\text{semantic inference: } \arg\max_c\sum_{i}p_i(c)\,m_i[h,w]`]
});

CV.arch({
  id: 'oneformer', name: 'OneFormer', task: 'seg', match: /^OneFormer/, date: '2022-11', org: 'SHI Labs / Picsart / UIUC',
  paper: 'https://arxiv.org/abs/2211.06220', code: 'https://github.com/SHI-Labs/OneFormer', tags: 'oneformer,task token,universal segmentation,multi-task',
  tagline: 'Mask2Former trained once for all three tasks: a text "task token" (semantic / instance / panoptic) conditions the queries, and a query-text contrastive loss separates tasks.',
  anatomy: { backbone: 'Swin-L / ConvNeXt / DiNAT-L.', neck: 'MSDeformAttn pixel decoder.', head: 'Task-conditioned queries → Mask2Former-style decoder.', loss: 'Mask2Former losses + query-text contrastive loss.' },
  pipeline: [P('input', 'Image + task token', '"the task is panoptic"'), P('backbone', 'DiNAT / Swin / ConvNeXt'), P('neck', 'Pixel decoder'), P('decoder', 'Task-conditioned decoder'), P('output', 'Masks for the chosen task')]
});

CV.arch({
  id: 'upernet', name: 'UperNet (decoder)', task: 'seg', match: null, date: '2018-07', org: 'Peking Univ. / MIT',
  paper: 'https://arxiv.org/abs/1807.10221', tags: 'upernet,ppm,fpn,segmentation decoder,backbone benchmark',
  tagline: 'The standard decoder for comparing backbones on ADE20K (Swin, ConvNeXt, Next-ViT all report "+ UperNet"): a Pyramid Pooling Module on C5 plus an FPN, fused at ¼ resolution.',
  anatomy: { backbone: 'Any hierarchical backbone (C2–C5).', neck: 'PPM (bins 1, 2, 3, 6) on C5 + FPN top-down path; all levels upsampled to ÷4 and concatenated.', head: '3×3 conv → 1×1 conv to K; auxiliary FCN head on C4 (weight 0.4).', loss: 'CE (+ 0.4 × auxiliary CE).' },
  pipeline: [P('input', 'Image', '512×512'), P('backbone', 'Backbone', 'Swin / ConvNeXt / Next-ViT', [], 'C2–C5'), P('neck', 'PPM + FPN', '', ['PPM on C5: pool 1/2/3/6 → concat', 'FPN top-down to C2', 'Upsample all to ÷4, concat']), P('head', 'Fuse + classify', '', ['3×3 conv 512', '1×1 conv → K']), P('output', 'Class map')],
  sections: [{ h: 'Standard protocol', html: '<p>ADE20K, 160k iterations, batch 16, AdamW (transformers) or SGD, crop 512, report single-scale and multi-scale + flip mIoU. Params and FLOPs are dominated by the UperNet head (~1 TFLOP at 512×2048), which is why "backbone + UperNet" FLOPs look huge.</p>' }]
});

CV.arch({
  id: 'semantic-fpn', name: 'Semantic FPN (decoder)', task: 'seg', match: null, date: '2019-01', org: 'FAIR (Panoptic FPN)',
  paper: 'https://arxiv.org/abs/1901.02446', tags: 'semantic fpn,panoptic fpn,lightweight decoder,backbone benchmark',
  tagline: 'Lightweight decoder used to benchmark efficient backbones (PVT, PoolFormer, Next-ViT): FPN levels are upsampled with conv + 2× steps to ¼ resolution and summed.',
  anatomy: { backbone: 'Any hierarchical backbone.', neck: 'FPN P2–P5 (256 ch).', head: 'Each level: repeated (3×3 conv 128 + GN + ReLU + 2× upsample) until ÷4; sum; 1×1 conv to K; ×4 upsample.', loss: 'CE.' },
  pipeline: [P('input', 'Image', '512×512'), P('backbone', 'Backbone', '', [], 'C2–C5'), P('neck', 'FPN', '', ['P2–P5, 256 ch']), P('head', 'Semantic head', '', ['Per level: conv + upsample to ÷4', 'Element-wise sum', '1×1 conv → K']), P('output', 'Class map')],
  sections: [{ h: 'Protocol used in backbone papers', html: '<p>80k iterations, batch 32 (PVT protocol) or 16, AdamW 2e-4 (1e-4 in some papers), poly schedule. Because the decoder is cheap, differences mostly reflect the backbone.</p>' }]
});

CV.arch({
  id: 'deeplabv3p', name: 'DeepLabv3+', task: 'seg', match: /^DeepLabV3/i, date: '2018-02', org: 'Google',
  paper: 'https://arxiv.org/abs/1802.02611', code: 'https://github.com/open-mmlab/mmsegmentation', tags: 'deeplabv3+,aspp,atrous,encoder decoder',
  tagline: 'Atrous convolution keeps resolution (output stride 16/8), ASPP gathers multi-scale context, and a light decoder recovers sharp boundaries.',
  anatomy: { backbone: 'ResNet-101 / Xception with atrous convs in the last stage(s).', neck: 'ASPP: 1×1, 3×3 (rates 6/12/18 at OS16), image pooling → concat → 1×1 (256).', head: 'Decoder: upsample ×4, concat low-level C2 (reduced to 48 ch), 2× 3×3 conv, 1×1 to K, upsample ×4.', loss: 'CE (OHEM optional).' },
  pipeline: [P('input', 'Image'), P('backbone', 'ResNet-101 (atrous)', 'output stride 16', [], 'C2 + C5'), P('neck', 'ASPP', '', ['1×1 conv', '3×3 rate 6 / 12 / 18', 'Global avg pool branch', 'Concat → 1×1']), P('head', 'Decoder', '', ['Upsample ×4 + C2 (48 ch)', '2 × 3×3 conv', '1×1 → K, upsample ×4']), P('output', 'Class map')]
});

CV.arch({
  id: 'pspnet', name: 'PSPNet', task: 'seg', match: /^PSPNet/, date: '2016-12', org: 'CUHK / SenseTime',
  paper: 'https://arxiv.org/abs/1612.01105', tags: 'pspnet,pyramid pooling module,global context',
  tagline: 'Pyramid Pooling Module adds global scene context to dilated-ResNet features; won ImageNet scene parsing 2016.',
  anatomy: { backbone: 'Dilated ResNet (output stride 8).', neck: 'PPM: adaptive avg-pool to 1×1, 2×2, 3×3, 6×6 → 1×1 conv (C/4) → upsample → concat with input.', head: '3×3 conv → 1×1 conv to K; auxiliary loss on C4 (0.4).', loss: 'CE.' },
  pipeline: [P('input', 'Image'), P('backbone', 'Dilated ResNet', 'OS 8'), P('neck', 'Pyramid Pooling Module', '', ['Bins 1, 2, 3, 6', 'Concat with features']), P('head', 'Conv classifier'), P('output', 'Class map')]
});

CV.arch({
  id: 'segnext', name: 'SegNeXt', task: 'seg', match: /^SegNeXt/, date: '2022-09', org: 'Tsinghua / Nankai',
  paper: 'https://arxiv.org/abs/2209.08575', code: 'https://github.com/Visual-Attention-Network/SegNeXt', tags: 'segnext,mscan,convolutional attention,hamburger decoder',
  tagline: 'Convolutional attention beats self-attention for segmentation: the MSCAN encoder uses multi-scale strip convolutions as attention, with a light Hamburger (matrix decomposition) decoder.',
  anatomy: { backbone: 'MSCAN: 4 stages; attention = 5×5 DW conv + parallel strip convs (1×7/7×1, 1×11/11×1, 1×21/21×1) → 1×1 → multiply with input.', neck: 'Features from stages 2–4.', head: 'Light Hamburger decoder (NMF-based global context).', loss: 'CE.' },
  pipeline: [P('input', 'Image'), P('backbone', 'MSCAN', 'multi-scale conv attention'), P('neck', 'Stage 2–4 aggregation'), P('head', 'Hamburger decoder'), P('output', 'Class map')],
  blocks: [{ t: 'MSCA attention', k: 'backbone', ops: ['DW conv 5×5', 'Strip convs 7 / 11 / 21 (DW, parallel)', 'Sum branches → 1×1 conv = attention map', 'Attention ⊙ input'], skips: [[0, 3, 'input']] }]
});

CV.arch({
  id: 'one-peace', name: 'ONE-PEACE', task: 'seg', match: /^ONE-PEACE/, date: '2023-05', org: 'Alibaba DAMO',
  paper: 'https://arxiv.org/abs/2305.11172', code: 'https://github.com/OFA-Sys/ONE-PEACE', tags: 'one-peace,multimodal foundation,vision language audio',
  tagline: '4B-parameter general representation model for vision, audio and language (shared self-attention, modality-specific adapters/FFNs). As a segmentation backbone with ViT-Adapter + Mask2Former it reached 63.0 ADE20K mIoU.',
  anatomy: { backbone: 'ONE-PEACE ViT-style encoder (~1.5B vision branch).', neck: 'ViT-Adapter multi-scale features.', head: 'Mask2Former.', loss: 'Mask2Former losses (fine-tuning).' },
  pipeline: [P('input', 'Image'), P('backbone', 'ONE-PEACE encoder'), P('neck', 'ViT-Adapter'), P('head', 'Mask2Former'), P('output', 'Class map')]
});

CV.arch({
  id: 'sam', name: 'Segment Anything (SAM → SAM 2 → SAM 3)', task: 'seg', match: null, date: '2023-04', org: 'Meta AI',
  paper: 'https://arxiv.org/abs/2304.02643', code: 'https://github.com/facebookresearch/sam2', tags: 'sam,sam 2,sam 3,promptable segmentation,image encoder,prompt encoder,mask decoder',
  tagline: 'Promptable segmentation: a heavy image encoder runs once, then a tiny prompt-conditioned mask decoder answers each click or box in ~50 ms. SAM 2 adds video memory; SAM 3 adds text and exemplar "concept" prompts.',
  anatomy: { backbone: 'SAM: MAE ViT-H/16 at 1024² → 64×64×256 embedding. SAM 2: Hiera (hierarchical) encoder.', neck: 'Prompt encoder: points/boxes → positional encodings + learned type embeddings; masks → conv downsampling. SAM 2: memory attention over a memory bank of past frames.', head: 'Two-way transformer mask decoder (2 layers): tokens ↔ image cross-attention both ways → 3 candidate masks + IoU predictions (ambiguity-aware).', loss: 'Focal (20) + Dice (1) on masks, MSE on predicted IoU; trained with simulated interactive prompting on SA-1B (1.1B masks).' },
  pipeline: [P('input', 'Image + prompts', 'points, boxes, masks (SAM 3: text / exemplars)'), P('backbone', 'Image encoder', 'ViT-H (SAM) / Hiera (SAM 2)', ['Runs once per image', '64×64×256 embedding']),
    P('neck', 'Prompt encoder (+ memory in SAM 2)', '', ['Sparse: points, boxes → tokens', 'Dense: mask → conv', 'SAM 2: memory attention over past frames']),
    P('decoder', 'Mask decoder', 'two-way transformer ×2', ['Token ↔ image cross-attention', 'Upscale ×4 → 256×256', 'MLP hyper-networks → 3 masks + IoU scores']), P('output', 'Masks', 'upsampled to image size')],
  sections: [{ h: 'Versions', html: T(['Model', 'Date', 'Adds'], [['SAM', '2023-04', 'promptable image segmentation, SA-1B'], ['SAM 2', '2024-07', 'streaming video with memory bank, Hiera encoder, faster images'], ['SAM 3', '2025-11', 'concept prompts (noun phrase / exemplar) → detect, segment and track all instances; presence head']]) }]
});

CV.arch({
  id: 'unet', name: 'U-Net / nnU-Net', task: 'seg', match: null, date: '2015-05', org: 'Univ. of Freiburg / DKFZ',
  paper: 'https://arxiv.org/abs/1505.04597', code: 'https://github.com/MIC-DKFZ/nnUNet', tags: 'unet,nnunet,medical segmentation,encoder decoder,skip connections',
  tagline: 'Symmetric encoder-decoder with skip connections at every resolution; still the default for medical imaging (nnU-Net auto-configures it) and the classic diffusion denoiser.',
  anatomy: { backbone: 'Contracting path: 4× [2 × (3×3 conv-BN-ReLU) → 2×2 max-pool], channels 64→512, bottleneck 1024.', neck: 'Skip connections concatenate encoder features into the decoder at the same resolution.', head: 'Expanding path: 2×2 up-conv → concat skip → 2 × 3×3 conv; final 1×1 conv to K.', loss: 'CE + Dice (nnU-Net), deep supervision at several decoder levels.' },
  pipeline: [P('input', 'Image / patch', '2D slice or 3D patch'), P('backbone', 'Encoder', '', ['64 → 128 → 256 → 512 → 1024', 'Max-pool ÷2 each level']), P('neck', 'Skip connections', 'concat at each level'), P('head', 'Decoder', '', ['Up-conv ×2', 'Concat skip', '2 × conv', '1×1 → K']), P('output', 'Mask', 'same size as input')]
});
})();

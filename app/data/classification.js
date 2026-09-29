(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'cls', title: 'Classification', icon: '🏷', boards: ['cls-in1k'],
  blurb: 'CNN → ViT → hybrid & foundation backbones, layer-by-layer tables, losses, metrics, recipes.',
  intro: 'Image → label. Every backbone here is also the feature extractor for detection, segmentation and pose.',
  sections: [
  { id: 'heads', title: 'Problem Setup & Heads', items: [
    { id: 'multiclass', t: 'Multi-class vs multi-label head', tags: 'softmax head,sigmoid head,multi-label,classifier head',
      f: [R`\text{multi-class: } p=\mathrm{softmax}(W\,\mathrm{GAP}(F)+b)`, R`\text{multi-label: } p_c=\sigma(w_c^\top h+b_c),\;\;L=\sum_c\text{BCE}(y_c,p_c)`],
      h: R`<p>Head = GAP (or CLS token) → (dropout) → Linear. Hierarchical labels: multi-head or tree softmax. Open-set: threshold max-softmax/energy score \(E=-\tau\log\sum e^{z_c/\tau}\).</p>` },
    { id: 'zeroshot', t: 'Zero-shot classification (CLIP-style)', tags: 'zero-shot,clip,siglip,prompt,open vocabulary',
      f: [R`p(c|x)=\frac{\exp(\cos(f_I(x),f_T(t_c))/\tau)}{\sum_{c'}\exp(\cos(f_I(x),f_T(t_{c'}))/\tau)}`],
      h: R`<p>Prompt ensembling ("a photo of a {c}.") +1–3%. Linear probe on frozen features usually beats zero-shot with ≥ 4–16 shots/class.</p>` }
  ]},
  { id: 'arch', title: 'Architectures & Layer Formations', items: [
    { id: 'classic', t: 'Classic CNN milestones', tags: 'lenet,alexnet,vgg,googlenet,inception,history',
      h: T(['Model (year)', 'Key idea', 'Params', 'IN top-1'], [['LeNet-5 (1998)', 'conv-pool-FC for digits', '60k', '—'], ['AlexNet (2012)', 'ReLU, dropout, GPU, LRN', '61M', '56.5'], ['VGG-16 (2014)', 'stack 3×3 convs, depth', '138M', '71.6'], ['GoogLeNet/Inception (2014)', 'multi-branch 1×1/3×3/5×5, 1×1 reductions, aux heads', '6.8M', '69.8'], ['Inception-v3 (2015)', 'factorized convs (n×1, 1×n), label smoothing', '24M', '77.3'], ['ResNet-50 (2015)', 'residual learning', '25.6M', '76.1'], ['DenseNet-121 (2016)', 'concat all previous features, growth rate k', '8M', '74.4'], ['ResNeXt-50 32×4d (2016)', 'grouped conv, cardinality', '25M', '77.6']]) },
    { id: 'resnet50', t: 'ResNet-50 layer by layer', tags: 'resnet,resnet50,layers,bottleneck,stages',
      h: T(['Stage', 'Layers', 'Output (224 input)', 'Stride'], [['stem', '7×7, 64, s2 → BN-ReLU → 3×3 maxpool s2', '64 × 56 × 56', '4'], ['conv2_x', '[1×1,64 · 3×3,64 · 1×1,256] × 3', '256 × 56 × 56', '4'], ['conv3_x', '[1×1,128 · 3×3,128 · 1×1,512] × 4', '512 × 28 × 28', '8'], ['conv4_x', '[1×1,256 · 3×3,256 · 1×1,1024] × 6', '1024 × 14 × 14', '16'], ['conv5_x', '[1×1,512 · 3×3,512 · 1×1,2048] × 3', '2048 × 7 × 7', '32'], ['head', 'GAP → FC 1000', '1000', '—']]) + R`<p>25.6M params, 4.1 GFLOPs. Depths: R18 [2,2,2,2] basic, R34 [3,4,6,3] basic, R101 [3,4,23,3], R152 [3,8,36,3]. "ResNet-D" tweak: stride in 3×3 not 1×1, avg-pool shortcut. Stages C3–C5 (strides 8/16/32) feed FPNs.</p>` },
    { id: 'effnet', t: 'EfficientNet (compound scaling) & B0 layout', tags: 'efficientnet,compound scaling,mbconv,efficientnetv2',
      f: [R`d=\alpha^\phi,\;w=\beta^\phi,\;r=\gamma^\phi,\;\;\text{s.t. }\alpha\beta^2\gamma^2\approx2,\;\;\alpha{=}1.2,\beta{=}1.1,\gamma{=}1.15`],
      h: T(['Stage', 'Op', 'k', 'Stride', 'Ch', 'Layers'], [['1', 'Conv', '3', '2', '32', '1'], ['2', 'MBConv1', '3', '1', '16', '1'], ['3', 'MBConv6', '3', '2', '24', '2'], ['4', 'MBConv6', '5', '2', '40', '2'], ['5', 'MBConv6', '3', '2', '80', '3'], ['6', 'MBConv6', '5', '1', '112', '3'], ['7', 'MBConv6', '5', '2', '192', '4'], ['8', 'MBConv6', '3', '1', '320', '1'], ['9', 'Conv1×1 → GAP → FC', '1', '—', '1280', '1']]) + R`<p>B0: 5.3M params, 0.39 GFLOPs, 224². B7: 600². EfficientNetV2: Fused-MBConv early, progressive learning (small image + weak aug → large image + strong aug).</p>` },
    { id: 'mobilenet', t: 'MobileNet v1 → v4', tags: 'mobilenet,mobilenetv3,mobilenetv4,uib,edge,nas',
      h: T(['Version', 'Block', 'Key additions'], [['v1 (2017)', 'DW 3×3 + PW 1×1', 'width multiplier α, resolution ρ'], ['v2 (2018)', 'inverted residual, linear bottleneck', 'ReLU6, t=6'], ['v3 (2019)', 'MBv2 + SE + h-swish', 'NAS (MnasNet) + NetAdapt'], ['v4 (2024)', 'Universal Inverted Bottleneck (UIB): optional DW before/after expansion (ExtraDW, ConvNext-like, FFN, IB)', 'Mobile MQA attention; Pareto-optimal across CPU/GPU/DSP/EdgeTPU']]) },
    { id: 'vit', t: 'Vision Transformer (ViT) layer by layer', tags: 'vit,vision transformer,patch,cls token,vit-b/16',
      f: [R`z_0=[x_{cls};\,x_p^1E;\dots;x_p^NE]+E_{pos},\;\;E\in\mathbb R^{(P^2C)\times D}`, R`z'_l=z_{l-1}+\mathrm{MSA}(\mathrm{LN}(z_{l-1})),\;\;z_l=z'_l+\mathrm{MLP}(\mathrm{LN}(z'_l)),\;\;y=\mathrm{LN}(z_L^0)`],
      h: T(['Variant', 'Layers', 'Hidden D', 'MLP', 'Heads', 'Params', 'GFLOPs (224, /16)'], [['ViT-Ti', '12', '192', '768', '3', '5.7M', '1.3'], ['ViT-S', '12', '384', '1536', '6', '22M', '4.6'], ['ViT-B', '12', '768', '3072', '12', '86M', '17.6'], ['ViT-L', '24', '1024', '4096', '16', '304M', '61.6'], ['ViT-H', '32', '1280', '5120', '16', '632M', '167 (/14)'], ['ViT-g', '40', '1408–1536', '6144', '16–24', '1.0–1.1B', '—']]) + R`<p>224/16 → 196 patch tokens + CLS. Lacks conv inductive bias → needs big data (JFT/IN-21k) or DeiT-style aug/distillation. Plain ViTs as detection backbones: ViTDet (simple feature pyramid from last map).</p>` },
    { id: 'deit', t: 'DeiT (data-efficient ViT)', tags: 'deit,distillation token,vit training recipe',
      h: R`<p>Trains ViT on ImageNet-1k only: AdamW 5e-4×bs/512, WD 0.05, 300 ep, RandAug(9,0.5), mixup 0.8, cutmix 1.0, random erasing 0.25, drop-path 0.1, repeated aug, label smoothing 0.1. Distillation token learns from a CNN teacher (hard labels). DeiT III: simpler aug (3-Augment), LayerScale, BCE loss.</p>` },
    { id: 'swin', t: 'Swin Transformer', tags: 'swin,hierarchical transformer,shifted window,patch merging',
      f: [R`\mathrm{Attn}=\mathrm{softmax}(QK^\top/\sqrt d+B)V,\;\;B\in\mathbb{R}^{M^2\times M^2}\text{ relative position bias}`],
      h: T(['Stage', 'Resolution', 'Swin-T dims × blocks'], [['Patch partition 4×4 + linear', 'H/4', '96 × 2'], ['Patch merging (2×2 concat → linear)', 'H/8', '192 × 2'], ['Patch merging', 'H/16', '384 × 6'], ['Patch merging', 'H/32', '768 × 2']]) + R`<p>Window M=7. Swin-T/S/B: C=96/96/128, depths [2,2,6,2]/[2,2,18,2]/[2,2,18,2]. Linear complexity in pixels; hierarchical maps plug into FPN.</p>` },
    { id: 'convnext', t: 'ConvNeXt (v1/v2)', tags: 'convnext,modernized resnet,grn,fcmae',
      h: T(['Change vs ResNet-50', 'Top-1 gain path'], [['Training recipe (AdamW, 300 ep, mixup/cutmix, RandAug)', '76.1 → 78.8'], ['Stage ratio 3:3:9:3, patchify stem 4×4 s4', '→ 79.5'], ['Depthwise conv, width 96', '→ 80.5'], ['Inverted bottleneck (4× expansion)', '→ 80.6'], ['7×7 DW kernel (moved up)', '→ 80.6'], ['GELU, fewer activations/norms, LN, separate downsampling layers', '→ 82.0']]) + R`<p>Block: DW 7×7 → LN → 1×1 (4C) → GELU → 1×1 (C) → LayerScale → DropPath +x. <b>V2</b> adds Global Response Normalization \(\text{GRN}(x)=\gamma\, x\,\frac{\|x\|_{HW}}{\text{mean}_c\|x\|_{HW}}+\beta+x\) and FCMAE masked pre-training.</p>` },
    { id: 'hybrid', t: 'Hybrid & efficient transformers', tags: 'coatnet,maxvit,efficientvit,fastvit,levit,hybrid',
      h: T(['Model', 'Idea'], [['CoAtNet', 'MBConv stages then transformer stages; relative attention'], ['MaxViT', 'block (local) + grid (dilated global) attention, multi-axis'], ['EfficientViT (MIT)', 'ReLU linear attention + multi-scale tokens for high-res'], ['FastViT / MobileOne', 'structural reparameterization: train multi-branch, infer single conv'], ['RepViT', 'MobileNet-style CNN with ViT design lessons'], ['Hiera', 'hierarchical ViT stripped of extras, MAE-pretrained; SAM 2 backbone']]) },
    { id: 'foundation', t: 'Foundation backbones (SSL / CLIP)', tags: 'dinov2,dinov3,clip,siglip,eva,foundation model,backbone',
      h: T(['Backbone', 'Pre-training', 'Strength'], [['CLIP / OpenCLIP', 'image-text contrastive (400M–2B pairs)', 'zero-shot, VLM encoders'], ['SigLIP / SigLIP 2', 'pairwise sigmoid loss (+captioning, self-distillation in v2)', 'best VLM vision tower, multilingual'], ['EVA-02', 'MIM reconstructing CLIP features', 'top fine-tune accuracy'], ['DINOv2 (2023)', 'self-distillation + iBOT on curated LVD-142M', 'frozen features for dense tasks'], ['DINOv3 (2025)', '7B ViT, Gram anchoring against dense-feature degradation, 1.7B images', 'SOTA frozen dense features (seg/depth/detection)'], ['MAE', 'mask 75% patches, reconstruct pixels', 'scalable, efficient pre-training'], ['AIMv2 / Web-SSL', 'autoregressive / scaled SSL on web data', 'multimodal / scaling studies']]) },
    { id: 'mamba', t: 'State-space vision backbones', tags: 'vmamba,vision mamba,mambaout,ssm backbone',
      h: R`<p>Vim (bidirectional scans), VMamba (SS2D, 4-direction cross-scan): VMamba-T 82.6% top-1. MambaOut argues the SSM is unnecessary for classification (gated CNN matches) but helpful for long-sequence dense tasks.</p>` }
  ]},
  { id: 'loss', title: 'Loss Functions (classification)', items: [
    { id: 'ce', t: 'Cross-entropy + label smoothing', tags: 'cross entropy,label smoothing,softmax loss',
      f: [R`L=-\sum_c\left[(1-\varepsilon)y_c+\tfrac{\varepsilon}{K}\right]\log p_c`] },
    { id: 'bcecls', t: 'BCE for single-label (DeiT III / RSB)', tags: 'bce,binary cross entropy,resnet strikes back',
      f: [R`L=-\sum_c\big[y_c\log\sigma(z_c)+(1-y_c)\log(1-\sigma(z_c))\big]`],
      h: R`<p>Works well with mixup/cutmix soft targets (binarize with threshold 0.2 in RSB).</p>` },
    { id: 'asl', t: 'Asymmetric loss (multi-label)', tags: 'asymmetric loss,asl,multi-label,negative imbalance',
      f: [R`L_+=(1-p)^{\gamma_+}\log p,\;\;L_-=p_m^{\gamma_-}\log(1-p_m),\;\;p_m=\max(p-m,0)`],
      h: R`<p>γ₊=0, γ₋=4, m=0.05. Discards easy negatives and mislabeled negatives; SOTA on MS-COCO multi-label / OpenImages.</p>` },
    { id: 'cb', t: 'Class-balanced / LDAM / logit adjustment', tags: 'long tail,class balanced loss,ldam,logit adjustment,balanced softmax',
      f: [R`w_c=\frac{1-\beta}{1-\beta^{n_c}}`, R`\text{LDAM: } z_y-\Delta_y,\;\Delta_y\propto n_y^{-1/4}`, R`\text{Logit adj.: } \mathrm{CE}(z+\tau\log\pi,\,y)`] },
    { id: 'poly', t: 'PolyLoss & focal for classification', tags: 'polyloss,focal loss classification',
      f: [R`L_{Poly-1}=-\log p_t+\epsilon_1(1-p_t)`],
      h: R`<p>ε₁=1–2 small but consistent gains; focal loss (γ=1–2) helps heavily imbalanced or noisy classification.</p>` }
  ]},
  { id: 'metrics', title: 'Metrics', items: [
    { id: 'topk', t: 'Top-k, balanced accuracy, mAP (multi-label)', tags: 'top-1,top-5,accuracy,balanced accuracy,multi-label map',
      f: [R`\text{Top-}k=\frac1N\sum_i\mathbb{1}[y_i\in\text{top}_k(p_i)]`, R`\text{BalAcc}=\frac1K\sum_c\text{Recall}_c`, R`\text{mAP}_{ml}=\frac1K\sum_c\text{AP}_c`],
      h: R`<p>Robustness benchmarks: ImageNet-V2 (−10%), -R, -Sketch, -A, -C (mCE). Report them for deployment-relevant claims.</p>` }
  ]},
  { id: 'train', title: 'Training Recipes & Transfer', items: [
    { id: 'rsb', t: 'Modern ResNet-50 recipe (ResNet Strikes Back A1)', tags: 'resnet strikes back,timm recipe,lamb',
      h: T(['Hyperparameter', 'A1 (80.4%)'], [['Epochs / batch', '600 / 2048'], ['Optimizer', 'LAMB, LR 5e-3, WD 0.01'], ['Schedule', 'cosine, 5 warmup'], ['Loss', 'BCE'], ['Aug', 'RandAug(7,0.5), mixup 0.1, cutmix 1.0, repeated aug'], ['Reg', 'stochastic depth 0.05, no label smoothing'], ['Res train/test', '224 / 224']]) },
    { id: 'transfer', t: 'Linear probe vs fine-tune vs LP-FT', tags: 'linear probe,fine-tuning,lp-ft,peft,adapter,frozen backbone',
      h: T(['Method', 'Trainable', 'When'], [['k-NN on features', 'none', 'instant baseline, retrieval'], ['Linear probe', 'head', 'tiny data, strong frozen features (DINOv2/v3)'], ['LP-FT', 'head first, then all at low LR', 'keeps OOD robustness'], ['Partial fine-tune', 'last N stages', 'medium data, compute-limited'], ['Full fine-tune + LLRD', 'all', 'enough data, best ID accuracy'], ['PEFT (LoRA, adapters, VPT)', '< 1–5%', 'many tasks sharing one backbone']]) }
  ]},
  { id: 'data', title: 'Datasets', items: [
    { id: 'ds', t: 'Classification datasets', tags: 'imagenet,cifar,datasets,inaturalist,imagenet-21k',
      h: T(['Dataset', 'Size', 'Notes'], [['ImageNet-1k (ILSVRC12)', '1.28M train / 50k val, 1000 cls', 'standard benchmark'], ['ImageNet-21k', '14M, ~21k cls', 'pre-training'], ['CIFAR-10/100', '50k/10k 32×32', 'quick experiments'], ['iNaturalist 2021', '2.7M, 10k species', 'long-tail, fine-grained'], ['Places365', '1.8M scenes', 'scene recognition'], ['VTAB / ELEVATER', '19 / 20 transfer tasks', 'transfer evaluation'], ['LAION-5B / DataComp', 'billions of image-text pairs', 'CLIP-style pre-training']]) }
  ]}
  ]
});
})();

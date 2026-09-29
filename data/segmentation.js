(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'seg', title: 'Segmentation', icon: '◩', boards: ['seg-ade', 'seg-cocoinst'],
  blurb: 'Semantic, instance, panoptic, promptable (SAM 1/2/3), U-Net → Mask2Former, Dice/Lovász/Tversky losses.',
  intro: 'Pixel-level labelling. Semantic = class per pixel; instance = mask per object; panoptic = both (things + stuff).',
  sections: [
  { id: 'tasks', title: 'Task Formulations', items: [
    { id: 'kinds', t: 'Semantic vs instance vs panoptic vs promptable', tags: 'semantic segmentation,instance segmentation,panoptic,promptable,interactive',
      h: T(['Task', 'Output', 'Metric', 'Typical models'], [['Semantic', 'H×W class map', 'mIoU', 'DeepLabv3+, SegFormer, UperNet, Mask2Former'], ['Instance', 'set of (mask, class, score)', 'mask AP', 'Mask R-CNN, YOLO-seg, Mask2Former, RF-DETR-Seg'], ['Panoptic', 'every pixel → (class, instance id)', 'PQ', 'Panoptic FPN, Mask2Former, OneFormer'], ['Promptable', 'mask from point/box/text/exemplar', 'mIoU @ clicks, J&F (video)', 'SAM, SAM 2, SAM 3'], ['Referring', 'mask from sentence', 'cIoU/gIoU', 'LISA, GLaMM, VLM + SAM']]) },
    { id: 'perpixel', t: 'Per-pixel vs mask classification', tags: 'mask classification,per-pixel classification,maskformer',
      f: [R`\text{per-pixel: } p(y_{ij}=c)=\mathrm{softmax}(z_{ij})_c`, R`\text{mask cls: } \{(p_i,m_i)\}_{i=1}^{N},\;\;\text{sem}(x,y)=\arg\max_c\sum_i p_i(c)\,m_i(x,y)`],
      h: R`<p>Mask classification (MaskFormer) unifies semantic, instance and panoptic with one architecture and loss.</p>` }
  ]},
  { id: 'arch', title: 'Architectures & Layer Formations', items: [
    { id: 'fcn', t: 'FCN / SegNet / PSPNet', tags: 'fcn,segnet,pspnet,pyramid pooling',
      h: R`<p><b>FCN</b>: replace FC with 1×1 conv, upsample (FCN-32s/16s/8s with skips). <b>SegNet</b>: decoder upsamples with max-pool indices. <b>PSPNet</b>: pyramid pooling module (bins 1,2,3,6) → concat → conv, captures global context.</p>` },
    { id: 'unet', t: 'U-Net layer by layer', tags: 'unet,u-net,encoder decoder,skip connections,medical,nnunet',
      h: T(['Level', 'Encoder', 'Decoder', 'Ch'], [['1', '2×(3×3 conv-BN-ReLU)', 'concat skip → 2×conv → 1×1 conv to K', '64'], ['2', 'maxpool 2 → 2×conv', 'up-conv 2×2 → concat → 2×conv', '128'], ['3', 'maxpool → 2×conv', 'up → concat → 2×conv', '256'], ['4', 'maxpool → 2×conv', 'up → concat → 2×conv', '512'], ['bottleneck', 'maxpool → 2×conv', '—', '1024']]) + R`<p>Skips restore spatial detail lost in downsampling. <b>nnU-Net</b> self-configures patch size, spacing, depth, and uses Dice+CE, deep supervision, 1000 epochs × 250 iters, SGD 0.01 poly — still a top medical baseline. U-Net (with attention & timestep embedding) is also the classic diffusion denoiser.</p>` },
    { id: 'deeplab', t: 'DeepLabv3+ (ASPP)', tags: 'deeplab,aspp,atrous spatial pyramid pooling,encoder decoder',
      h: T(['Block', 'Details'], [['Backbone', 'ResNet/Xception with atrous conv → output stride 16 (8 for accuracy)'], ['ASPP', 'parallel: 1×1 conv, 3×3 dil 6/12/18 (OS16), image-level GAP → 1×1 → upsample; concat → 1×1 256'], ['Decoder', 'upsample ×4 → concat with low-level (C2 reduced to 48 ch) → 2×3×3 conv → upsample ×4']]) },
    { id: 'hrnet', t: 'HRNet (high-resolution network)', tags: 'hrnet,high resolution,multi-branch,parallel streams',
      h: R`<p>Keeps a high-resolution branch throughout, adds lower-resolution branches stage by stage (1/4, 1/8, 1/16, 1/32) with repeated multi-scale fusion (strided conv down, 1×1+upsample up). Widths W18/W32/W48 = channels of the high-res branch. Strong for segmentation and pose (see Pose tab).</p>` },
    { id: 'segformer', t: 'SegFormer', tags: 'segformer,mix transformer,mit,efficient self-attention,all-mlp decoder',
      f: [R`\text{Efficient attn: } K'=\text{Linear}(\text{Reshape}(K,\tfrac{N}{R},CR)),\;\;O\!\left(\tfrac{N^2}{R}\right)`, R`\text{Mix-FFN: } x_{out}=\text{MLP}(\text{GELU}(\text{DWConv}_{3\times3}(\text{MLP}(x_{in}))))+x_{in}`],
      h: T(['Stage', 'Stride', 'B0 / B5 channels', 'Reduction R'], [['1 (overlap patch embed 7×7 s4)', '4', '32 / 64', '8'], ['2 (3×3 s2)', '8', '64 / 128', '4'], ['3', '16', '160 / 320', '2'], ['4', '32', '256 / 512', '1'], ['All-MLP decoder', 'unify to C, upsample to 1/4, concat, fuse, predict', '256 / 768', '']]) + R`<p>No positional encoding (Mix-FFN conv leaks position) → robust to resolution changes.</p>` },
    { id: 'maskrcnn', t: 'Mask R-CNN / YOLO-seg / RF-DETR-Seg', tags: 'mask r-cnn,yolo-seg,yolact,prototype masks,instance segmentation',
      f: [R`\text{YOLACT/YOLO-seg: } M=\sigma(P\,C^\top),\;\;P\in\mathbb R^{HW\times k}\text{ prototypes},\;C\in\mathbb R^{n\times k}\text{ coefficients}`],
      h: R`<p><b>Mask R-CNN</b>: Faster R-CNN + parallel FCN mask head on 14×14 RoIAlign → 28×28 per-class binary mask (BCE). <b>YOLO-seg</b>: 32 prototype masks from P3 + 32 coefficients per detection, crop by box. <b>RF-DETR-Seg</b>: query-based masks as dot products of query embeddings with a pixel embedding map.</p>` },
    { id: 'mask2former', t: 'MaskFormer / Mask2Former / OneFormer', tags: 'mask2former,maskformer,oneformer,masked attention,universal segmentation',
      f: [R`X_l=\mathrm{softmax}(\mathcal{M}_{l-1}+Q_lK_l^\top)V_l+X_{l-1},\;\;\mathcal M_{l-1}(x,y)=\begin{cases}0 & M_{l-1}(x,y)=1\\-\infty&\text{else}\end{cases}`],
      h: T(['Component', 'Details'], [['Backbone', 'ResNet / Swin / ConvNeXt / DINOv2 (ViT-Adapter)'], ['Pixel decoder', 'MSDeformAttn (6 layers) over 1/8,1/16,1/32 → per-pixel embeddings at 1/4'], ['Transformer decoder', '9 layers (3 rounds × 3 scales), masked cross-attn → self-attn → FFN; 100–200 queries'], ['Outputs', 'per query: class (K+1) + mask embedding · pixel embedding'], ['Training trick', 'mask loss on K=12544 sampled points (importance sampling) → 3× less memory']]) + R`<p>OneFormer adds a task token (semantic/instance/panoptic) for one model trained once for all three tasks.</p>` },
    { id: 'sam', t: 'Segment Anything (SAM → SAM 2 → SAM 3)', tags: 'sam,sam 2,sam 3,segment anything,promptable,video segmentation,memory attention',
      h: T(['Model (date)', 'Architecture', 'Notes'], [['SAM (2023-04)', 'MAE ViT-H image encoder (1024², run once) + prompt encoder (points/boxes/masks) + light two-way transformer mask decoder → 3 masks + IoU scores', 'SA-1B: 11M images, 1.1B masks; ~50 ms decoder'], ['SAM 2 (2024-07)', 'Hiera encoder + memory attention + memory bank (recent frames + prompted frames) + occlusion head', 'video; streaming; SA-V dataset'], ['SAM 3 (2025-11)', 'adds concept prompts (noun phrase / image exemplar) → detect, segment and track all instances; presence token decouples recognition from localization', 'open-vocab instance seg + tracking'], ['Efficient variants', 'MobileSAM, EfficientSAM, EdgeSAM, SAM2-tiny', 'edge devices']]),
      tip: 'SAM-family output is class-agnostic (SAM 1/2): pair with a detector (Grounded-SAM) or use SAM 3 for text prompts. Great for label bootstrapping: box → mask conversion of detection datasets.' }
  ]},
  { id: 'loss', title: 'Loss Functions (segmentation)', items: [
    { id: 'pixce', t: 'Pixel-wise (weighted) cross-entropy & OHEM', tags: 'pixel cross entropy,weighted ce,ohem,ignore index',
      f: [R`L=-\frac1{|\Omega|}\sum_{i\in\Omega}w_{y_i}\log p_{i,y_i}`, R`\text{OHEM: keep pixels with } p_{i,y_i}<0.7\text{, at least } n_{min}=10^5`],
      h: R`<p>Use <code>ignore_index=255</code> for void/boundary pixels. Median-frequency class weights \(w_c=\text{median}(f)/f_c\).</p>` },
    { id: 'dice', t: 'Dice / soft Dice loss', tags: 'dice loss,f1,soft dice,generalized dice,medical',
      f: [R`L_{Dice}=1-\frac{2\sum_i p_ig_i+\epsilon}{\sum_i p_i+\sum_i g_i+\epsilon}`, R`\text{Generalized Dice: } w_c=\frac1{(\sum_i g_{ic})^2}`],
      h: R`<p>Region-based, robust to foreground/background imbalance. Noisy for tiny/empty masks → combine: \(L=L_{CE}+L_{Dice}\) (nnU-Net, Mask2Former mask loss = BCE(5) + Dice(5) + cls CE(2)).</p>` },
    { id: 'iou', t: 'Jaccard / Lovász-Softmax', tags: 'jaccard loss,iou loss,lovasz,lovász softmax',
      f: [R`L_{Jacc}=1-\frac{\sum p_ig_i}{\sum p_i+\sum g_i-\sum p_ig_i}`, R`\text{Lovász: } L=\frac1{|C|}\sum_c\overline{\Delta_{J_c}}(m(c)),\;m_i(c)=\begin{cases}1-p_i(c)&c=y_i\\p_i(c)&\text{else}\end{cases}`],
      h: R`<p>Lovász extension is a tight convex surrogate of IoU; fine-tune with it after CE training for +1–2 mIoU.</p>` },
    { id: 'tversky', t: 'Tversky & Focal Tversky', tags: 'tversky loss,focal tversky,false negatives,small lesions',
      f: [R`TI=\frac{\sum p_ig_i}{\sum p_ig_i+\alpha\sum p_i(1-g_i)+\beta\sum(1-p_i)g_i},\;\;L=(1-TI)^{\gamma}`],
      h: R`<p>β>α (e.g. 0.7/0.3) penalizes false negatives more → better recall on small structures. γ=0.75 in focal Tversky.</p>` },
    { id: 'boundary', t: 'Boundary / Hausdorff / clDice', tags: 'boundary loss,hausdorff,cldice,topology,thin structures',
      f: [R`L_B=\int_\Omega\phi_G(q)\,s_\theta(q)\,dq\;\;(\phi_G:\text{signed distance to GT boundary})`, R`\text{clDice}=2\frac{T_{prec}\cdot T_{sens}}{T_{prec}+T_{sens}}\;\text{(skeleton-based)}`],
      h: R`<p>Boundary loss for highly unbalanced segmentation (add gradually to Dice). clDice preserves connectivity of vessels/roads.</p>` },
    { id: 'focalseg', t: 'Focal / combo / mask losses', tags: 'focal loss segmentation,combo loss,mask loss,point sampling',
      f: [R`L_{combo}=\alpha L_{CE/Focal}+(1-\alpha)L_{Dice}`],
      h: T(['Setting', 'Recommended loss'], [['Balanced semantic seg', 'CE (+ aux CE 0.4)'], ['Imbalanced / medical', 'Dice + CE, or Focal Tversky'], ['Binary thin structures', 'BCE + Dice + clDice'], ['Mask2Former-style', 'point-sampled BCE + Dice, CE for classes (no-object 0.1)'], ['Instance (Mask R-CNN)', 'per-pixel BCE on 28×28 mask'], ['Boundary quality', '+ boundary / Lovász fine-tune']]) }
  ]},
  { id: 'metrics', title: 'Metrics', items: [
    { id: 'miou', t: 'mIoU, pixel accuracy, Dice', tags: 'miou,mean iou,pixel accuracy,dice score,fwiou',
      f: [R`\text{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c},\;\;\text{mIoU}=\frac1K\sum_c\text{IoU}_c`, R`\text{Dice}=\frac{2TP}{2TP+FP+FN}=\frac{2\,\text{IoU}}{1+\text{IoU}}`],
      h: R`<p>Accumulate TP/FP/FN over the whole dataset (not per image) for standard mIoU. Pixel accuracy is dominated by large classes.</p>` },
    { id: 'pq', t: 'Panoptic Quality (PQ)', tags: 'pq,panoptic quality,sq,rq',
      f: [R`\text{PQ}=\underbrace{\frac{\sum_{(p,g)\in TP}\text{IoU}(p,g)}{|TP|}}_{SQ}\times\underbrace{\frac{|TP|}{|TP|+\tfrac12|FP|+\tfrac12|FN|}}_{RQ}`],
      h: R`<p>Match if IoU > 0.5 (unique). Report PQ<sup>Th</sup>/PQ<sup>St</sup> for things/stuff.</p>` },
    { id: 'bnd', t: 'Mask AP, Boundary IoU, J&F', tags: 'mask ap,boundary iou,boundary ap,j&f,video object segmentation',
      h: R`<p>Mask AP = COCO AP using mask IoU. Boundary IoU measures overlap in a band of width d around contours (sensitive to edge quality on large objects). Video object segmentation: \(\mathcal J\) (region IoU) & \(\mathcal F\) (contour F-measure), averaged as J&F.</p>` }
  ]},
  { id: 'practical', title: 'Practical Tips', items: [
    { id: 'tips', t: 'Segmentation practitioner notes', tags: 'segmentation tips,sliding window,crf,resolution,annotation',
      h: R`<ul><li>High-res images: train on crops, infer with sliding window (overlap 1/3) and average logits.</li><li>Resize masks with <b>nearest</b> interpolation only; resize logits (not argmax) with bilinear.</li>
        <li>Output stride 8 vs 16: +1–2 mIoU at ~3× compute in the decoder.</li><li>Boundary refinement: SegFix, PointRend (render uncertain points), or CRF for legacy CNN outputs.</li>
        <li>Annotation cost: SAM-assisted labelling cuts polygon time 5–10×; polygon quality directly caps achievable mIoU.</li><li>Class imbalance: rare-class crop sampling (mmseg <code>cat_max_ratio=0.75</code>).</li></ul>` }
  ]},
  { id: 'data', title: 'Datasets', items: [
    { id: 'ds', t: 'Segmentation datasets', tags: 'ade20k,cityscapes,coco-stuff,pascal context,mapillary,sa-1b,datasets',
      h: T(['Dataset', 'Size / classes', 'Task'], [['ADE20K', '20k train / 2k val, 150 cls', 'semantic (standard)'], ['Cityscapes', '2975 / 500 fine, 19 cls, 2048×1024', 'urban semantic/instance/panoptic'], ['COCO (panoptic / stuff)', '118k, 133 (80 things + 53 stuff)', 'instance, panoptic'], ['Pascal VOC / Context', '10.6k aug / 59 cls', 'semantic'], ['Mapillary Vistas', '25k, 124 cls', 'street scenes'], ['SA-1B / SA-V', '11M imgs, 1.1B masks / 51k videos', 'promptable'], ['Medical (MSD, BraTS, KiTS)', 'CT/MRI 3D', 'nnU-Net benchmarks']]) }
  ]}
  ]
});
})();

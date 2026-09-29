(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'det', title: 'Object Detection', short: 'Detection', icon: '▣', boards: ['det-rt', 'det-coco'],
  blurb: 'Box encoding, IoU, NMS, label assignment, two-stage / one-stage / DETR, YOLO26, RF-DETR, open-vocab, losses.',
  intro: 'Image → set of (box, class, score). Leaderboards are COCO val2017 box AP@[.5:.95] so every model is measured on the same data.',
  sections: [
  { id: 'fund', title: 'Fundamentals', items: [
    { id: 'boxenc', t: 'Box parameterization & encoding', tags: 'bounding box,xyxy,xywh,cxcywh,box encoding,anchor offsets',
      f: [R`t_x=\frac{x-x_a}{w_a},\;t_y=\frac{y-y_a}{h_a},\;t_w=\log\frac{w}{w_a},\;t_h=\log\frac{h}{h_a}`, R`\text{YOLOv5: } b_{xy}=2\sigma(t_{xy})-0.5+c_{xy},\;\;b_{wh}=p_{wh}\,(2\sigma(t_{wh}))^2`, R`\text{Anchor-free (FCOS/YOLOv8+): } (l,t,r,b)\text{ distances from point, } \times\text{stride}`],
      h: R`<p>Formats: COCO json = [x,y,w,h] (top-left), YOLO txt = normalized [cx,cy,w,h], Pascal VOC = [x1,y1,x2,y2]. Off-by-one and normalization mismatches are the #1 data bug.</p>` },
    { id: 'iou', t: 'IoU and variants', tags: 'iou,giou,diou,ciou,intersection over union',
      f: [R`\text{IoU}=\frac{|A\cap B|}{|A\cup B|}`, R`\text{GIoU}=\text{IoU}-\frac{|C\setminus(A\cup B)|}{|C|}`, R`\text{DIoU}=\text{IoU}-\frac{\rho^2(b,b^{gt})}{c^2}`, R`\text{CIoU}=\text{DIoU}-\alpha v,\;\;v=\tfrac{4}{\pi^2}\left(\arctan\tfrac{w^{gt}}{h^{gt}}-\arctan\tfrac{w}{h}\right)^2,\;\alpha=\tfrac{v}{(1-\text{IoU})+v}`],
      h: R`<p>\(C\): smallest enclosing box, \(\rho\): center distance, \(c\): enclosing-box diagonal. GIoU gives gradients for non-overlapping boxes; DIoU/CIoU converge faster.</p>` },
    { id: 'anchors', t: 'Anchors vs anchor-free vs queries', tags: 'anchors,anchor-free,object queries,priors,k-means anchors',
      h: T(['Paradigm', 'Priors', 'Examples'], [['Anchor-based', 'k boxes per location (scales × ratios; k-means on data for YOLOv2–v5)', 'Faster R-CNN, RetinaNet, YOLOv3–v5'], ['Anchor-free (point)', 'one point per location, regress distances', 'FCOS, CenterNet, YOLOX, YOLOv8–26'], ['Query-based (set prediction)', 'N learned queries + Hungarian matching', 'DETR, DINO, RT-DETR, D-FINE, RF-DETR']]) },
    { id: 'nms', t: 'NMS, Soft-NMS, WBF & NMS-free', tags: 'nms,non-maximum suppression,soft-nms,weighted boxes fusion,nms-free',
      f: [R`\text{Soft-NMS: } s_i\leftarrow s_i\,e^{-\text{IoU}(M,b_i)^2/\sigma}`, R`\text{WBF: } b=\frac{\sum_i s_ib_i}{\sum_i s_i}`],
      h: R`<p>Greedy NMS: sort by score, keep max, drop boxes with IoU > thr (0.5–0.7). Class-aware vs agnostic. NMS-free models (DETR family, YOLOv10/YOLO26 one-to-one head) avoid the latency tail and threshold tuning.</p>`,
      code: R`
import torchvision
keep = torchvision.ops.batched_nms(boxes_xyxy, scores, class_ids, iou_threshold=0.6)` },
    { id: 'assign', t: 'Label assignment strategies', tags: 'label assignment,atss,simota,tal,task aligned,hungarian matching,positive samples',
      f: [R`\text{ATSS: IoU thr}=\mu_g+\sigma_g\text{ of top-}k\text{ candidates per level}`, R`\text{TAL: } t=s^{\alpha}\cdot u^{\beta}\;(\alpha{=}1,\beta{=}6),\;\text{top-}k{=}10\text{–}13`, R`\text{Hungarian: } \hat\sigma=\arg\min_{\sigma}\sum_i\mathcal{L}_{match}(y_i,\hat y_{\sigma(i)})`],
      h: T(['Strategy', 'Rule', 'Used by'], [['Max-IoU', 'IoU>0.7 pos, <0.3 neg', 'RPN, Faster R-CNN'], ['ATSS', 'adaptive IoU threshold from statistics', 'ATSS, PP-YOLOE (early)'], ['SimOTA', 'optimal transport with dynamic-k', 'YOLOX'], ['TAL', 'score^α·IoU^β alignment', 'YOLOv8–26, PP-YOLOE'], ['Hungarian (1-to-1)', 'bipartite matching on cls+L1+GIoU cost', 'DETR family'], ['1-to-many + 1-to-1 (dual)', 'extra many-to-one branch in training for dense supervision', 'YOLOv10, YOLO26, H-DETR, Co-DETR, DEIM']]) }
  ]},
  { id: 'arch', title: 'Architectures & Layer Formations', items: [
    { id: 'rcnn', t: 'Faster R-CNN (+FPN, RoIAlign)', tags: 'faster r-cnn,rpn,roi align,two-stage,cascade r-cnn',
      h: T(['Stage', 'Operation'], [['Backbone + FPN', 'ResNet C2–C5 → P2–P6 (256 ch)'], ['RPN', '3×3 conv → objectness (k) + deltas (4k) per anchor; 3 ratios × 1 scale per level; ~2000 proposals (train) / 1000 (test) after NMS 0.7'], ['RoIAlign', 'bilinear sampling to 7×7 (no quantization), level \\(k=\\lfloor4+\\log_2(\\sqrt{wh}/224)\\rfloor\\)'], ['Box head', '2×FC-1024 → cls (K+1) + class-specific deltas'], ['Sampling', '512 RoIs/img, 25% positive (IoU ≥ 0.5)']]) + R`<p>Cascade R-CNN: 3 heads at IoU 0.5/0.6/0.7 for higher-quality boxes (+3–4 AP). Still the reference for high-precision small-data problems.</p>` },
    { id: 'retina', t: 'RetinaNet / FCOS / CenterNet', tags: 'retinanet,fcos,centernet,one-stage,centerness',
      f: [R`\text{FCOS centerness}=\sqrt{\frac{\min(l,r)}{\max(l,r)}\cdot\frac{\min(t,b)}{\max(t,b)}}`, R`\text{CenterNet heatmap: }Y_{xyc}=\exp\!\left(-\frac{(x-\tilde p_x)^2+(y-\tilde p_y)^2}{2\sigma_p^2}\right)`],
      h: R`<p>RetinaNet: FPN P3–P7, 4×conv subnets (cls/box), 9 anchors, focal loss. FCOS: per-pixel (l,t,r,b), scale ranges per level, centerness. CenterNet: objects as center points + size regression, no NMS (3×3 max-pool peak extraction).</p>` },
    { id: 'yolo', t: 'YOLO evolution v1 → YOLO26', tags: 'yolo,yolov5,yolov8,yolo11,yolov12,yolo26,ultralytics,real-time',
      h: T(['Version', 'Date', 'Key changes'], [['YOLOv1', '2015-06', 'S×S grid regression, single pass'], ['YOLOv2/9000', '2016-12', 'anchors (k-means), BN, high-res fine-tune'], ['YOLOv3', '2018-04', 'Darknet-53, 3 scales, logistic cls'], ['YOLOv4', '2020-04', 'CSPDarknet, SPP, PAN, Mosaic, CIoU, Mish'], ['YOLOv5', '2020-06', 'PyTorch, auto-anchor, SiLU, SPPF, strong engineering'], ['YOLOX', '2021-07', 'anchor-free, decoupled head, SimOTA'], ['YOLOv6 / v7', '2022-06/07', 'RepVGG re-param / E-ELAN, aux heads'], ['YOLOv8', '2023-01', 'anchor-free, C2f, decoupled head, TAL, DFL'], ['YOLOv9', '2024-02', 'GELAN, programmable gradient information (PGI)'], ['YOLOv10', '2024-05', 'NMS-free dual assignment, efficiency-driven design'], ['YOLO11', '2024-09', 'C3k2, C2PSA attention block'], ['YOLOv12', '2025-02', 'area attention (A2), R-ELAN, FlashAttention'], ['YOLO26', '2026-01', 'end-to-end NMS-free, DFL removed, ProgLoss + STAL (small targets), MuSGD optimizer; faster CPU inference']]) },
    { id: 'yololayers', t: 'YOLOv8/11-style layer formation', tags: 'yolo backbone,c2f,c3k2,sppf,pan,decoupled head,dfl,yolo layers',
      h: T(['#', 'Module', 'Out (640 input, "s" width)', 'Stride'], [['0', 'Conv 3×3 s2', '32 × 320²', 'P1/2'], ['1', 'Conv 3×3 s2', '64 × 160²', 'P2/4'], ['2', 'C2f/C3k2 ×n', '64/128 × 160²', ''], ['3', 'Conv s2', '128 × 80²', 'P3/8'], ['4', 'C2f/C3k2 ×2n', '128/256 × 80²', '→ neck'], ['5', 'Conv s2', '256 × 40²', 'P4/16'], ['6', 'C2f/C3k2 ×2n', '256 × 40²', '→ neck'], ['7', 'Conv s2', '512 × 20²', 'P5/32'], ['8', 'C2f/C3k2 ×n', '512 × 20²', ''], ['9', 'SPPF (5×5 maxpool ×3, concat)', '512 × 20²', ''], ['10', '(YOLO11) C2PSA', '512 × 20²', '→ neck'], ['Neck', 'PAN-FPN: upsample+concat+C2f top-down, conv s2+concat+C2f bottom-up', 'P3,P4,P5', ''], ['Head', 'decoupled per level: box branch (4×reg_max DFL bins) + cls branch (K sigmoid)', '', '']]) + R`<p><b>C2f</b>: split input, pass one half through n bottlenecks, concat all intermediate outputs (rich gradient flow). Scaling: n/s/m/l/x vary depth multiple (0.33→1.0) and width multiple (0.25→1.25).</p>` },
    { id: 'detr', t: 'DETR layer formation', tags: 'detr,detection transformer,object queries,set prediction,encoder decoder',
      h: T(['Block', 'Details'], [['Backbone', 'ResNet-50 C5 (2048 × H/32 × W/32) → 1×1 conv to d=256'], ['Encoder ×6', 'self-attn over HW/1024 tokens + 2-D sine PE, FFN 2048'], ['Decoder ×6', 'N=100 learned queries: self-attn → cross-attn to memory → FFN'], ['Heads', 'per query: Linear → K+1 classes ("no object"), 3-layer MLP → normalized (cx,cy,w,h)'], ['Aux losses', 'Hungarian loss after every decoder layer']]) + R`<p>Pure set prediction, no NMS/anchors. Weakness: 500 epochs, poor small objects → Deformable DETR (multi-scale deformable attn, 50 ep), DAB (queries as 4-D anchors), DN-DETR (denoising queries), DINO (contrastive denoising + mixed query selection + look-forward-twice).</p>` },
    { id: 'rtdetr', t: 'RT-DETR / D-FINE / DEIM / RF-DETR', tags: 'rt-detr,d-fine,deim,deimv2,rf-detr,real-time detr',
      h: T(['Model (date)', 'Key idea'], [['RT-DETR (2023-04)', 'efficient hybrid encoder: AIFI (attention on S5 only) + CCFF (conv cross-scale fusion); IoU-aware query selection; tunable decoder layers'], ['RT-DETRv2 (2024-07)', 'bag-of-freebies, discrete sampling for deployment'], ['D-FINE (2024-10)', 'Fine-grained Distribution Refinement (FDR): iterative refinement of edge distributions; GO-LSD self-distillation'], ['DEIM (2024-12)', 'Dense O2O matching (more positives via mosaic) + Matchability-Aware Loss; halves training time'], ['RF-DETR (2025-03, ICLR 2026)', 'DINOv2 backbone + LW-DETR style deformable decoder; weight-sharing NAS gives a family of sizes; first real-time model > 60 AP'], ['DEIMv2 (2025-09)', 'DINOv3 features + Spatial Tuning Adapter; Atto → X (0.5M → 50M params)']]) },
    { id: 'openvocab', t: 'Open-vocabulary & grounding detectors', tags: 'open vocabulary,grounding dino,yolo-world,owl-vit,glip,florence-2,zero-shot detection,yoloe',
      h: T(['Model', 'Mechanism', 'Use'], [['GLIP / Grounding DINO', 'detection as phrase grounding; text-image fusion in neck & decoder', 'zero-shot boxes from text prompts'], ['OWL-ViT / OWLv2', 'CLIP ViT + per-token box heads; self-training on web data', 'zero/one-shot, image-query'], ['YOLO-World / YOLOE', 'CLIP text embeddings as classifier weights (re-parameterizable)', 'real-time open-vocab'], ['Florence-2 / PaliGemma / Qwen-VL', 'VLM outputs boxes as tokens', 'captions + boxes, auto-labelling'], ['SAM 3 (2025)', 'promptable concept segmentation (text/exemplar → all instances)', 'detect + segment + track concepts']]),
      tip: 'Use open-vocab models to auto-label, then train a small closed-set real-time detector on the pseudo-labels (distillation for deployment).' }
  ]},
  { id: 'loss', title: 'Loss Functions (detection)', items: [
    { id: 'focal', t: 'Focal loss', tags: 'focal loss,retinanet,class imbalance,hard negative',
      f: [R`\text{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t`],
      h: R`<p>γ=2, α=0.25; normalize by #positives. Bias prior init π=0.01. Replaced OHEM/hard-negative mining (3:1 neg:pos in SSD).</p>` },
    { id: 'qfl', t: 'Quality Focal Loss / Varifocal Loss', tags: 'qfl,vfl,varifocal,iou-aware classification,gfl',
      f: [R`\text{QFL}(\sigma)=-|y-\sigma|^{\beta}\big[y\log\sigma+(1-y)\log(1-\sigma)\big],\;y=\text{IoU}`, R`\text{VFL}(p,q)=\begin{cases}-q\,[q\log p+(1-q)\log(1-p)] & q>0\\ -\alpha p^\gamma\log(1-p) & q=0\end{cases}`],
      h: R`<p>Class score target = IoU of the predicted box → score reflects localization quality (better NMS ranking). YOLOv8+ BCE with TAL-weighted targets is a similar idea. RT-DETR uses VFL.</p>` },
    { id: 'dfl', t: 'Distribution Focal Loss (DFL)', tags: 'dfl,distribution focal loss,gfl,reg_max,bins',
      f: [R`\hat y=\sum_{i=0}^{n}P(y_i)\,y_i,\;\;\text{DFL}=-\big((y_{i+1}-y)\log S_i+(y-y_i)\log S_{i+1}\big)`],
      h: R`<p>Each box edge is a discrete distribution over n+1 bins (reg_max=16); models edge ambiguity. YOLO26 drops DFL for simpler export. D-FINE refines these distributions across decoder layers (FDR).</p>` },
    { id: 'iouloss', t: 'IoU-based regression losses', tags: 'iou loss,giou loss,diou loss,ciou loss,eiou,siou,wiou,nwd,box loss',
      f: [R`L_{IoU}=1-\text{IoU},\;\;L_{GIoU}=1-\text{GIoU},\;\;L_{CIoU}=1-\text{IoU}+\frac{\rho^2}{c^2}+\alpha v`, R`\text{EIoU: }1-\text{IoU}+\frac{\rho^2(b,b^{gt})}{c^2}+\frac{\rho^2(w,w^{gt})}{C_w^2}+\frac{\rho^2(h,h^{gt})}{C_h^2}`, R`\text{NWD}=\exp\!\left(-\frac{W_2(\mathcal N_a,\mathcal N_b)}{C}\right)\;(\text{tiny objects})`],
      h: R`<p>Scale-invariant, directly optimizes the metric. Typical: CIoU (YOLOv5–v11, weight 7.5), GIoU (DETR, weight 2) + L1 (weight 5).</p>` },
    { id: 'detrloss', t: 'DETR Hungarian loss', tags: 'hungarian loss,bipartite matching,detr loss,matching cost',
      f: [R`\mathcal{L}_{match}=-\mathbb 1_{c_i\ne\varnothing}\hat p_{\sigma(i)}(c_i)+\mathbb 1_{c_i\ne\varnothing}\big[\lambda_{L1}\|b_i-\hat b_{\sigma(i)}\|_1+\lambda_{giou}L_{GIoU}\big]`, R`\mathcal{L}=\sum_{i}\Big[-\log\hat p_{\hat\sigma(i)}(c_i)+\mathbb 1_{c_i\ne\varnothing}\mathcal L_{box}\Big],\;\;\lambda_{L1}{=}5,\lambda_{giou}{=}2`],
      h: R`<p>No-object class weight 0.1 (or focal loss in Deformable/DINO with cost weight 2). Solved with <code>scipy.optimize.linear_sum_assignment</code>.</p>` },
    { id: 'yololoss', t: 'YOLO total loss', tags: 'yolo loss,box gain,cls gain,objectness',
      f: [R`L=\lambda_{box}L_{CIoU}+\lambda_{cls}L_{BCE}+\lambda_{dfl}L_{DFL}\;\;(7.5,\,0.5,\,1.5)`, R`\text{YOLOv5: }+\lambda_{obj}\sum_{l}w_l L_{BCE}^{obj},\;w=[4.0,1.0,0.4]\text{ for P3,P4,P5}`] },
    { id: 'rpnloss', t: 'Two-stage (Faster R-CNN) loss', tags: 'rpn loss,multi-task loss,fast r-cnn loss',
      f: [R`L=\frac1{N_{cls}}\sum_i L_{cls}(p_i,p_i^*)+\lambda\frac{1}{N_{reg}}\sum_i p_i^*\,\text{SmoothL1}(t_i-t_i^*)`] }
  ]},
  { id: 'metrics', title: 'Metrics', items: [
    { id: 'map', t: 'mAP (COCO & VOC)', tags: 'map,mean average precision,coco ap,ap50,ap75,aps,apm,apl,ar',
      f: [R`\text{AP}=\int_0^1 p_{interp}(r)\,dr\approx\frac1{101}\sum_{r\in\{0,.01,..,1\}}p_{interp}(r),\;\;p_{interp}(r)=\max_{\tilde r\ge r}p(\tilde r)`, R`\text{AP}_{COCO}=\frac1{10}\sum_{t\in\{.5,.55,..,.95\}}\text{AP}_t\;\text{averaged over classes}`],
      h: R`<p>AP<sub>S/M/L</sub>: area < 32², 32²–96², > 96². AR@100: max recall with 100 dets/img. VOC mAP = AP@0.5 (11-point pre-2010). Ultralytics "mAP50-95" = COCO AP. LVIS uses AP<sub>r/c/f</sub> (rare/common/frequent).</p>`,
      code: R`
from pycocotools.coco import COCO; from pycocotools.cocoeval import COCOeval
gt = COCO('instances_val2017.json'); dt = gt.loadRes('detections.json')
E = COCOeval(gt, dt, 'bbox'); E.evaluate(); E.accumulate(); E.summarize()` },
    { id: 'speed', t: 'Latency reporting (fair comparison)', tags: 'latency,fps,tensorrt,t4,benchmark fairness',
      h: R`<p>Compare at the same: GPU (T4 is common), precision (FP16 TensorRT), batch size 1, input size, and <b>including</b> NMS for NMS-based models (end-to-end latency). FLOPs and params are hardware-independent but do not predict latency on their own.</p>` }
  ]},
  { id: 'practical', title: 'Practical Tips', items: [
    { id: 'tips', t: 'Detection practitioner notes', tags: 'detection tips,confidence threshold,deployment,label quality',
      h: R`<ul><li>Choose operating threshold on a PR curve for your cost of FP vs FN; mAP is threshold-free.</li><li>Label consistency (tight boxes, occlusion policy) matters more than model choice beyond ~YOLO-s.</li>
        <li>Crowded scenes: raise max detections / queries, use NMS-free or Soft-NMS, repulsion loss.</li><li>Rotated objects (aerial): OBB heads (YOLO-OBB, Oriented R-CNN) with ProbIoU/KLD losses.</li>
        <li>Fine-tuning DETR-likes on few images: freeze backbone first epochs, raise LR warmup, keep EMA.</li><li>Background images (no labels) 0–10% reduce false positives.</li></ul>` }
  ]},
  { id: 'data', title: 'Datasets', items: [
    { id: 'ds', t: 'Detection datasets', tags: 'coco,objects365,lvis,open images,pascal voc,rf100-vl,datasets',
      h: T(['Dataset', 'Size', 'Notes'], [['COCO 2017', '118k train / 5k val, 80 cls, 860k boxes', 'standard benchmark (val2017, test-dev)'], ['Objects365', '1.7M imgs, 365 cls, 30M boxes', 'pre-training (+2–5 AP)'], ['LVIS v1', '164k imgs, 1203 cls', 'long-tail, federated labels'], ['Open Images V7', '1.9M imgs, 600 box classes', 'large, noisier'], ['Pascal VOC', '11k imgs, 20 cls', 'legacy'], ['RF100-VL / ODinW', '100 / 35 domain datasets', 'transfer to real-world domains'], ['CrowdHuman, VisDrone, DOTA', 'crowds / drones / aerial OBB', 'specialized']]) }
  ]}
  ]
});
})();

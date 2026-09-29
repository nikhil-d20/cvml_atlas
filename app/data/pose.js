(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'pose', title: 'Keypoint & Pose Estimation', short: 'Pose', icon: '✦', boards: ['pose-coco'],
  blurb: 'Top-down / bottom-up / one-stage, heatmaps vs SimCC vs regression, HRNet, ViTPose, RTMPose, OKS, 3D lifting.',
  intro: 'Localize K keypoints per instance (COCO: 17 body joints). Also faces (68/98 landmarks), hands (21), whole-body (133), animals.',
  sections: [
  { id: 'form', title: 'Formulations', items: [
    { id: 'paradigms', t: 'Top-down vs bottom-up vs one-stage', tags: 'top-down,bottom-up,one-stage,paradigm',
      h: T(['Paradigm', 'Pipeline', 'Pros / cons', 'Examples'], [['Top-down', 'person detector → crop (256×192) → single-person pose', 'most accurate; cost ∝ #people', 'SimpleBaseline, HRNet, ViTPose, RTMPose'], ['Bottom-up', 'all keypoints + grouping (PAFs / associative embedding)', 'constant cost; weaker on small people', 'OpenPose, HigherHRNet, DEKR'], ['One-stage', 'detector predicts box + keypoints jointly', 'real-time, simple deploy', 'YOLO-pose, RTMO, ED-Pose, DETRPose']]) },
    { id: 'heatmap', t: 'Heatmap encoding & decoding', tags: 'heatmap,gaussian heatmap,dark,udp,sub-pixel,argmax',
      f: [R`H_k(x,y)=\exp\!\left(-\frac{(x-x_k)^2+(y-y_k)^2}{2\sigma^2}\right),\;\sigma=2\;(64{\times}48)`, R`\text{DARK: } \hat\mu=m-\big(\nabla^2\log H(m)\big)^{-1}\nabla\log H(m)`],
      h: R`<p>Classic decode: argmax + ¼-pixel shift toward the higher neighbour. DARK (Taylor expansion) and UDP (unbiased data processing: use \((w-1)/(W-1)\) scaling) remove quantization bias (+1–2 AP). Heatmap stride 4 limits precision at low res.</p>` },
    { id: 'simcc', t: 'SimCC (coordinate classification)', tags: 'simcc,coordinate classification,rtmpose,1d heatmap',
      f: [R`p_x=\mathrm{softmax}(z_x)\in\mathbb R^{W\cdot k},\;\;p_y\in\mathbb R^{H\cdot k},\;\;k=2\;(\text{split ratio})`, R`\text{target: 1-D Gaussian label smoothing},\;\;L=D_{KL}(\tilde y_x\|p_x)+D_{KL}(\tilde y_y\|p_y)`],
      h: R`<p>Treats x and y as separate classification over sub-pixel bins → no 2-D heatmaps, cheap, sub-pixel accurate. Backbone of RTMPose.</p>` },
    { id: 'regress', t: 'Direct regression & RLE', tags: 'regression,residual log-likelihood,rle,normalizing flow',
      f: [R`\mathcal L_{RLE}=-\log P_\phi(\bar\mu)+\log\hat\sigma,\;\;\bar\mu=\frac{\mu_g-\hat\mu}{\hat\sigma}`],
      h: R`<p>Residual Log-likelihood Estimation learns the error distribution with a normalizing flow; brings regression close to heatmap accuracy with far less compute.</p>` }
  ]},
  { id: 'arch', title: 'Architectures & Layer Formations', items: [
    { id: 'openpose', t: 'OpenPose (PAFs)', tags: 'openpose,part affinity fields,paf,bottom-up',
      f: [R`E=\int_{u=0}^{1}L_c\big(p(u)\big)\cdot\frac{d_{j_2}-d_{j_1}}{\|d_{j_2}-d_{j_1}\|}\,du`],
      h: R`<p>Multi-stage CNN predicts confidence maps \(S\) and part-affinity vector fields \(L\); limbs matched by bipartite matching on line-integral scores.</p>` },
    { id: 'hourglass', t: 'Stacked Hourglass & SimpleBaseline', tags: 'hourglass,simplebaseline,deconvolution head',
      h: R`<p><b>Hourglass</b>: repeated down/up-sampling modules with intermediate supervision (8 stacks). <b>SimpleBaseline</b>: ResNet + 3 deconv layers (4×4, 256 ch, stride 2 each) + 1×1 → K heatmaps at 1/4 res. Showed simple designs match complex ones.</p>` },
    { id: 'hrnet', t: 'HRNet-W32 layer formation', tags: 'hrnet,hrnet-w32,high resolution,multi-resolution fusion,pose backbone',
      h: T(['Stage', 'Branches (channels)', 'Modules × blocks'], [['Stem', '2× 3×3 s2 conv → 1/4 res, 64 ch', '—'], ['Stage 1', '1/4: 4 bottleneck blocks (256) → 32', '1 × 4'], ['Stage 2', '1/4 (32), 1/8 (64)', '1 × 4 basic/branch'], ['Stage 3', '+ 1/16 (128)', '4 × 4'], ['Stage 4', '+ 1/32 (256)', '3 × 4'], ['Head', 'high-res branch → 1×1 conv → 17 heatmaps (64×48)', '']]) + R`<p>Fusion after each module: upsample (1×1 conv + nearest) and downsample (3×3 s2 conv) exchange between all branches. W32: 28.5M params, 7.1 GFLOPs @256×192, 74.4 AP. HigherHRNet adds a deconv to 1/2 res + associative embedding for bottom-up.</p>` },
    { id: 'vitpose', t: 'ViTPose', tags: 'vitpose,plain vit,pose transformer,mae',
      h: R`<p>Plain MAE-pretrained ViT (no pyramid) → simple decoder (2 deconv or bilinear ×4 + conv) → heatmaps. Scales B/L/H/G (up to 1B params, 80.9 AP test-dev w/ multi-dataset). ViTPose++ adds task-specific MoE FFNs for multi-species/whole-body.</p>` },
    { id: 'rtmpose', t: 'RTMPose / RTMO / DWPose', tags: 'rtmpose,rtmo,dwpose,real-time pose,cspnext,gau',
      h: T(['Component', 'RTMPose'], [['Backbone', 'CSPNeXt (t/s/m/l/x)'], ['Head', '7×7 conv → FC → Gated Attention Unit (GAU) → SimCC x/y classifiers'], ['Training', 'AdamW 4e-3, cosine, 420 ep, EMA, strong aug in 2 stages, KL loss'], ['Speed', 'RTMPose-m: 75.8 AP, > 90 FPS on i7 CPU (ONNX)']]) + R`<p><b>RTMO</b>: one-stage YOLO-style with dual 1-D heatmaps (coordinate classification) per detection. <b>DWPose</b>: distilled whole-body (133 kpts) RTMPose, widely used for ControlNet pose conditioning.</p>` },
    { id: 'yolopose', t: 'YOLO-pose (Ultralytics)', tags: 'yolo pose,yolov8-pose,yolo11-pose,yolo26-pose,oks loss',
      h: R`<p>Detection head + keypoint branch predicting K×(x, y, visibility) per anchor point; OKS-based loss + BCE on visibility. Real-time multi-person with no grouping step. YOLO26-pose adds RLE-style residual likelihood for keypoint precision.</p>` },
    { id: 'lift3d', t: '3D human pose & mesh', tags: '3d pose,lifting,videopose3d,smpl,hmr,motionbert',
      f: [R`\text{SMPL: } M(\beta,\theta)=W\big(T(\beta,\theta),J(\beta),\theta,\mathcal W\big),\;\beta\in\mathbb R^{10},\theta\in\mathbb R^{72}`],
      h: R`<p>2D→3D lifting (VideoPose3D temporal dilated conv, MotionBERT transformer) or direct mesh recovery (HMR 2.0, 4DHumans; SMPL-X for hands/face). Sapiens (2024) provides human-centric ViT foundation models (pose, seg, depth, normals) at 1K resolution.</p>` }
  ]},
  { id: 'loss', title: 'Loss Functions (pose)', items: [
    { id: 'hmse', t: 'Heatmap MSE (target-weighted)', tags: 'heatmap loss,mse,keypoint mse,target weight',
      f: [R`L=\frac1{K}\sum_{k=1}^{K}v_k\,\big\|\hat H_k-H_k\big\|_2^2`],
      h: R`<p>\(v_k\)=visibility weight (0 for unlabelled). Imbalance of mostly-zero heatmaps → AWing / focal-heatmap variants help.</p>` },
    { id: 'oksloss', t: 'OKS loss', tags: 'oks loss,object keypoint similarity loss,yolo pose loss',
      f: [R`L_{OKS}=1-\frac{\sum_k\exp\!\left(-\frac{d_k^2}{2s^2\kappa_k^2}\right)\delta(v_k>0)}{\sum_k\delta(v_k>0)}`] },
    { id: 'wing', t: 'Wing / Adaptive Wing (landmarks)', tags: 'wing loss,adaptive wing,face landmarks,facial alignment',
      f: [R`\text{Wing}(x)=\begin{cases}w\ln(1+|x|/\epsilon)&|x|<w\\|x|-C&\text{else}\end{cases},\;C=w-w\ln(1+w/\epsilon)`],
      h: R`<p>Amplifies small/medium errors (w=10, ε=2). Adaptive Wing extends it to heatmaps. Face alignment metric: NME normalized by inter-ocular distance.</p>` },
    { id: 'kl', t: 'SimCC KL / RLE / bone losses', tags: 'kl loss,simcc loss,bone length loss,symmetry',
      f: [R`L_{bone}=\sum_{(i,j)\in\mathcal B}\big|\,\|\hat p_i-\hat p_j\|-\|p_i-p_j\|\,\big|`],
      h: R`<p>Structural priors (bone length, symmetry) regularize 3D lifting; 2D top-down usually needs only heatmap/SimCC losses.</p>` }
  ]},
  { id: 'metrics', title: 'Metrics', items: [
    { id: 'oks', t: 'OKS & COCO keypoint AP', tags: 'oks,object keypoint similarity,keypoint ap,coco keypoints',
      f: [R`\text{OKS}=\frac{\sum_k\exp\!\left(-d_k^2/2s^2\kappa_k^2\right)\delta(v_k>0)}{\sum_k\delta(v_k>0)},\;\;\kappa_k=2\sigma_k`],
      h: R`<p>\(s^2\) = object segment area, \(\sigma_k\) per-keypoint falloff (eyes 0.025 … hips 0.107). AP averaged over OKS thresholds .50:.05:.95 — the pose analogue of box AP.</p>` },
    { id: 'pck', t: 'PCK / PCKh / MPJPE', tags: 'pck,pckh,mpjpe,pa-mpjpe,3d pose metric',
      f: [R`\text{PCK@}\alpha=\frac1K\sum_k\mathbb 1\!\left[\|\hat p_k-p_k\|\le\alpha\cdot L\right]\;(\text{PCKh: }L=\text{head size},\alpha=0.5)`, R`\text{MPJPE}=\frac1K\sum_k\|\hat P_k-P_k\|_2\;\text{(mm, root-aligned)};\;\;\text{PA-MPJPE after Procrustes}`] }
  ]},
  { id: 'practical', title: 'Practical Tips', items: [
    { id: 'tips', t: 'Pose practitioner notes', tags: 'pose tips,flip test,keypoint flip,detector quality,smoothing',
      h: R`<ul><li>Horizontal flip must swap left/right keypoint indices (<code>flip_idx</code>) — a classic silent bug.</li><li>Top-down accuracy is capped by the person detector; evaluate with GT boxes to isolate pose error.</li>
        <li>Flip-test (average heatmaps of image and flipped image) = +1 AP.</li><li>Video: temporal smoothing (One-Euro filter) removes jitter; or use tracking-by-pose.</li>
        <li>Custom skeletons: define σ per keypoint for OKS (estimate from annotation variance).</li><li>Occlusion: train with visibility flags; synthetic occluders (random erasing on limbs).</li></ul>` }
  ]},
  { id: 'data', title: 'Datasets', items: [
    { id: 'ds', t: 'Pose datasets', tags: 'coco keypoints,mpii,crowdpose,h36m,cocowholebody,datasets',
      h: T(['Dataset', 'Size', 'Notes'], [['COCO Keypoints', '57k imgs / 150k persons, 17 kpts', 'standard 2D benchmark'], ['COCO-WholeBody', '133 kpts (body, feet, face, hands)', 'whole-body'], ['MPII', '25k imgs, 16 kpts', 'PCKh'], ['CrowdPose / OCHuman', 'crowded / occluded', 'robustness'], ['AI Challenger, Halpe', 'large/extra kpts', 'pre-training'], ['Human3.6M / 3DPW', '3.6M frames mocap / in-the-wild 3D', '3D pose'], ['AP-10K / AnimalKingdom', 'animals', 'cross-species']]) }
  ]}
  ]
});
})();

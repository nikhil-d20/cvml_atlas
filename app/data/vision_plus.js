(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'vplus', title: 'More Vision: SSL, Tracking, Depth, 3D, Video, OCR', short: 'More Vision', icon: '◈',
  blurb: 'Self-supervised learning, multi-object tracking, depth, NeRF & Gaussian splatting, video, OCR, anomaly detection, classic CV.',
  intro: 'Everything else a CV engineer runs into: pre-training without labels, tracking, geometry, and the classical toolbox.',
  sections: [
  { id: 'ssl', title: 'Self-Supervised Learning', items: [
    { id: 'contrast', t: 'SimCLR / MoCo', tags: 'simclr,moco,contrastive learning,memory queue,momentum encoder',
      f: [R`\ell_{i,j}=-\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k\ne i}^{2N}\exp(\text{sim}(z_i,z_k)/\tau)}`, R`\text{MoCo: }\theta_k\leftarrow m\theta_k+(1-m)\theta_q,\;m=0.999`],
      h: R`<p>SimCLR: strong aug (crop + color jitter + blur), projection MLP, batch 4096, LARS. MoCo: queue of 65k negatives with a momentum key encoder → small batches OK.</p>` },
    { id: 'byol', t: 'BYOL / DINO (self-distillation)', tags: 'byol,dino,self-distillation,teacher student,centering,no negatives',
      f: [R`\text{DINO: } \mathcal L=-\sum P_t(x)\log P_s(x'),\;\;P_t=\mathrm{softmax}\big((g_t(x)-c)/\tau_t\big)`, R`c\leftarrow mc+(1-m)\tfrac1B\sum g_t(x_i),\;\;\tau_t=0.04,\;\tau_s=0.1`],
      h: R`<p>No negatives; collapse avoided by teacher centering + sharpening and EMA teacher. Multi-crop (2×224 global + 10×96 local). DINO attention maps segment objects without labels.</p>` },
    { id: 'mae', t: 'MAE / iBOT / DINOv2 / DINOv3 / I-JEPA', tags: 'mae,masked image modeling,ibot,dinov2,dinov3,jepa,gram anchoring',
      f: [R`\mathcal L_{MAE}=\frac1{|M|}\sum_{i\in M}\|x_i-\hat x_i\|^2\;(\text{normalized pixels, mask }75\%)`, R`\text{Gram anchoring (DINOv3): } \mathcal L=\|X_sX_s^\top-X_gX_g^\top\|_F^2`],
      h: T(['Method', 'Target', 'Note'], [['MAE', 'pixels of masked patches', 'encoder sees only 25% tokens → 3× faster'], ['iBOT', 'teacher tokens of masked patches (+DINO CLS)', 'dense features'], ['DINOv2', 'DINO + iBOT + KoLeo, curated data', 'frozen features SOTA (2023)'], ['DINOv3', '7B ViT + Gram anchoring + high-res adaptation', 'clean dense features at scale (2025)'], ['I-JEPA / V-JEPA 2', 'predict latent embeddings of masked regions', 'no pixel reconstruction; video world-model (V-JEPA 2)']]) }
  ]},
  { id: 'track', title: 'Multi-Object Tracking', items: [
    { id: 'kalman', t: 'Kalman filter (SORT motion model)', tags: 'kalman filter,sort,motion model,prediction',
      f: [R`\hat x_{k|k-1}=F\hat x_{k-1},\;\;P_{k|k-1}=FP_{k-1}F^\top+Q`, R`K_k=P_{k|k-1}H^\top(HP_{k|k-1}H^\top+R)^{-1}`, R`\hat x_k=\hat x_{k|k-1}+K_k(z_k-H\hat x_{k|k-1}),\;\;P_k=(I-K_kH)P_{k|k-1}`],
      h: R`<p>State \([u,v,s,r,\dot u,\dot v,\dot s]\) (SORT) or \([x,y,a,h,\dots]\) (DeepSORT). Association: Hungarian on IoU / Mahalanobis / appearance cosine.</p>` },
    { id: 'trackers', t: 'Tracker family', tags: 'deepsort,bytetrack,bot-sort,oc-sort,motr,samurai,tracking by detection',
      h: T(['Tracker', 'Key idea'], [['SORT', 'Kalman + IoU Hungarian'], ['DeepSORT', '+ Re-ID appearance embedding, cascade matching'], ['ByteTrack', 'associate low-score detections in a 2nd pass (recover occluded)'], ['OC-SORT', 'observation-centric re-update, fixes Kalman drift in non-linear motion'], ['BoT-SORT', 'camera-motion compensation + Re-ID + improved KF'], ['MOTR / MOTRv2', 'end-to-end transformer track queries'], ['SAM 2 / SAMURAI', 'mask propagation with memory (single/multi-object video)']]),
      tip: 'Tracking quality is dominated by detector recall at low thresholds; ByteTrack with a strong detector is a hard-to-beat baseline. Tune track_buffer (frames to keep lost tracks) to frame rate.' },
    { id: 'motmetrics', t: 'MOTA, IDF1, HOTA', tags: 'mota,idf1,hota,tracking metrics',
      f: [R`\text{MOTA}=1-\frac{\sum_t(FN_t+FP_t+IDSW_t)}{\sum_tGT_t}`, R`\text{IDF1}=\frac{2\,IDTP}{2\,IDTP+IDFP+IDFN}`, R`\text{HOTA}_\alpha=\sqrt{\text{DetA}_\alpha\cdot\text{AssA}_\alpha}`] }
  ]},
  { id: 'depth', title: 'Depth & Geometry', items: [
    { id: 'camera', t: 'Pinhole camera & projection', tags: 'camera model,intrinsics,extrinsics,projection matrix,calibration',
      f: [R`s\begin{bmatrix}u\\v\\1\end{bmatrix}=K[R\,|\,t]\begin{bmatrix}X\\Y\\Z\\1\end{bmatrix},\;\;K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}`, R`\text{radial distortion: } x_d=x(1+k_1r^2+k_2r^4+k_3r^6)`] },
    { id: 'epipolar', t: 'Epipolar geometry & stereo', tags: 'epipolar,fundamental matrix,essential matrix,stereo,disparity,homography',
      f: [R`x'^\top F x=0,\;\;E=K'^\top FK=[t]_\times R`, R`Z=\frac{f\,B}{d}\;(\text{disparity }d)`, R`\text{Homography: } x'\sim Hx,\;H\in\mathbb R^{3\times3}\;(8\text{ DoF, 4 points})`],
      h: R`<p>Estimate with RANSAC (8-point / 5-point / 4-point algorithms). Depth error grows ∝ \(Z^2\).</p>` },
    { id: 'mono', t: 'Monocular depth estimation', tags: 'monocular depth,depth anything,midas,scale-invariant loss,metric depth,marigold',
      f: [R`\mathcal L_{SI}=\frac1n\sum_id_i^2-\frac{\lambda}{n^2}\Big(\sum_id_i\Big)^2,\;\;d_i=\log\hat y_i-\log y_i`, R`\text{Affine-invariant: } \hat d^*=s\hat d+t,\;(s,t)=\arg\min\sum(s\hat d_i+t-d_i)^2`, R`\text{AbsRel}=\frac1n\sum\frac{|\hat y-y|}{y},\;\;\delta_1=\%\Big(\max(\tfrac{\hat y}{y},\tfrac{y}{\hat y})<1.25\Big)`],
      h: R`<p>Relative depth: MiDaS, Depth Anything V1/V2 (DINOv2 + DPT decoder, pseudo-labelled 62M images), Marigold (diffusion). Metric depth: ZoeDepth, UniDepth, Metric3D v2, Depth Pro, Depth Anything 3 (2025, any-view geometry).</p>` },
    { id: 'nerf', t: 'NeRF volume rendering', tags: 'nerf,neural radiance field,volume rendering,positional encoding',
      f: [R`\hat C(r)=\sum_{i=1}^{N}T_i\big(1-e^{-\sigma_i\delta_i}\big)c_i,\;\;T_i=\exp\Big(-\sum_{j<i}\sigma_j\delta_j\Big)`, R`\gamma(p)=\big(\sin(2^0\pi p),\cos(2^0\pi p),\dots,\sin(2^{L-1}\pi p),\cos(2^{L-1}\pi p)\big)`],
      h: R`<p>MLP \((x,d)\to(\sigma,c)\), hierarchical sampling. Instant-NGP: multi-resolution hash grids → seconds to train.</p>` },
    { id: 'gs', t: '3D Gaussian Splatting', tags: 'gaussian splatting,3dgs,splatting,real-time rendering',
      f: [R`G(x)=e^{-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)},\;\;\Sigma=RSS^\top R^\top`, R`C=\sum_{i}c_i\,\alpha_i\prod_{j<i}(1-\alpha_j),\;\;\Sigma'=JW\Sigma W^\top J^\top`],
      h: R`<p>Millions of anisotropic Gaussians (position, covariance, opacity, SH colour) rasterized with tile-based α-blending; adaptive densification/pruning. Real-time (100+ FPS) novel-view synthesis; loss = L1 + 0.2·D-SSIM.</p>` },
    { id: 'feedforward3d', t: 'Feed-forward 3D reconstruction', tags: 'dust3r,mast3r,vggt,feed-forward 3d,pointmap,sfm',
      h: R`<p>DUSt3R/MASt3R regress per-pixel pointmaps from image pairs (no calibration). VGGT (CVPR 2025 best paper) predicts cameras, depth, pointmaps and tracks for many views in one forward pass — a learned replacement for much of COLMAP-style SfM.</p>` },
    { id: 'flow', t: 'Optical flow', tags: 'optical flow,raft,brightness constancy,epe',
      f: [R`I(x+u,y+v,t+1)=I(x,y,t)\Rightarrow I_xu+I_yv+I_t=0`, R`\text{EPE}=\|\hat f-f\|_2`],
      h: R`<p>RAFT: all-pairs 4-D correlation volume + recurrent GRU updates (loss with γ=0.8 decay over iterations). Successors: GMFlow, SEA-RAFT, video point tracking (CoTracker, TAPIR).</p>` }
  ]},
  { id: 'video', title: 'Video Understanding', items: [
    { id: 'videomodels', t: 'Action recognition architectures', tags: 'action recognition,i3d,slowfast,timesformer,videomae,video classification',
      h: T(['Model', 'Idea'], [['Two-stream', 'RGB + optical flow CNNs'], ['C3D / I3D', '3-D convs; I3D inflates ImageNet 2-D kernels to 3-D'], ['R(2+1)D', 'factorize 3-D conv into 2-D spatial + 1-D temporal'], ['SlowFast', 'slow path (low fps, many channels) + fast path (high fps, β=1/8 channels)'], ['TimeSformer / ViViT', 'divided space-time attention'], ['VideoMAE / V2', 'tube masking 90–95%, MAE pre-training'], ['InternVideo2 / VLM-based', 'video-text foundation models, zero-shot']]) }
  ]},
  { id: 'ocr', title: 'OCR & Documents', items: [
    { id: 'ctc', t: 'CTC loss & CRNN', tags: 'ctc,ocr,crnn,text recognition,connectionist temporal classification',
      f: [R`p(y|x)=\sum_{\pi\in\mathcal B^{-1}(y)}\prod_{t=1}^{T}p(\pi_t|x),\;\;\mathcal L_{CTC}=-\log p(y|x)`],
      h: R`<p>\(\mathcal B\) collapses repeats and removes blanks. CRNN = CNN → BiLSTM → CTC. Text detection: DBNet (differentiable binarization), CRAFT. End-to-end: TrOCR, PaddleOCR, Donut (OCR-free), and VLM-based OCR (Qwen-VL, olmOCR, dots.ocr, DeepSeek-OCR) now lead document parsing.</p>` },
    { id: 'ocrmetric', t: 'CER / WER', tags: 'cer,wer,edit distance,ocr metric',
      f: [R`\text{CER}=\frac{S+D+I}{N}\;\text{(character level)},\;\;\text{WER analogously on words}`] }
  ]},
  { id: 'anomaly', title: 'Anomaly Detection & Re-ID', items: [
    { id: 'patchcore', t: 'Industrial anomaly detection', tags: 'anomaly detection,patchcore,padim,mvtec,defect detection',
      f: [R`s(x)=\max_{p\in P(x)}\;\min_{m\in\mathcal M}\|p-m\|_2\;\;(\text{PatchCore})`],
      h: R`<p>Train on normal images only. PatchCore: memory bank of mid-level patch features (coreset-subsampled), nearest-neighbour distance = anomaly score. PaDiM: per-patch Gaussian + Mahalanobis. Metrics: image AUROC, pixel AUROC, PRO. Benchmarks: MVTec AD, VisA, MVTec AD 2.</p>` },
    { id: 'reid', t: 'Person / vehicle re-identification', tags: 're-id,reid,person re-identification,cmc,metric learning',
      h: R`<p>Backbone + BNNeck; loss = ID CE (label smoothing) + triplet (hard mining). Metrics: CMC Rank-1 and mAP. Datasets: Market-1501, MSMT17, VeRi-776.</p>` }
  ]},
  { id: 'classic', title: 'Classical Computer Vision', items: [
    { id: 'filters', t: 'Filtering & edges', tags: 'gaussian blur,sobel,canny,laplacian,convolution kernel,edge detection',
      f: [R`G_\sigma(x,y)=\frac1{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}`, R`S_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\;\;|\nabla I|=\sqrt{G_x^2+G_y^2},\;\theta=\operatorname{atan2}(G_y,G_x)`, R`\text{LoG}=\nabla^2G_\sigma`],
      h: R`<p>Canny: Gaussian → gradient → non-max suppression → double threshold (ratio 1:2–1:3) → hysteresis.</p>` },
    { id: 'features', t: 'Keypoints & descriptors', tags: 'harris,sift,orb,superpoint,lightglue,feature matching',
      f: [R`M=\sum w\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix},\;\;R=\det M-k(\operatorname{tr}M)^2,\;k\in[0.04,0.06]`],
      h: R`<p>SIFT (DoG extrema, 128-D gradient histograms, ratio test 0.8), ORB (FAST + rotated BRIEF, binary). Learned: SuperPoint + SuperGlue/LightGlue, LoFTR (detector-free), RoMa (dense).</p>` },
    { id: 'morph', t: 'Histogram, thresholding & morphology', tags: 'histogram equalization,clahe,otsu,morphology,erosion,dilation',
      f: [R`\text{HE: } s_k=(L-1)\sum_{j\le k}p_r(r_j)`, R`\text{Otsu: }\max_t\;\sigma_B^2(t)=\omega_0\omega_1(\mu_0-\mu_1)^2`, R`\text{Opening}=(A\ominus B)\oplus B,\;\;\text{Closing}=(A\oplus B)\ominus B`] },
    { id: 'color', t: 'Color spaces & image formation', tags: 'color space,rgb,hsv,lab,ycbcr,gamma,bayer',
      h: R`<p>sRGB gamma ≈ 2.2 (linearize before physics-based ops). HSV/HSL for hue-based segmentation; Lab for perceptual distances (ΔE); YCbCr for compression. Raw sensor → demosaic (Bayer) → white balance → CCM → gamma → JPEG: ISP differences cause domain shift between cameras.</p>` }
  ]}
  ]
});
})();

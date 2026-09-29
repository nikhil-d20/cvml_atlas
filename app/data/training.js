(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'train', title: 'Training Toolkit', short: 'Training', icon: '⚙',
  blurb: 'Optimizers, LR schedulers, general loss functions, regularization, augmentation, metrics, distributed training.',
  intro: 'Task-agnostic training machinery. Task-specific losses live in each task tab (search "loss" to see them all).',
  sections: [
  { id: 'optim', title: 'Optimizers', intro: 'Notation: \\(g_t=\\nabla_\\theta L\\), LR \\(\\eta\\), weight decay \\(\\lambda\\).', items: [
    { id: 'sgd', t: 'SGD + Momentum / Nesterov', tags: 'sgd,momentum,nesterov,optimizer',
      f: [R`v_t=\mu v_{t-1}+g_t,\;\;\theta_t=\theta_{t-1}-\eta v_t`, R`\text{Nesterov: } v_t=\mu v_{t-1}+\nabla L(\theta_{t-1}-\eta\mu v_{t-1})`],
      h: R`<p>\(\mu=0.9\) (0.937 in YOLO). Typical CNN: LR 0.1 @ batch 256, WD 1e-4 (5e-4 small data). Often generalizes slightly better than Adam for ConvNets; needs more LR tuning.</p>` },
    { id: 'adam', t: 'Adam', tags: 'adam,adaptive,optimizer,bias correction',
      f: [R`m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\;\;v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2`, R`\hat m_t=\frac{m_t}{1-\beta_1^t},\;\hat v_t=\frac{v_t}{1-\beta_2^t},\;\;\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}`],
      h: R`<p>Defaults \(\beta=(0.9,0.999),\epsilon=10^{-8}\). For large transformers use \(\beta_2=0.95\) (more responsive, fewer loss spikes). L2 penalty inside Adam is <i>not</i> weight decay → use AdamW.</p>` },
    { id: 'adamw', t: 'AdamW (decoupled weight decay)', tags: 'adamw,weight decay,optimizer,transformer',
      f: [R`\theta_t=\theta_{t-1}-\eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\,\theta_{t-1}\right)`],
      h: R`<p>Default for ViT/DETR/diffusion/fine-tuning. Typical: LR 1e-4…1e-3 (from scratch, batch 1024: 1e-3 ×bs/1024), WD 0.05 (ViT), 1e-4 (DETR), 0.01 (fine-tune). Exclude bias, norm and position/cls tokens from WD.</p>`,
      code: R`
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    (no_decay if p.ndim <= 1 or n.endswith('.bias') or 'pos_embed' in n or 'cls_token' in n else decay).append(p)
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.05},
                         {'params': no_decay, 'weight_decay': 0.0}], lr=1e-3, betas=(0.9, 0.999))` },
    { id: 'rmsprop', t: 'AdaGrad / RMSProp', tags: 'adagrad,rmsprop,optimizer',
      f: [R`\text{AdaGrad: } G_t=G_{t-1}+g_t^2,\;\theta_t=\theta_{t-1}-\frac{\eta}{\sqrt{G_t}+\epsilon}g_t`, R`\text{RMSProp: } v_t=\rho v_{t-1}+(1-\rho)g_t^2,\;\theta_t=\theta_{t-1}-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t`],
      h: R`<p>RMSProp (ρ=0.9, ε=1e-3 TF-style) trained EfficientNet/MobileNet originally. Mostly superseded by AdamW.</p>` },
    { id: 'lamb', t: 'LARS / LAMB (large batch)', tags: 'lars,lamb,large batch,layer-wise trust ratio',
      f: [R`\theta^{(l)}_t=\theta^{(l)}_{t-1}-\eta\,\frac{\phi(\|\theta^{(l)}\|)}{\|u^{(l)}\|}\,u^{(l)}`],
      h: R`<p>Layer-wise trust ratio keeps update/weight ratio uniform across layers. LARS (SGD-based) for SimCLR/BYOL with batch 4k–32k; LAMB (Adam-based) for BERT/large ViT batches.</p>` },
    { id: 'lion', t: 'Lion', tags: 'lion,sign,optimizer,memory efficient',
      f: [R`u_t=\mathrm{sign}\big(\beta_1m_{t-1}+(1-\beta_1)g_t\big),\;\;\theta_t=\theta_{t-1}-\eta(u_t+\lambda\theta_{t-1})`, R`m_t=\beta_2m_{t-1}+(1-\beta_2)g_t\;\;(\beta_1{=}0.9,\beta_2{=}0.99)`],
      h: R`<p>One state buffer (half Adam's memory). Update has larger norm → use LR 3–10× <b>smaller</b> and WD 3–10× <b>larger</b> than AdamW.</p>` },
    { id: 'muon', t: 'Muon (orthogonalized momentum)', tags: 'muon,newton-schulz,optimizer,2d weights',
      f: [R`B_t=\mu B_{t-1}+G_t,\;\;O_t=\mathrm{NS}_5(B_t)\approx UV^\top\;(B_t=U\Sigma V^\top),\;\;W_t=W_{t-1}-\eta\,O_t`],
      h: R`<p>Newton–Schulz iterations orthogonalize the update for 2-D weight matrices; embeddings, heads, norms and biases stay on AdamW. Popular since 2025 for LLM pre-training (sample-efficiency gains); still being validated for vision.</p>` },
    { id: 'sam', t: 'SAM (Sharpness-Aware Minimization)', tags: 'sam,flat minima,generalization,asam',
      f: [R`\min_\theta\max_{\|\epsilon\|\le\rho}L(\theta+\epsilon),\;\;\hat\epsilon=\rho\frac{g}{\|g\|}`],
      h: R`<p>Two forward/backward passes per step (2× cost). ρ≈0.05 (SGD), 0.05–2 for ViT. Helps ViTs without heavy aug and label-noise settings.</p>` },
    { id: 'schedfree', t: 'Schedule-Free AdamW / SGD', tags: 'schedule-free,no scheduler,averaging',
      f: [R`y_t=(1-\beta)z_t+\beta x_t,\;\;z_{t+1}=z_t-\eta\nabla L(y_t),\;\;x_{t+1}=(1-c_{t+1})x_t+c_{t+1}z_{t+1}`],
      h: R`<p>Replaces the LR schedule with iterate averaging (Defazio et al., 2024); only warmup needed; no need to know total steps. Call <code>opt.eval()</code> before validation.</p>` },
    { id: 'ema', t: 'EMA of weights', tags: 'ema,model ema,polyak averaging,mean teacher',
      f: [R`\theta_{EMA}\leftarrow d\,\theta_{EMA}+(1-d)\,\theta,\;\;d=0.9998\text{–}0.9999`, R`\text{YOLO ramp: } d_t=d\,(1-e^{-t/2000})`],
      h: R`<p>Evaluate/export the EMA weights. Standard in YOLO, DETR variants, diffusion (d=0.9999), DeiT, self-supervised teachers (momentum 0.996→1).</p>` },
    { id: 'optcmp', t: 'Optimizer selection table', tags: 'which optimizer,optimizer comparison',
      h: T(['Scenario', 'Optimizer', 'LR (per 256 imgs unless noted)', 'WD'], [
        ['ResNet/ConvNet from scratch', 'SGD m=0.9 nesterov', '0.1', '1e-4 – 5e-5'],
        ['ViT / Swin / ConvNeXt from scratch', 'AdamW (0.9,0.999)', '5e-4×bs/512', '0.05'],
        ['Fine-tune pretrained backbone', 'AdamW', '1e-5 – 1e-4 (+LLRD 0.65–0.8)', '0.01–0.05'],
        ['YOLO (Ultralytics)', 'SGD 0.01 / AdamW auto', 'lr0 0.01 (SGD), 0.001–0.002 (AdamW)', '5e-4'],
        ['DETR family', 'AdamW', '1e-4 (backbone 1e-5), bs 16–32', '1e-4'],
        ['Diffusion / DiT', 'AdamW (0.9,0.999/0.95)', '1e-4 const (bs 256)', '0–0.01'],
        ['Contrastive SSL, huge batch', 'LARS / LAMB', '0.3×bs/256', '1e-6 – 1e-4'],
        ['Memory-bound large models', 'Lion / Adafactor / 8-bit AdamW', 'Lion: AdamW/3–10', 'Lion: ×3–10']]) }
  ]},
  { id: 'sched', title: 'Learning Rate Schedulers', intro: '\\(t\\): step, \\(T\\): total steps, \\(\\eta_{max}\\): peak LR.', items: [
    { id: 'warmup', t: 'Linear warmup', tags: 'warmup,lr warmup,stability',
      f: [R`\eta_t=\eta_{max}\cdot\frac{t}{T_w},\;\;t<T_w`],
      h: R`<p>Adam's \(v_t\) estimates are noisy early; large batches/transformers diverge without warmup. Typical: 1–5 epochs (ViT 5–20), DETR ~0, YOLO 3 epochs (with warmup bias LR 0.1 & momentum 0.8→0.937).</p>` },
    { id: 'step', t: 'Step / MultiStep decay', tags: 'step decay,multistep,piecewise',
      f: [R`\eta_t=\eta_0\,\gamma^{\lfloor t/s\rfloor}\;\;\text{or}\;\;\eta_0\,\gamma^{|\{m_i\le t\}|}`],
      h: R`<p>Classic: ×0.1 at epochs 30/60/90 (ResNet), Detectron "1×" = 90k iters, drop at 60k/80k. DETR: drop ×0.1 at epoch 200 (of 300/500).</p>` },
    { id: 'cosine', t: 'Cosine annealing', tags: 'cosine,cosine decay,annealing',
      f: [R`\eta_t=\eta_{min}+\tfrac12(\eta_{max}-\eta_{min})\left(1+\cos\frac{\pi t}{T}\right)`],
      h: R`<p>Default for most modern training. \(\eta_{min}\) ≈ 0–1e-6 (or 1% of max, YOLO lrf=0.01). Needs known \(T\).</p>`,
      code: R`
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
warm = LinearLR(opt, start_factor=1e-3, total_iters=warmup_steps)
cos  = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
sched = SequentialLR(opt, [warm, cos], milestones=[warmup_steps])   # call sched.step() every iteration` },
    { id: 'sgdr', t: 'Cosine with warm restarts (SGDR)', tags: 'sgdr,restarts,snapshot ensembles',
      f: [R`\eta_t=\eta_{min}+\tfrac12(\eta_{max}-\eta_{min})\left(1+\cos\frac{\pi T_{cur}}{T_i}\right),\;\;T_{i+1}=T_{mult}T_i`],
      h: R`<p>Restarts escape sharp minima; checkpoints at each cycle end make snapshot ensembles.</p>` },
    { id: 'onecycle', t: 'One-Cycle', tags: 'one cycle,super convergence,lr range test',
      f: [R`\eta: \tfrac{\eta_{max}}{25}\nearrow\eta_{max}\;(30\%)\searrow\tfrac{\eta_{max}}{25\cdot10^4},\;\;\;\mu: 0.95\searrow0.85\nearrow0.95`],
      h: R`<p>Fast convergence for short budgets. Pick \(\eta_{max}\) via <b>LR range test</b>: increase LR exponentially over ~100 iters; choose ~1/10 of the LR where loss is minimal (just before it explodes).</p>` },
    { id: 'poly', t: 'Polynomial decay', tags: 'poly,polynomial,segmentation schedule',
      f: [R`\eta_t=\eta_0\left(1-\frac{t}{T}\right)^{p},\;\;p=0.9\;(\text{p=1 linear})`],
      h: R`<p>Standard in semantic segmentation (DeepLab, mmseg: 80k/160k iters, SGD 0.01 or AdamW 6e-5 for SegFormer).</p>` },
    { id: 'wsd', t: 'Warmup-Stable-Decay (WSD / trapezoidal)', tags: 'wsd,constant lr,trapezoid,cooldown',
      f: [R`\eta_t=\begin{cases}\eta_{max}t/T_w & t<T_w\\ \eta_{max} & T_w\le t<T-T_d\\ \eta_{max}\,f\!\left(\tfrac{T-t}{T_d}\right) & \text{cooldown }(T_d\approx10\text{–}20\%T)\end{cases}`],
      h: R`<p>Allows continuing training and branching cooldowns at any point; matches cosine with a 1-sqrt or linear cooldown. Popular for LLM and large-scale pre-training.</p>` },
    { id: 'isqrt', t: 'Inverse square-root / exponential', tags: 'inverse sqrt,noam,exponential decay',
      f: [R`\eta_t=d^{-0.5}\min(t^{-0.5},\,t\,T_w^{-1.5})\;\;(\text{Noam})`, R`\eta_t=\eta_0\gamma^{t},\;\gamma\approx0.97\text{/epoch (EfficientNet: }0.97\text{ per 2.4 ep)}`] },
    { id: 'plateau', t: 'ReduceLROnPlateau', tags: 'plateau,adaptive schedule,patience',
      h: R`<p>Multiply LR by factor (0.1–0.5) when val metric stalls for <i>patience</i> epochs. Good for unknown budgets, small datasets and medical imaging; not reproducible across runs; avoid with noisy val metrics.</p>` },
    { id: 'llrd', t: 'Layer-wise LR decay (LLRD) & differential LR', tags: 'llrd,layer decay,discriminative lr,fine-tuning',
      f: [R`\eta_l=\eta\cdot\alpha^{L-l},\;\;\alpha\in[0.65,0.9]`],
      h: R`<p>Lower layers (generic features) change less. Used in BEiT/MAE/DINOv2/EVA fine-tuning (α=0.65–0.75 base, 0.8–0.9 large). Simpler variant: backbone LR = head LR ×0.1 (DETR).</p>` },
    { id: 'schedcmp', t: 'Scheduler selection table', tags: 'which scheduler,scheduler comparison',
      h: T(['Situation', 'Scheduler'], [
        ['Known budget, general case', 'linear warmup + cosine to ~0'],
        ['Short fine-tune (≤ 20 ep)', 'warmup 5–10% + cosine or linear decay'],
        ['Unknown budget / continual training', 'WSD (constant + cooldown), schedule-free, or plateau'],
        ['Segmentation (iteration-based)', 'poly p=0.9 with 1.5k-iter warmup'],
        ['Detectron-style detection', 'multistep ×0.1 at ~67% and ~89%'],
        ['YOLO', 'linear (lrf=0.01) or cosine, 3-epoch warmup, close-mosaic last 10 ep'],
        ['Very short time budget', 'One-Cycle'],
        ['Diffusion', 'constant after warmup (1–10k steps) + EMA']]) }
  ]},
  { id: 'loss', title: 'Loss Functions (general)', intro: 'Task-specific losses: see Classification, Detection, Segmentation, Pose and GenAI tabs.', items: [
    { id: 'mse', t: 'MSE / MAE / Huber (Smooth L1)', tags: 'regression loss,l2,l1,huber,smooth l1',
      f: [R`\text{MSE}=\tfrac1n\sum(y-\hat y)^2,\;\;\text{MAE}=\tfrac1n\sum|y-\hat y|`, R`\text{Huber}_\delta(r)=\begin{cases}\tfrac12r^2 & |r|\le\delta\\ \delta(|r|-\tfrac12\delta) & \text{else}\end{cases},\;\;\text{SmoothL1}_\beta=\text{Huber}_\beta/\beta`],
      h: R`<p>L2 = Gaussian noise, outlier-sensitive; L1 = median, robust; Huber combines both (Fast R-CNN box regression, β=1/9 in Detectron).</p>` },
    { id: 'ce', t: 'Cross-entropy & Binary CE', tags: 'cross entropy,bce,log loss,nll',
      f: [R`\text{CE}=-\sum_{c}y_c\log p_c,\;\;\text{BCE}=-[y\log\sigma(z)+(1-y)\log(1-\sigma(z))]`, R`\text{weighted: } -w_{y}\log p_y,\;\;w_c\propto 1/\text{freq}_c\;\text{ or }\;\tfrac{1-\beta}{1-\beta^{n_c}}`] },
    { id: 'ls', t: 'Label smoothing', tags: 'label smoothing,regularization,calibration',
      f: [R`y^{LS}_c=(1-\varepsilon)\,y_c+\frac{\varepsilon}{K}`],
      h: R`<p>ε=0.1 standard for ImageNet. Improves calibration/accuracy but hurts knowledge-distillation teachers and feature transfer slightly.</p>` },
    { id: 'focal', t: 'Focal loss', tags: 'focal loss,class imbalance,retinanet,hard examples',
      f: [R`\text{FL}(p_t)=-\alpha_t(1-p_t)^{\gamma}\log(p_t),\;\;p_t=\begin{cases}p & y=1\\1-p&y=0\end{cases}`],
      h: R`<p>γ=2, α=0.25 (RetinaNet). Down-weights easy negatives. Details and variants (QFL, VFL) in <a href="#/det">Detection › Loss Functions</a>.</p>` },
    { id: 'kd', t: 'Knowledge distillation (KL)', tags: 'distillation,kd,teacher student,kl',
      f: [R`L=(1-\alpha)\,\text{CE}(y,p_s)+\alpha\,\tau^2\,D_{KL}\!\left(\sigma(z_t/\tau)\,\|\,\sigma(z_s/\tau)\right)`],
      h: R`<p>τ=2–4, α=0.5–0.9. \(\tau^2\) keeps gradient scale. Feature distillation (FitNets), DeiT distillation token, and "patient & consistent" (same aug for teacher & student, long schedules) work best.</p>` },
    { id: 'infonce', t: 'Contrastive (InfoNCE / NT-Xent)', tags: 'contrastive,infonce,nt-xent,simclr,clip,moco',
      f: [R`\mathcal{L}_i=-\log\frac{\exp(\text{sim}(z_i,z_i^+)/\tau)}{\sum_{k\neq i}\exp(\text{sim}(z_i,z_k)/\tau)},\;\;\text{sim}=\frac{z_i^\top z_k}{\|z_i\|\|z_k\|}`],
      h: R`<p>τ≈0.07–0.2 (CLIP learns τ, init 0.07). More negatives (big batch / memory queue) → better. SigLIP replaces softmax with pairwise sigmoid (no global normalization).</p>` },
    { id: 'metric', t: 'Triplet / ArcFace / CosFace', tags: 'metric learning,triplet loss,arcface,face recognition,re-id,margin',
      f: [R`L_{trip}=\max(0,\,d(a,p)-d(a,n)+m)`, R`L_{Arc}=-\log\frac{e^{s\cos(\theta_y+m)}}{e^{s\cos(\theta_y+m)}+\sum_{j\ne y}e^{s\cos\theta_j}}`],
      h: R`<p>ArcFace s=64, m=0.5; CosFace uses \(\cos\theta_y-m\) (m=0.35). Triplet needs hard/semi-hard mining; margin losses on normalized embeddings are easier to train.</p>` },
    { id: 'hinge', t: 'Hinge / Charbonnier / perceptual', tags: 'hinge loss,svm,charbonnier,perceptual loss,lpips,ssim',
      f: [R`\text{Hinge}=\max(0,1-y\hat y)`, R`\text{Charbonnier}=\sqrt{(y-\hat y)^2+\epsilon^2}`, R`\text{Perceptual}=\sum_l\lambda_l\|\phi_l(y)-\phi_l(\hat y)\|_2^2`, R`\text{SSIM}=\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}`],
      h: R`<p>Restoration/super-resolution: L1/Charbonnier + perceptual (VGG/LPIPS) + adversarial. SSIM loss = 1−SSIM.</p>` }
  ]},
  { id: 'reg', title: 'Regularization', items: [
    { id: 'wd', t: 'Weight decay', tags: 'weight decay,l2 regularization',
      f: [R`\theta\leftarrow\theta-\eta\lambda\theta\;\;(\text{decoupled}),\;\;\text{effective decay per step}=\eta\lambda`],
      h: R`<p>With BN, WD mostly controls the <i>effective LR</i> (scale invariance). Tune \(\eta\lambda\) jointly; when you scale LR ×k with AdamW consider keeping \(\eta\lambda\) fixed for long runs.</p>` },
    { id: 'dropout', t: 'Dropout / DropPath (stochastic depth) / DropBlock', tags: 'dropout,drop path,stochastic depth,dropblock',
      f: [R`\text{Dropout: } y=\frac{m\odot x}{1-p},\;m\sim\text{Bern}(1-p)`, R`\text{DropPath: } y=x+\frac{b}{1-p_l}\mathcal{F}(x),\;b\sim\text{Bern}(1-p_l),\;p_l=\tfrac{l}{L}p_L`],
      h: R`<p>DropPath \(p_L\): 0.1 (ViT-S/ConvNeXt-T), 0.2–0.4 (B), 0.4–0.5 (L). Dropout in conv layers is ineffective → DropBlock drops contiguous regions. Transformers pre-training on big data: dropout 0.</p>` },
    { id: 'mixup', t: 'Mixup / CutMix', tags: 'mixup,cutmix,augmentation,soft labels',
      f: [R`\tilde x=\lambda x_i+(1-\lambda)x_j,\;\tilde y=\lambda y_i+(1-\lambda)y_j,\;\lambda\sim\text{Beta}(\alpha,\alpha)`, R`\text{CutMix: } \tilde x=M\odot x_i+(1-M)\odot x_j,\;\lambda=1-\tfrac{r_wr_h}{WH}`],
      h: R`<p>DeiT recipe: mixup α=0.8, cutmix α=1.0, switch prob 0.5. Use BCE or soft-target CE. Not for detection labels directly (use mosaic/copy-paste).</p>` },
    { id: 'early', t: 'Early stopping & checkpoint averaging', tags: 'early stopping,swa,stochastic weight averaging',
      f: [R`\text{SWA: }\bar\theta=\frac1n\sum_{i=1}^{n}\theta_{t_i}`],
      h: R`<p>Patience 10–50 epochs (YOLO default 100). SWA/EMA/model soups (average fine-tuned models with different hparams) improve robustness for free.</p>` }
  ]},
  { id: 'aug', title: 'Data Augmentation', items: [
    { id: 'augtab', t: 'Augmentation catalogue', tags: 'augmentation,randaugment,trivialaugment,autoaugment,albumentations,color jitter,random resized crop',
      h: T(['Aug', 'Params (typical)', 'Tasks'], [
        ['RandomResizedCrop', 'scale (0.08,1) [0.35,1 for small models], ratio (3/4,4/3)', 'cls'],
        ['Horizontal flip', 'p=0.5 (swap left/right keypoints!)', 'all'],
        ['Color jitter / HSV', 'YOLO: h 0.015, s 0.7, v 0.4', 'all'],
        ['RandAugment', 'N=2, M=9 (±0.5 std)', 'cls'],
        ['TrivialAugmentWide', 'one op, random magnitude — no tuning', 'cls'],
        ['AutoAugment / AugMix', 'learned policy / mixed chains + JSD (robustness)', 'cls'],
        ['Random Erasing / Cutout', 'p=0.25', 'cls, re-id'],
        ['Mosaic', '4 images in 2×2, p=1, off last 10 ep', 'det, seg, pose'],
        ['MixUp (det)', 'p=0.0–0.15 on mosaic images', 'det'],
        ['Copy-Paste', 'paste instances between images, p=0.1–0.3', 'inst seg, det'],
        ['Large-scale jitter (LSJ)', 'resize 0.1–2.0 then crop 1024²', 'det/seg (Mask2Former, ViTDet)'],
        ['Scale/translate/rotate/shear', 'YOLO: scale 0.5, translate 0.1, deg 0', 'det, pose'],
        ['Photometric (blur, noise, JPEG)', 'p=0.01–0.1 each', 'robustness, deployment domain'],
        ['Multi-scale training', 'short side 480–800 (DETR), ±50% imgsz (YOLO multi_scale)', 'det/seg']]),
      tip: 'Augmentation strength should scale with model capacity and inversely with data size: small models on ImageNet use weaker aug (RandAug M=5, no mixup); ViTs need strong aug + regularization unless pre-trained.' },
    { id: 'tta', t: 'Test-time augmentation (TTA)', tags: 'tta,test time augmentation,flip test,multi-scale inference',
      f: [R`\hat y=\frac1K\sum_{k=1}^{K}T_k^{-1}\big(f(T_k(x))\big)`],
      h: R`<p>Flip + multi-scale (0.5–1.75 in seg). +0.5–2 points, K× cost. Detection TTA needs box merging (WBF) instead of averaging.</p>` }
  ]},
  { id: 'metrics', title: 'Evaluation Metrics (general)', items: [
    { id: 'prf', t: 'Precision, Recall, F1, Accuracy', tags: 'precision,recall,f1,confusion matrix,accuracy,specificity',
      f: [R`P=\frac{TP}{TP+FP},\;\;R=\frac{TP}{TP+FN},\;\;F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}`, R`\text{Acc}=\frac{TP+TN}{N},\;\;\text{Specificity}=\frac{TN}{TN+FP},\;\;\text{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP{+}FP)(TP{+}FN)(TN{+}FP)(TN{+}FN)}}`],
      h: R`<p>Macro averaging treats classes equally (use for imbalance); micro averaging pools counts.</p>` },
    { id: 'auc', t: 'ROC-AUC & PR-AUC', tags: 'roc,auc,pr curve,average precision',
      f: [R`\text{TPR}=R,\;\;\text{FPR}=\frac{FP}{FP+TN},\;\;\text{AUC}=P(s^+>s^-)`, R`\text{AP}=\sum_n(R_n-R_{n-1})P_n`],
      h: R`<p>Heavy imbalance → PR-AUC is more informative than ROC-AUC.</p>` },
    { id: 'ece', t: 'Calibration (ECE)', tags: 'calibration,ece,temperature scaling,reliability',
      f: [R`\text{ECE}=\sum_{b=1}^{B}\frac{|B_b|}{n}\left|\text{acc}(B_b)-\text{conf}(B_b)\right|`],
      h: R`<p>Fix with <b>temperature scaling</b> (fit a single τ on val by NLL). Needed when scores drive thresholds or downstream decisions.</p>` },
    { id: 'stat', t: 'Statistical significance', tags: 'confidence interval,seed variance,bootstrap',
      f: [R`\text{CI}_{95\%}\approx\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}`],
      h: R`<p>ImageNet val (50k): ±0.2% is noise-level; COCO AP seed variance ≈ ±0.1–0.3. Report mean±std over ≥3 seeds when claiming < 0.5-point gains.</p>` }
  ]},
  { id: 'dist', title: 'Scaling & Distributed Training', items: [
    { id: 'lsr', t: 'Batch size ↔ learning rate (linear scaling rule)', tags: 'linear scaling rule,batch size,sqrt scaling,gradient accumulation',
      f: [R`\eta=\eta_{base}\cdot\frac{B}{B_{base}}\;\;(\text{SGD}),\;\;\;\eta=\eta_{base}\sqrt{\frac{B}{B_{base}}}\;\;(\text{Adam, often})`, R`B_{eff}=B_{gpu}\times N_{gpu}\times\text{accum\_steps}`],
      h: R`<p>Holds up to a critical batch size (~8k for ImageNet ResNet); beyond it, needs LARS/LAMB and longer warmup. With gradient accumulation, BN still sees only \(B_{gpu}\).</p>` },
    { id: 'ddp', t: 'DDP / FSDP / ZeRO', tags: 'ddp,fsdp,zero,deepspeed,data parallel,model parallel',
      h: T(['Strategy', 'What is sharded', 'Memory per GPU (Adam, P params)'], [
        ['DDP', 'nothing (grads all-reduced)', '≈16P bytes (fp32 master+grads+2 moments)'],
        ['ZeRO-1', 'optimizer states', '≈4P + 12P/N'], ['ZeRO-2', '+ gradients', '≈2P + 14P/N'], ['ZeRO-3 / FSDP', '+ parameters', '≈16P/N'],
        ['Tensor / pipeline parallel', 'layers split across GPUs', 'for > 10B models']]) + R`<p>Plus activation memory ∝ batch × resolution × depth → use gradient checkpointing (≈+30% compute, big memory saving).</p>`,
      code: R`
# torchrun --nproc_per_node=8 train.py
torch.distributed.init_process_group('nccl')
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # CNNs with small per-GPU batch
model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank])
sampler = torch.utils.data.DistributedSampler(ds, shuffle=True); sampler.set_epoch(epoch)` },
    { id: 'scaling', t: 'Scaling laws & compute', tags: 'scaling law,chinchilla,compute optimal,flops budget',
      f: [R`L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},\;\;C\approx6ND\;\text{FLOPs}`, R`\text{Chinchilla (LLM): } D_{opt}\approx20N`],
      h: R`<p>Vision: ViT scaling (Zhai et al.) shows power-law gains with compute; data quality/curation (DINOv2 LVD-142M, DataComp) often beats raw size. Training time estimate: \(t\approx 6ND/(\text{GPUs}\times\text{peak FLOPs}\times\text{MFU}\approx0.3\text{–}0.5)\).</p>` }
  ]}
  ]
});
})();

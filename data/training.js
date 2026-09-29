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
  { id: 'sched', title: 'Learning Rate Schedulers', intro: 'The LR schedule often matters as much as the peak LR. \\(t\\): current step, \\(T\\): total steps, \\(\\eta_{max}\\): peak LR, \\(T_w\\): warmup steps. Every card shows the curve (normalized to \\(\\eta_{max}\\)). Use the explorer at the end to plot any schedule with your own numbers and get PyTorch code.', items: [
    { id: 'why', t: 'Why schedule the learning rate at all?', tags: 'learning rate schedule,why decay,lr decay,exploration exploitation',
      s: 'High LR early explores the loss landscape and escapes poor regions; low LR late settles into a minimum and reduces gradient noise.',
      f: [R`\theta_{t+1}=\theta_t-\eta_t\,g_t,\;\;\text{SGD noise scale}\;\propto\;\frac{\eta_t}{B}`],
      h: R`<ul><li><b>Warmup</b> protects the start: Adam's second-moment estimates, BatchNorm statistics and randomly-initialized heads are unreliable in the first few hundred steps.</li><li><b>Decay</b> is where most of the final accuracy is gained: validation metrics typically jump when the LR drops.</li><li>The <b>area under the LR curve</b> (≈ total "distance" travelled) and the <b>final LR</b> matter more than the exact shape — cosine, linear and WSD reach similar results when tuned.</li></ul>` },
    { id: 'constant', t: 'Constant (± warmup)', tags: 'constant lr,constantlr,warmup constant,no decay',
      f: [R`\eta_t=\eta_{max}\;\;(\text{after warmup})`], lr: { s: 'warmup_constant', warm: 5 },
      h: R`<p><b>Use:</b> diffusion / DiT training (paired with EMA weights), LoRA or adapter fine-tunes, debugging (removes one variable). <b>Avoid:</b> supervised training to convergence — you leave accuracy on the table without a final decay.</p>` },
    { id: 'warmup', t: 'Linear warmup', tags: 'warmup,lr warmup,linearlr,gradual warmup,stability',
      f: [R`\eta_t=\eta_{max}\cdot\frac{t+1}{T_w},\;\;t<T_w`], lr: { s: 'warmup_constant', warm: 15 },
      h: R`<p>Almost always combined with another schedule. Typical length: 1–5% of training or 1–5 epochs (ViT 5–20 epochs, YOLO 3 epochs with bias-LR 0.1 and momentum 0.8 → 0.937, DETR-family ~2k iterations, fine-tuning 5–10%).</p>`,
      tip: 'Increase warmup when you raise the batch size or LR, train transformers from scratch, or see loss spikes / NaNs in the first epoch.' },
    { id: 'step', t: 'StepLR', tags: 'step decay,steplr,step lr,gamma,step size',
      f: [R`\eta_t=\eta_0\,\gamma^{\lfloor t/s\rfloor}\;\;(\gamma=0.1,\;s=30\text{ epochs})`], lr: { s: 'step', stepSize: 30, gamma: 0.1 },
      h: R`<p><b>Use:</b> reproducing classic CNN papers (ResNet: ×0.1 every 30 epochs of 90), quick baselines. <b>Avoid:</b> new recipes — the sudden drops waste the early high-LR phase and need milestone tuning; cosine is almost always equal or better.</p>` },
    { id: 'multistep', t: 'MultiStepLR', tags: 'multistep,multisteplr,milestones,detectron schedule,1x 3x',
      f: [R`\eta_t=\eta_0\,\gamma^{\,|\{m_i\,:\,m_i\le t\}|}`], lr: { s: 'multistep', milestones: [0.67, 0.89], gamma: 0.1 },
      h: R`<p><b>Use:</b> Detectron2 / MMDetection "1× / 3×" schedules (drop ×0.1 at ~67% and ~89% of iterations), DETR (drop at epoch 200 of 300/500), RT-DETR fine-tunes. Good when you want the model to stay at high LR for most of training and then consolidate.</p>` },
    { id: 'exponential', t: 'ExponentialLR', tags: 'exponential decay,exponentiallr,gamma per epoch',
      f: [R`\eta_t=\eta_0\,\gamma^{t}\;\;(\gamma\approx0.97\text{ per epoch})`], lr: { s: 'exponential', expGamma: 0.97 },
      h: R`<p><b>Use:</b> old TF recipes (EfficientNet: ×0.97 every 2.4 epochs), RL, when training length is unknown but you want continuous decay. <b>Watch:</b> LR can decay too fast or too slow depending on γ — compute \(\gamma=(\eta_{end}/\eta_0)^{1/T}\).</p>` },
    { id: 'linear', t: 'Linear decay (± warmup)', tags: 'linear decay,linear schedule,lrf,yolo schedule,transformer finetune',
      f: [R`\eta_t=\eta_{max}\Big(1-(1-r)\frac{t-T_w}{T-T_w}\Big),\;\;r=\eta_{end}/\eta_{max}`], lr: { s: 'linear', warm: 5, min: 0.01 },
      h: R`<p><b>Use:</b> Ultralytics YOLO default (to lrf = 0.01 of lr0), BERT/ViT fine-tuning (warmup + linear to 0), short runs. Behaves close to cosine; slightly more time at mid LR.</p>` },
    { id: 'poly', t: 'Polynomial (PolyLR)', tags: 'poly,polynomial,polylr,segmentation schedule,power 0.9',
      f: [R`\eta_t=(\eta_0-\eta_{min})\Big(1-\frac{t}{T}\Big)^{p}+\eta_{min},\;\;p=0.9`], lr: { s: 'poly', power: 0.9 },
      h: R`<p><b>Use:</b> semantic segmentation (DeepLab, PSPNet, mmsegmentation 80k/160k iterations; SegFormer uses power 1.0), iteration-based training. Stays high longer than cosine then drops steeply near the end.</p>` },
    { id: 'cosine', t: 'Cosine annealing (+ warmup)', tags: 'cosine,cosine annealing,cosineannealinglr,cosine decay,warmup cosine',
      f: [R`\eta_t=\eta_{min}+\tfrac12(\eta_{max}-\eta_{min})\Big(1+\cos\frac{\pi (t-T_w)}{T-T_w}\Big)`], lr: { s: 'cosine', warm: 5, min: 0.01 },
      h: R`<p><b>Use:</b> the default for classification (timm, DeiT, ConvNeXt), ViTs, DETR variants, most fine-tuning, YOLO with <code>cos_lr=True</code>. Needs the total budget \(T\) up front. \(\eta_{min}\) ≈ 0–1% of \(\eta_{max}\).</p>`,
      tip: 'If you later want to train longer, you must restart the schedule (or use WSD). Resuming a cosine run with a larger T changes the curve you already followed.' },
    { id: 'sgdr', t: 'Cosine with warm restarts (SGDR)', tags: 'sgdr,warm restarts,cosineannealingwarmrestarts,snapshot ensemble,t_0 t_mult',
      f: [R`\eta_t=\eta_{min}+\tfrac12(\eta_{max}-\eta_{min})\big(1+\cos(\pi T_{cur}/T_i)\big),\;\;T_{i+1}=T_{mult}T_i`], lr: { s: 'sgdr', T0: 0.14, Tmult: 2 },
      h: R`<p><b>Use:</b> long runs where you want good checkpoints periodically (end of each cycle), snapshot ensembles, escaping sharp minima. <b>Avoid:</b> short fine-tunes — the restarts disturb a pre-trained model.</p>` },
    { id: 'onecycle', t: 'OneCycleLR', tags: 'one cycle,onecyclelr,super convergence,lr range test,momentum cycling',
      f: [R`\eta:\;\tfrac{\eta_{max}}{25}\nearrow\eta_{max}\;(\text{first }30\%)\searrow\tfrac{\eta_{max}}{25\cdot10^{4}},\;\;\mu:\;0.95\searrow0.85\nearrow0.95`], lr: { s: 'onecycle' },
      h: R`<p><b>Use:</b> small budgets and fast iteration (fastai-style), small datasets, CNNs with SGD. Find \(\eta_{max}\) with an <b>LR range test</b>: ramp the LR exponentially for ~100 steps and take ~1/10 of the LR at which loss is lowest. Momentum is cycled inversely.</p>` },
    { id: 'wsd', t: 'Warmup–Stable–Decay (WSD / trapezoidal)', tags: 'wsd,warmup stable decay,trapezoidal,cooldown,constant then decay',
      f: [R`\eta_t=\begin{cases}\eta_{max}\,t/T_w & t<T_w\\ \eta_{max} & T_w\le t<T-T_d\\ \eta_{max}\,f\!\big(\tfrac{T-t}{T_d}\big) & t\ge T-T_d,\;T_d\approx10\text{–}20\%\,T\end{cases}`], lr: { s: 'wsd', warm: 5, decayFrac: 0.2 },
      h: R`<p><b>Use:</b> large-scale pre-training and continual training: you can branch a cooldown from any checkpoint of the stable phase, or extend training without restarting. Matches cosine when the cooldown uses 10–20% of steps (linear or 1−√ shape).</p>` },
    { id: 'invsqrt', t: 'Inverse square root (Noam)', tags: 'inverse sqrt,noam,rsqrt,transformer schedule',
      f: [R`\eta_t=\eta_{max}\min\!\Big(\frac{t}{T_w},\sqrt{\frac{T_w}{t}}\Big)`], lr: { s: 'invsqrt', warm: 8 },
      h: R`<p><b>Use:</b> original Transformer / NLP recipes, open-ended training. Decays slowly forever; for vision a cosine or WSD cooldown usually gives a better final model.</p>` },
    { id: 'cyclic', t: 'CyclicLR (triangular)', tags: 'cyclic,cycliclr,triangular,clr',
      f: [R`\eta_t=\eta_{base}+(\eta_{max}-\eta_{base})\max\!\big(0,1-|t/s-2c-1|\big)`], lr: { s: 'cyclic', base: 0.1, half: 0.1 },
      h: R`<p><b>Use:</b> exploring good LR bounds, some small-data or GAN setups. Mostly superseded by OneCycle and cosine restarts.</p>` },
    { id: 'plateau', t: 'ReduceLROnPlateau', tags: 'plateau,reducelronplateau,patience,adaptive schedule,metric based',
      f: [R`\eta\leftarrow\eta\cdot\text{factor}\;\;\text{if no improvement for }\textit{patience}\text{ epochs}`], lr: { s: 'plateau' },
      h: R`<p><b>Use:</b> unknown training length, small or medical datasets, when you monitor a trustworthy validation metric. Typical: factor 0.1–0.5, patience 5–10 epochs, <code>min_lr</code> 1e-6, combine with early stopping. <b>Avoid:</b> noisy validation metrics (drops fire randomly) and when you need reproducible schedules.</p>` },
    { id: 'llrd', t: 'Layer-wise LR decay & differential LR', tags: 'llrd,layer decay,discriminative lr,backbone lr,head lr,fine-tuning',
      f: [R`\eta_l=\eta\cdot\alpha^{L-l},\;\;\alpha\in[0.65,0.9]`],
      h: R`<p>Not a time schedule — a per-layer multiplier combined with any schedule above. Lower layers (generic features) change less. BEiT/MAE/DINOv2/EVA fine-tuning: α = 0.65–0.75 (base), 0.8–0.9 (large). Simpler: backbone LR = head LR × 0.1 (DETR family, RF-DETR).</p>`,
      code: R`
def llrd_groups(model, base_lr, decay=0.75, n_layers=12):
    groups = []
    for n, p in model.named_parameters():
        layer = n_layers                     # head / norm: full LR
        if n.startswith(('patch_embed', 'cls_token', 'pos_embed')): layer = 0
        elif n.startswith('blocks.'): layer = int(n.split('.')[1]) + 1
        groups.append({'params': [p], 'lr': base_lr * decay ** (n_layers - layer)})
    return groups
opt = torch.optim.AdamW(llrd_groups(model, 1e-4), weight_decay=0.05)` },
    { id: 'schedcmp', t: 'Which scheduler when? (decision guide)', tags: 'which scheduler,scheduler comparison,choose lr schedule,when to use', wide: true,
      h: T(['Situation', 'Recommended schedule', 'Typical settings'], [
        ['Known budget, training from scratch (cls / det / seg)', 'Linear warmup + cosine', 'warmup 1–5 %, η_min ≈ 1 % of η_max'],
        ['Fine-tuning a pre-trained backbone', 'Warmup + cosine or linear, + LLRD', 'warmup 5–10 %, 10–100× lower LR than scratch'],
        ['YOLO (Ultralytics)', 'Linear (default) or cosine (<code>cos_lr=True</code>)', 'lr0 0.01, lrf 0.01, warmup 3 epochs, close mosaic last 10'],
        ['DETR / RT-DETR / D-FINE / RF-DETR', 'Warmup (≈2k iters) + constant then step, or cosine', 'drop ×0.1 near the end; EMA 0.9999'],
        ['Detectron-style two-stage', 'MultiStep ×0.1 at 67 % and 89 %', '1× = 12 epochs, 3× = 36 epochs'],
        ['Semantic segmentation (iteration based)', 'Poly (p = 0.9) with 1.5k-iter warmup', '80k / 160k iterations'],
        ['Very short time budget', 'OneCycle', 'η_max from LR range test'],
        ['Unknown budget / continual training', 'WSD or ReduceLROnPlateau (or schedule-free AdamW)', 'cooldown 10–20 %; plateau factor 0.5, patience 5'],
        ['Diffusion / DiT / flow matching', 'Warmup + constant', 'warmup 1–10k steps; evaluate the EMA weights'],
        ['LoRA / adapters', 'Constant or cosine with short warmup', '1e-4 (LoRA), few hundred steps warmup'],
        ['Large-batch training', 'Longer warmup + cosine (LARS / LAMB)', 'warmup 5–10 epochs, LR ∝ batch']]),
      tip: 'Debug order: first get the peak LR right (range test), then warmup length, then the decay shape. A good peak LR with any sensible decay beats a perfect shape with the wrong peak.' },
    { id: 'explorer', t: 'LR schedule explorer (interactive)', tags: 'lr plotter,visualize learning rate schedule,scheduler code generator', wide: true,
      s: 'Pick a scheduler, set your steps / warmup / minimum LR, see the curve and copy the matching PyTorch code.', widget: 'lrplot' }
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
  { id: 'kd', title: 'Knowledge Distillation (teacher → student)', intro: 'Transfer what a large or slow model knows into a small or fast one. The teacher is frozen (or an EMA of the student); the student learns from the teacher\'s outputs, features or relations in addition to (or instead of) ground-truth labels.', items: [
    { id: 'process', t: 'The distillation process, step by step', tags: 'knowledge distillation,distillation process,teacher student,kd pipeline,model compression', wide: true,
      s: 'Train (or pick) a strong teacher → run it on the training data → train the student to match the teacher’s soft outputs / features while also fitting the labels.',
      dg: [
        { k: 'input', t: 'Data', d: 'labelled + (ideally) lots of unlabelled in-domain images; same augmentation for both models' },
        { k: 'other', t: 'Teacher (frozen)', d: 'large / ensemble / foundation model', b: ['Forward pass only (no grad)', 'Soft probabilities p_t = softmax(z_t / τ)', 'Intermediate features F_t', 'Boxes / masks as pseudo-labels'] },
        { k: 'backbone', t: 'Student (trainable)', d: 'small / fast deployable model', b: ['Logits z_s', 'Features F_s → 1×1 adapter to teacher dims'] },
        { k: 'loss', t: 'Distillation losses', b: ['Task loss vs ground truth', 'KL(p_t ‖ p_s) · τ²', 'Feature MSE / attention / relation terms'], o: 'L = (1−α) L_task + α L_KD (+ β L_feat)' },
        { k: 'output', t: 'Deploy the student', d: 'teacher discarded' }],
      h: R`<ol><li><b>Teacher.</b> Train or choose the most accurate model you can afford to run offline (large backbone, ensemble, TTA, or a foundation model / VLM).</li><li><b>Student.</b> Pick the architecture that meets the latency budget. Keep the capacity gap reasonable (e.g. ViT-L → ViT-S/B, YOLO-x → YOLO-s); for huge gaps use a teacher assistant.</li><li><b>What to transfer.</b> Logits (always), plus features for dense tasks, plus relations if embeddings matter.</li><li><b>Data.</b> Distillation needs no labels for the KD term — add unlabelled in-domain data. Feed teacher and student the <i>same</i> augmented view ("consistent" teaching).</li><li><b>Loss & hyperparameters.</b> τ = 2–4, α = 0.5–0.9 (see table below). Normalize feature losses so they are on the scale of the task loss.</li><li><b>Train long.</b> Students keep improving far past the usual schedule ("patient" distillation: 2–10× epochs).</li><li><b>Evaluate</b> the student on the same validation set and deploy constraints (latency, INT8) as the teacher.</li></ol>` },
    { id: 'logit', t: 'Response-based KD (logits, Hinton)', tags: 'logit distillation,soft targets,temperature,hinton kd,kl divergence,dark knowledge',
      f: [R`\mathcal L=(1-\alpha)\,\mathrm{CE}(y,\sigma(z_s))+\alpha\,\tau^{2}\,D_{KL}\big(\sigma(z_t/\tau)\,\|\,\sigma(z_s/\tau)\big)`, R`\text{DKD: }\;\mathcal L_{KD}=\alpha\,\text{TCKD}+\beta\,\text{NCKD}\;\;(\text{target-class vs non-target-class parts})`],
      h: R`<p>Soft targets carry "dark knowledge": which wrong classes are similar (a cat image scoring 0.1 for "dog" but 0.0001 for "car"). \(\tau^2\) keeps gradient magnitudes comparable as τ changes. <b>DKD</b> splits KD into target and non-target terms and up-weights the non-target part (β ≈ 8), usually +0.5–1%.</p>`,
      code: R`
def kd_loss(z_s, z_t, y, T=4.0, alpha=0.9):
    ce = F.cross_entropy(z_s, y)
    kl = F.kl_div(F.log_softmax(z_s / T, 1), F.softmax(z_t / T, 1), reduction='batchmean') * T * T
    return (1 - alpha) * ce + alpha * kl

with torch.no_grad():
    z_t = teacher(x)            # same augmented x as the student
loss = kd_loss(student(x), z_t, y)` },
    { id: 'feature', t: 'Feature-based KD (FitNets, attention, dense tasks)', tags: 'feature distillation,fitnets,hint learning,attention transfer,cwd,fgd,mgd,detection distillation',
      f: [R`\mathcal L_{hint}=\big\|\,r(F_s)-F_t\,\big\|_2^2\;\;(r:\text{1×1 conv adapter})`, R`\mathcal L_{AT}=\sum_l\Big\|\frac{Q_s^l}{\|Q_s^l\|_2}-\frac{Q_t^l}{\|Q_t^l\|_2}\Big\|_2,\;\;Q=\textstyle\sum_c|F_c|^2`, R`\text{CWD: }\;\tau^2\sum_c D_{KL}\big(\sigma(F_{t,c}/\tau)\,\|\,\sigma(F_{s,c}/\tau)\big)\;\;(\text{softmax over pixels per channel})`],
      h: CV.tbl(['Method', 'Idea', 'Best for'], [['FitNets hints', 'match intermediate features through an adapter', 'deep thin students'], ['Attention transfer', 'match spatial attention maps (Σ|F|²)', 'CNN classification'], ['CWD (channel-wise)', 'KL between per-channel spatial distributions', 'semantic segmentation, detection necks'], ['FGD / MGD / PKD', 'focal & global feature masks; masked generation; Pearson-correlation matching on FPN', 'object detection (FPN levels)'], ['LD (localization distillation)', 'KL between teacher and student DFL box distributions', 'YOLO / GFL-style detectors']]) },
    { id: 'relation', t: 'Relation-based KD', tags: 'relation distillation,rkd,similarity preserving,contrastive distillation,crd',
      f: [R`\mathcal L_{RKD\text{-}D}=\sum_{i,j}\ell_\delta\!\Big(\tfrac{\|t_i-t_j\|}{\mu_t},\tfrac{\|s_i-s_j\|}{\mu_s}\Big)`],
      h: R`<p>Match the <i>structure</i> between samples (distances, angles, similarity matrices) instead of individual outputs. Useful for embedding / retrieval / re-ID students, and when teacher and student output spaces differ. CRD adds a contrastive objective between teacher and student embeddings.</p>` },
    { id: 'self', t: 'Self-distillation, online & mutual learning', tags: 'self-distillation,born again networks,mutual learning,dml,ema teacher,mean teacher',
      h: CV.tbl(['Variant', 'Teacher', 'Notes'], [['Born-Again Networks', 'a trained copy of the same architecture', 'student often beats the teacher; iterate 2–3 generations'], ['EMA / momentum teacher', 'exponential moving average of the student', 'DINO, BYOL, Mean Teacher; no separate training'], ['Deep supervision self-KD', 'the deepest layer teaches shallower exits', 'early-exit networks, D-FINE GO-LSD (final decoder layer → earlier layers)'], ['Deep Mutual Learning (online)', 'peers teach each other simultaneously', 'no pre-trained teacher needed'], ['DeiT distillation token', 'a CNN teacher via a dedicated token', 'hard-label distillation works best for ViTs']]) },
    { id: 'semi', t: 'Teacher–student for unlabelled data (semi-supervised)', tags: 'pseudo labeling,noisy student,fixmatch,mean teacher,unbiased teacher,soft teacher,self-training',
      f: [R`\text{FixMatch: }\;\mathcal L_u=\frac1{\mu B}\sum_b\mathbb 1\big[\max q_b\ge\tau\big]\,\mathrm{CE}\big(\hat q_b,\,p_s(\mathcal A_{strong}(u_b))\big),\;\;\tau=0.95`],
      h: R`<ol><li><b>Pseudo-label:</b> the teacher predicts on unlabelled images; keep confident predictions (cls score ≥ 0.7–0.95; detection box score ≥ 0.5–0.7).</li><li><b>Train the student</b> on labels + pseudo-labels with <i>strong</i> augmentation / noise (Noisy Student: RandAugment, dropout, stochastic depth).</li><li><b>Iterate:</b> the student becomes the next teacher (2–3 rounds).</li></ol><p>Detection variants: Unbiased Teacher, Soft Teacher (EMA teacher + score-weighted pseudo boxes). Watch confirmation bias: tune the threshold on a labelled validation set and track per-class pseudo-label precision.</p>` },
    { id: 'foundation', t: 'Distilling foundation models into deployable detectors / segmenters', tags: 'auto-labeling,foundation model distillation,grounding dino to yolo,sam distillation,vlm pseudo labels',
      h: R`<ol><li>Run open-vocabulary models (Grounding DINO / OWLv2 / YOLOE / SAM 3 / a VLM) on your unlabelled images with your class prompts.</li><li>Filter: score threshold, NMS across prompts, size sanity checks; spot-check a random 2–5% by hand.</li><li>Train a closed-set real-time model (YOLO26, RF-DETR, D-FINE) on the pseudo-labels, then fine-tune on a small hand-labelled set.</li><li>Optionally add feature distillation from a DINOv2/v3 backbone into the student backbone.</li></ol>`,
      tip: 'Expect the student to reach 80–95% of the teacher\'s accuracy at 10–100× lower latency. The hand-labelled validation set is what tells you whether pseudo-label noise is hurting.' },
    { id: 'gap', t: 'Capacity gap, teacher assistants & multi-teacher', tags: 'capacity gap,teacher assistant,takd,ensemble distillation,multi-teacher',
      h: R`<ul><li>A much stronger teacher is not always better: when the gap is too large the student cannot mimic it (lower τ, or insert a mid-size <b>teacher assistant</b>: L → M → S).</li><li><b>Ensemble / multi-teacher:</b> average teacher probabilities (or weight by confidence); captures more dark knowledge than a single model.</li><li><b>Early-stopped teachers</b> sometimes distil better than fully converged, over-confident ones.</li></ul>` },
    { id: 'other', t: 'Other distillation approaches', tags: 'data-free distillation,dataset distillation,quantization aware distillation,cross-modal distillation',
      h: CV.tbl(['Approach', 'What it does'], [['Data-free KD', 'synthesize inputs from the teacher (BN statistics inversion, generators) when training data is unavailable'], ['Dataset distillation', 'compress a dataset into a few synthetic images that train a model almost as well (research, small scale)'], ['Quantization-aware distillation', 'FP32 teacher → INT8 student during QAT; recovers most quantization loss'], ['Cross-modal distillation', 'RGB teacher → depth / thermal / event-camera student using paired data'], ['Pruning + KD', 'prune the teacher into the student, then distil to recover accuracy']]) },
    { id: 'kdhp', t: 'Distillation hyperparameters & pitfalls', tags: 'distillation temperature,alpha,kd hyperparameters,kd pitfalls', wide: true,
      h: T(['Hyperparameter', 'Typical', 'Guidance'], [['Temperature τ', '2–4 (1 for hard-label KD)', 'higher τ = softer targets; lower it when the capacity gap is large'], ['KD weight α', '0.5–0.9', 'raise when labels are noisy or scarce; with only unlabelled data α = 1'], ['Feature loss weight β', 'scale so it is ~0.1–1× the task loss at init', 'normalize features (LayerNorm / channel-wise) before MSE'], ['Epochs', '2–10× the normal schedule', 'students keep improving ("patient")'], ['Augmentation', 'same view for teacher and student; mixup/cutmix work well', '"consistent" teaching beats pre-computed teacher logits'], ['LR', 'same as normal training of the student', 'AdamW / SGD as usual']]) +
        R`<p><b>Pitfalls:</b> precomputing teacher outputs on un-augmented images (loses most of the benefit) · mismatched preprocessing between teacher and student · forgetting <code>teacher.eval()</code> (BN/dropout in train mode) · distilling a mis-calibrated teacher with τ = 1 · evaluating only the KD loss instead of the task metric.</p>` }
  ]},
  { id: 'practice', title: 'Practical Training Concepts', intro: 'The things that decide whether a training run succeeds in practice — beyond the architecture.', items: [
    { id: 'units', t: 'Epochs, iterations, steps & effective batch', tags: 'epoch,iteration,step,batch size,effective batch,gradient accumulation',
      f: [R`\text{iterations per epoch}=\Big\lceil\frac{N_{train}}{B_{gpu}\cdot N_{gpu}}\Big\rceil,\;\;B_{eff}=B_{gpu}\cdot N_{gpu}\cdot k_{accum}`],
      h: R`<p>Schedulers and warmup can be defined per step or per epoch — make sure \(T\) and <code>sched.step()</code> use the same unit. Iteration-based schedules (segmentation 160k iters) are independent of dataset size; epoch-based ones are not. When you change batch size, re-scale LR (linear for SGD, ≈√ for Adam) and warmup.</p>` },
    { id: 'splits', t: 'Train / val / test splits & data leakage', tags: 'data split,validation set,test set,leakage,group k-fold,stratified split,near duplicates',
      h: R`<ul><li><b>Split by group</b>, not by image, whenever images are correlated: same video, patient, scene, camera, day or product. Otherwise validation is inflated.</li><li>Remove near-duplicates across splits (perceptual hash or embedding similarity > 0.95).</li><li><b>Stratify</b> by class (and by object size / domain) so rare classes appear in validation.</li><li>Keep a <b>frozen test set</b> you look at rarely; tune thresholds and hyperparameters on validation only.</li><li>Small datasets: 5-fold (group-)stratified cross-validation and report mean ± std.</li></ul>` },
    { id: 'preproc', t: 'Preprocessing parity & normalization', tags: 'normalization,mean std,imagenet mean,letterbox,resize,rgb bgr,preprocessing mismatch',
      f: [R`x'=\frac{x/255-\mu}{\sigma},\;\;\mu=(0.485,0.456,0.406),\;\sigma=(0.229,0.224,0.225)`],
      h: R`<p>Use the normalization the pre-trained weights expect (ImageNet stats; CLIP/SigLIP and DINOv2 have their own; YOLO uses x/255 only). Train, validation, export and production must apply identical resize (letterbox vs stretch, interpolation), color order (RGB vs BGR) and padding value (114 for YOLO). A mismatch silently costs several points.</p>` },
    { id: 'freeze', t: 'Freezing, unfreezing & BatchNorm handling', tags: 'freeze backbone,unfreeze,gradual unfreezing,frozen batchnorm,bn eval mode,linear probe first',
      h: R`<ul><li><b>Tiny data:</b> freeze the backbone, train the head (linear probe), then unfreeze with 10× lower LR (LP-FT).</li><li><b>Gradual unfreezing:</b> unfreeze stage by stage from the top every few epochs.</li><li><b>BatchNorm with small per-GPU batch (< 8–16):</b> freeze BN statistics (<code>FrozenBatchNorm2d</code>, or <code>m.eval()</code> for BN layers) or use SyncBN / GroupNorm.</li><li>Ultralytics: <code>freeze=10</code> freezes the first 10 layers (backbone) for fast transfer.</li></ul>`,
      code: R`
for p in model.backbone.parameters(): p.requires_grad = False   # stage 1: head only
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
# ...train a few epochs, then unfreeze with a lower LR for the backbone:
for p in model.backbone.parameters(): p.requires_grad = True
opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 1e-5},
                         {'params': model.head.parameters(), 'lr': 1e-4}], weight_decay=0.05)` },
    { id: 'ckpt', t: 'Checkpointing, resuming & model selection', tags: 'checkpoint,resume training,best model,last model,model selection,early stopping',
      h: R`<ul><li>Save <b>model, optimizer, scheduler, AMP scaler, EMA weights, epoch/step and RNG states</b> — resuming without the optimizer and scheduler restarts momentum and LR.</li><li>Keep <code>last.pt</code> (for resuming) and <code>best.pt</code> selected on the <b>metric you deploy with</b> (e.g. recall at your precision target, not just mAP).</li><li>Early stopping: patience 10–50 epochs on a smoothed validation metric; always evaluate the EMA weights.</li></ul>`,
      code: R`
torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(),
            'sched': sched.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
            'rng': torch.get_rng_state()}, 'last.pt')` },
    { id: 'curves', t: 'Reading training curves', tags: 'training curves,loss curve,overfitting diagnosis,learning curves,monitoring,tensorboard,wandb', wide: true,
      h: T(['What you see', 'Diagnosis', 'Action'], [['Train ↓, val ↓, gap small', 'healthy, maybe under-trained', 'train longer / bigger model'], ['Train ↓, val ↑ after epoch k', 'overfitting', 'more data/aug, WD ↑, drop-path ↑, early stop at k'], ['Both flat and high', 'underfitting or LR too low / too high', 'LR range test, bigger model, check labels'], ['Loss spikes / NaN', 'LR too high, bad batch, fp16 overflow', 'warmup ↑, grad clip, bf16, inspect batch'], ['Val metric jumps when LR drops', 'normal — decay phase consolidates', 'don’t stop early before the decay'], ['Val loss ↑ but val mAP ↑', 'over-confidence (calibration) not accuracy', 'select on mAP/F1, calibrate scores'], ['Train mAP ≪ expected from start', 'data / label bug', 'visualize augmented batches with labels']]) +
        R`<p>Log per step: loss components, LR, grad-norm. Per epoch: per-class AP / recall, confusion matrix, a grid of predictions on fixed validation images.</p>` },
    { id: 'noise', t: 'Label noise & hard examples', tags: 'label noise,noisy labels,cleanlab,co-teaching,ohem,hard example mining,curriculum learning',
      h: R`<ul><li><b>Find noise:</b> high-loss samples after a few epochs, disagreement between model and label (confident learning / cleanlab), duplicate images with different labels.</li><li><b>Tolerate noise:</b> label smoothing 0.1, symmetric / generalized CE, co-teaching (two nets exchange small-loss samples), early stopping.</li><li><b>Hard examples:</b> focal loss or OHEM emphasise hard samples — but hard can mean mislabelled: inspect them before up-weighting.</li><li><b>Curriculum:</b> easy → hard ordering or progressive resolution; mosaic off at the end (YOLO close-mosaic) is a form of curriculum.</li></ul>` },
    { id: 'threshold', t: 'Choosing the operating threshold', tags: 'confidence threshold,operating point,pr curve,f1 threshold,precision at recall,calibration,temperature scaling',
      f: [R`t^*=\arg\max_t F_\beta(t),\;\;\beta<1:\text{precision-heavy},\;\beta>1:\text{recall-heavy}`],
      h: R`<p>Metrics like mAP/AUC are threshold-free; deployment is not. Pick the threshold on validation from the PR curve for your cost of errors (F-beta, precision ≥ X at max recall, or recall ≥ Y at max precision), per class if needed. Calibrate first (temperature scaling) so scores are comparable. Re-check the threshold after every retraining — score distributions shift. See the <a href="#/hp/hp--errors--advisor">FP/FN advisor</a>.</p>` },
    { id: 'throughput', t: 'Training throughput checklist', tags: 'dataloader,num_workers,pin_memory,channels_last,torch.compile,gpu utilization,training speed',
      h: R`<ul><li>GPU util < 80%? DataLoader: <code>num_workers</code> = CPU cores / GPUs, <code>pin_memory=True</code>, <code>persistent_workers=True</code>, decode with Pillow-SIMD / DALI, cache resized images.</li><li><code>torch.backends.cudnn.benchmark=True</code> (fixed input sizes), <code>channels_last</code> for CNNs, bf16 autocast, <code>torch.compile</code>.</li><li>Profile one epoch before scaling up; larger batch only helps if the GPU is the bottleneck.</li></ul>` }
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

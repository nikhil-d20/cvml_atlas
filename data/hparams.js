(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'hp', title: 'Practical Guide: FP/FN Playbook & Hyperparameters', short: 'Practical Guide', icon: '🎛',
  blurb: 'Which hyperparameter for which situation: data size, compute, imbalance, small objects, debugging, recipes.',
  intro: 'Opinionated, field-tested defaults. Start here, change one thing at a time, and log everything.',
  sections: [
  { id: 'errors', title: 'FP / FN Playbook: reduce false positives, false negatives, or both', intro: 'Start here when a trained model makes the wrong kind of mistakes. Enter your task and error counts in the advisor to get an ordered strategy; the tables below explain every lever and what it trades off.', items: [
    { id: 'advisor', t: 'FP / FN strategy advisor (interactive)', tags: 'reduce false positives,reduce false negatives,fp fn strategy,false alarm,missed detection,improve precision,improve recall', wide: true,
      s: 'Pick the task and the problem (optionally paste TP / FP / FN from your validation run). You get the levers in the order to try them: cheapest and safest first.', widget: 'fpfn' },
    { id: 'calc', t: 'Precision / recall / F-beta calculator', tags: 'precision recall calculator,f1 score calculator,mcc,confusion matrix calculator,f2 f0.5', wide: true,
      s: 'Enter confusion-matrix counts from your validation set.', widget: 'prcalc',
      f: [R`P=\frac{TP}{TP+FP}\;(\text{FP}\downarrow\Rightarrow P\uparrow),\;\;R=\frac{TP}{TP+FN}\;(\text{FN}\downarrow\Rightarrow R\uparrow),\;\;F_\beta=\frac{(1+\beta^2)PR}{\beta^2P+R}`] },
    { id: 'cost', t: 'Which error is more expensive?', tags: 'cost of errors,fp vs fn cost,f-beta choice,operating point business,precision recall tradeoff',
      h: T(['Use case', 'Costlier error', 'Target'], [['Medical screening, safety (pedestrians, PPE, fire)', 'FN (a miss)', 'recall ≥ 0.95–0.99 at the best achievable precision; F2'], ['Alerting / security / fraud review queues', 'FP (alert fatigue)', 'precision ≥ 0.9; F0.5'], ['Industrial defect inspection', 'usually FN (escaped defect) — but FPs cost re-inspection', 'recall target + max FP rate per 1000 parts'], ['Auto-labeling pipelines', 'FP (bad labels propagate)', 'high precision, humans fill recall'], ['Retail / analytics counts', 'balanced', 'F1 or count error']]),
      tip: 'Write the target down as "recall ≥ X at precision ≥ Y on dataset Z" before training. It decides the threshold, the loss weighting and which data to collect.' },
    { id: 'levers', t: 'Lever effects on FP and FN (cheat-sheet)', tags: 'threshold tuning,nms iou,class weights,focal loss alpha,tversky,background images,hard negative mining,lever effects', wide: true,
      h: T(['Lever', 'FP', 'FN', 'Notes'], [
        ['Confidence / probability threshold ↑', '↓', '↑', 'cheapest; tune on validation PR curve, per class if needed'],
        ['NMS IoU threshold ↓ (detection)', '↓ (duplicates)', '↑ (crowded objects)', 'or switch to NMS-free / Soft-NMS'],
        ['Max detections ↑ / queries ↑', '—', '↓', 'crowded scenes (300 → 1000)'],
        ['Background / negative images (0–10%)', '↓', '≈', 'Ultralytics tip: images with no labels reduce false alarms'],
        ['Hard-negative mining (look-alikes)', '↓', '≈', 'mine from production false alarms'],
        ['More / more diverse positives', '≈', '↓', 'target the missed conditions (size, light, occlusion)'],
        ['Complete labels (no missing boxes)', '↓ (apparent)', '↓', 'unlabelled objects teach the model to ignore them and count as FPs in eval'],
        ['Positive class weight / pos_weight ↑', '↑', '↓', 'imbalanced binary problems'],
        ['Focal loss (γ = 2) / α toward positives', '≈/↑', '↓', 'imbalance; check calibration afterwards'],
        ['Tversky α > β (seg)', '↓', '↑', 'α = 0.7, β = 0.3'],
        ['Tversky β > α / Focal Tversky (seg)', '↑', '↓', 'small / thin structures'],
        ['Input resolution ↑ / tiling (SAHI)', '≈', '↓ (small objects)', 'cost: latency'],
        ['Bigger / better pre-trained model', '↓', '↓', 'the only lever that improves both at a fixed threshold, together with better data'],
        ['TTA / ensembles', '↓', '↓', 'offline only; WBF for detection'],
        ['Temporal consistency (k of N frames, tracking)', '↓', '≈/↓', 'video; tracking also bridges short misses'],
        ['Min-area / geometry / ROI filters', '↓', '≈', 'post-processing rules from domain knowledge']]) },
    { id: 'analysis', t: 'Error analysis before changing anything', tags: 'error analysis,tide,fp categories,fn categories,failure analysis,fiftyone,confusion matrix',
      h: R`<p>Sample 50–100 FPs and 50–100 FNs from the validation set and put each in a bucket. The biggest bucket decides the fix.</p>` +
        T(['Bucket', 'Typical fix'], [['FP: background (nothing there)', 'background images, hard negatives, higher threshold'], ['FP: duplicate box on same object', 'NMS IoU ↓, NMS-free model'], ['FP: wrong class (class confusion)', 'more data for the confused pair, merge classes, per-class thresholds'], ['FP: poor localization (IoU < 0.5)', 'box loss weight ↑, higher res, better box head'], ['FP that is actually correct (missing label)', 'fix labels — this is an evaluation error, not a model error'], ['FN: small / far objects', 'resolution ↑, tiling, P2 head, less downscale aug'], ['FN: occluded / truncated', 'more occluded examples, copy-paste occluders, lower NMS suppression'], ['FN: rare class / rare appearance', 'collect targeted data, oversample, class weights'], ['FN: domain shift (night, blur, new camera)', 'collect from the new domain, photometric / blur / JPEG aug']]) +
        R`<p>For detection, TIDE-style breakdowns (classification, localization, both, duplicate, background, missed) quantify how much mAP each error type costs.</p>` },
    { id: 'workflow', t: 'Step-by-step workflow for fixing FPs / FNs', tags: 'workflow reduce errors,model improvement process,iterative improvement,data centric workflow', wide: true,
      h: R`<ol>
        <li><b>Freeze a trustworthy validation set:</b> representative of production, labelled carefully, split by group (camera / video / patient), with enough examples per class (≥ 50–100 instances for stable per-class numbers).</li>
        <li><b>Define the target</b> (e.g. recall ≥ 0.95 at precision ≥ 0.8) and measure the baseline PR curve per class.</li>
        <li><b>Error analysis</b> (card above): bucket 50–100 FPs and FNs.</li>
        <li><b>Cheap levers first:</b> threshold per class, NMS IoU, max detections, post-processing filters. Re-measure.</li>
        <li><b>Data:</b> fix labels, add hard negatives / background images for FPs, add targeted positives for FNs. This is usually the biggest win.</li>
        <li><b>Loss & sampling:</b> class weights, focal / Tversky settings, assignment top-k.</li>
        <li><b>Model:</b> resolution, larger model, better pre-training, longer schedule, EMA.</li>
        <li><b>Re-measure on the same validation set</b> after each change; keep a log of (change → P, R, per-class AP).</li>
        <li><b>Production monitoring:</b> sample live predictions weekly, review FPs / low-confidence cases, feed them back as training data.</li></ol>` },
    { id: 'inputs', t: 'What training data (valid inputs) to collect', tags: 'how much data,data collection guide,images per class,instances per class,background images,label quality,annotation guideline',
      h: T(['Input', 'Guideline'], [['Images per class', 'detection: ≥ 1,500 images and ≥ 10,000 labelled instances per class is the Ultralytics recommendation for strong results; fewer works with good pre-training (see Few-shot tab)'], ['Variety', 'cover production conditions: time of day, weather, camera angle and height, distance, motion blur, occlusion, compression'], ['Background images', '0–10% of the dataset with no objects, taken from the deployment scene — reduces FPs'], ['Hard negatives', 'look-alikes that the current model fires on (reflections, posters, similar products)'], ['Label consistency', 'written guideline: every instance labelled, tight boxes, occlusion / truncation policy, "ignore" regions; review 5–10% twice'], ['Class balance', 'aim for ≤ 1:10 between common and rare classes, or use sampling / weights'], ['Resolution', 'label and train at the resolution you deploy; objects should be ≥ ~16 px at the model input'], ['Split hygiene', 'no near-duplicates or same-scene frames across train / val / test']]) }
  ]},
  { id: 'strategy', title: 'Choosing a Strategy', items: [
    { id: 'datasize', t: 'Data size → training strategy', tags: 'transfer learning,fine-tuning,from scratch,small dataset,few-shot',
      h: T(['Labelled images', 'Strategy', 'Key hyperparameters'], [
        ['< 100 / class', 'Linear probe or few-shot on frozen foundation features (DINOv2/v3, CLIP, SigLIP); open-vocab detectors (Grounding DINO, YOLO-World) for zero-shot', 'LR 1e-3 head only; strong aug; k-NN baseline'],
        ['100 – 1k / class', 'Fine-tune pretrained model, maybe freeze early stages', 'LR 1e-4 (AdamW) / 1e-3 (SGD); LLRD 0.7; WD 0.05; 30–100 ep; heavy aug'],
        ['1k – 10k / class', 'Full fine-tune', 'LR 1e-4–5e-4; cosine; warmup 3–5 ep; mixup/cutmix for cls'],
        ['> 1M total', 'Pre-train / train from scratch possible', 'Full recipe (see below), 300+ ep, EMA'],
        ['Unlabelled pool large', 'Self-supervised pre-training (MAE/DINO) or pseudo-labelling (teacher → student)', 'Confidence threshold 0.5–0.9; iterate']]) },
    { id: 'compute', t: 'Compute budget → model size', tags: 'model selection,latency budget,edge,compute',
      h: T(['Deployment target', 'Detection', 'Classification', 'Segmentation'], [
        ['MCU / tiny edge', 'DEIMv2-Pico/Femto, YOLO26n int8', 'MobileNetV4-S, MCUNet', 'Lite-HRNet, PP-LiteSeg'],
        ['Mobile / Jetson Orin Nano', 'YOLO26n/s, RF-DETR-N, D-FINE-N', 'MobileNetV4-M, EfficientNet-B0, FastViT', 'YOLO26n-seg, SegFormer-B0'],
        ['Edge GPU (Orin AGX, T4)', 'YOLO26m, RF-DETR-S/M, DEIMv2-S/M', 'ConvNeXt-T, EfficientNetV2-S', 'RF-DETR-Seg-S, SegFormer-B2'],
        ['Server GPU, real-time', 'RF-DETR-L/2XL, DEIMv2-X, YOLO26x', 'ConvNeXt-B, Swin-B, ViT-B', 'Mask2Former-R50/Swin-B'],
        ['Offline / max accuracy', 'Co-DETR, DINO Swin-L/ViT-L + O365', 'EVA-02-L, DINOv3 7B probes', 'Mask2Former/OneFormer Swin-L, SAM-based']]) }
  ]},
  { id: 'hparams', title: 'Hyperparameter Rules of Thumb', items: [
    { id: 'lr', t: 'Learning rate — the #1 knob', tags: 'learning rate,lr tuning,lr finder,learning rate scheduler',
      h: R`<ul><li>Tune LR on a log grid ×3: {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}. Tune it <b>first</b>, then WD, then aug.</li>
        <li>Loss explodes/NaN early → LR too high or no warmup. Loss decreasing very slowly & train acc low → LR too low.</li>
        <li>Fine-tuning: 10–100× lower than from-scratch. Pretrained backbone + new head: head LR 10× backbone LR.</li>
        <li>Changing batch size ×k: SGD LR ×k; AdamW LR ×√k (up to ×k for moderate k).</li>
        <li>Transformer training unstable? lower LR, longer warmup, β₂=0.95, clip-norm 1.0, QK-norm.</li></ul>` },
    { id: 'bs', t: 'Batch size', tags: 'batch size,memory,gradient accumulation,generalization',
      h: R`<ul><li>Use the largest batch that fits for throughput; beyond ~1–4k (cls) returns diminish and generalization can drop without LR retuning.</li>
        <li>BatchNorm needs ≥ 16/GPU (else SyncBN, GN, or frozen BN). Detection: 2–8 img/GPU typical, total 16–64.</li>
        <li>Very small datasets: small batches (16–32) add useful gradient noise.</li>
        <li>Gradient accumulation matches effective batch but not BN statistics.</li></ul>` },
    { id: 'wdguide', t: 'Weight decay & regularization strength', tags: 'weight decay,dropout,drop path,overfitting',
      h: T(['Symptom / setting', 'Adjust'], [
        ['Train ≫ val (overfit)', '↑ WD (×2–5), ↑ drop-path, ↑ aug, mixup/cutmix, label smoothing, early stop, more data'],
        ['Train & val both poor (underfit)', '↓ WD, ↓ aug, ↓ dropout, bigger model, train longer, higher LR'],
        ['ViT from scratch on < 1M imgs', 'Don’t — use pretrained or ConvNet; else strong aug + WD 0.05–0.3 + SAM'],
        ['Short fine-tune', 'WD 0.01–0.05, drop-path 0.1, mixup off or light']]) },
    { id: 'epochs', t: 'Epochs & schedule length', tags: 'epochs,training length,iterations,budget',
      h: T(['Task / setting', 'Typical length'], [
        ['ImageNet ResNet-50 (classic / modern)', '90 ep / 300–600 ep (RSB A1)'], ['ViT/DeiT from scratch', '300 ep (DeiT), 800–1600 MAE pre-train'],
        ['Fine-tune cls on custom data', '20–100 ep'], ['YOLO from pretrained on custom', '100–300 ep, patience 50'],
        ['Faster/Mask R-CNN', '1× = 12 ep, 3× = 36 ep'], ['DETR / Deformable / DINO / RT-DETR', '500 / 50 / 12–36 / 72 ep'],
        ['Semantic seg (ADE20K)', '160k iters @ bs16'], ['Pose (COCO top-down)', '210–420 ep'], ['DiT / latent diffusion', '400k – 7M steps']]) },
    { id: 'res', t: 'Input resolution', tags: 'resolution,image size,imgsz,fixres,small objects',
      h: R`<ul><li>Accuracy ∝ resolution until the object scale saturates; FLOPs ∝ \(H\times W\).</li>
        <li><b>FixRes effect</b>: RandomResizedCrop makes objects look bigger in training → test at ~1.15× train resolution, or fine-tune at test res for a few epochs.</li>
        <li>Detection small objects: raise imgsz (640 → 1024/1280) or tile (SAHI); add P2 (stride 4) head.</li>
        <li>Train at the aspect ratio you deploy at (letterbox vs. stretch must match).</li></ul>` },
    { id: 'imbalance', t: 'Class imbalance playbook', tags: 'class imbalance,long tail,resampling,focal loss,class weights',
      h: T(['Imbalance', 'Try (in order)'], [
        ['Mild (1:10)', 'Nothing; monitor macro-F1 / per-class AP'],
        ['Moderate (1:100)', 'Class-balanced sampling (sqrt freq), weighted CE (class-balanced β=0.999), focal loss'],
        ['Severe / long-tail (1:1000+)', 'Repeat-factor sampling (LVIS, t=0.001), logit adjustment \\(z_c-\\tau\\log\\pi_c\\), decoupled training (cRT), Seesaw/Equalization loss'],
        ['Segmentation', 'Dice/Focal+CE combo, OHEM, crop sampling biased to rare classes'],
        ['Threshold-based decisions', 'Tune per-class thresholds on val; calibrate']]) },
    { id: 'smallobj', t: 'Small objects', tags: 'small object detection,sahi,tiling,p2 head,high resolution',
      h: R`<ul><li>Higher input res or SAHI slicing (overlap 0.2) + merging (NMS/NMM).</li><li>Add stride-4 (P2) level; reduce min anchor / assign more positives (ATSS, TAL topk ↑).</li>
        <li>Avoid strong downscale aug (mosaic scale range); use copy-paste of small instances.</li><li>Losses: NWD (normalized Wasserstein distance) instead of IoU for tiny boxes.</li></ul>` },
    { id: 'search', t: 'Hyperparameter search methods', tags: 'hyperparameter search,optuna,bayesian optimization,asha,hyperband,random search',
      h: T(['Method', 'When'], [
        ['Manual, one-at-a-time (LR first)', 'Always the first pass; cheap intuition'],
        ['Random search (log-uniform)', 'Beats grid when few params matter'],
        ['Bayesian (TPE / Optuna, GP)', '≤ 10 params, expensive trials'],
        ['ASHA / Hyperband (early stopping)', 'Many cheap-to-evaluate configs; Ray Tune'],
        ['PBT (population-based)', 'Schedules (LR/aug) that evolve; RL, large clusters'],
        ['Proxy tuning (μP)', 'Tune small width, transfer LR to large model']]),
      code: R`
import optuna
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 3e-3, log=True)
    wd = trial.suggest_float('wd', 1e-4, 0.3, log=True)
    dp = trial.suggest_float('drop_path', 0.0, 0.3)
    return train_and_eval(lr=lr, wd=wd, drop_path=dp, epochs=20)   # return val metric
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=40)` }
  ]},
  { id: 'debug', title: 'Debugging & Troubleshooting', items: [
    { id: 'checklist', t: 'Training sanity checklist', tags: 'debugging,sanity check,overfit one batch',
      h: R`<ol><li>Visualize the data <b>after</b> augmentation with labels drawn (boxes, masks, keypoints). Most bugs are here.</li>
        <li>Check initial loss: CE ≈ \(\ln K\); focal with prior 0.01 ≈ small.</li><li>Overfit one batch (≈ 0 loss in 100–300 steps, no aug). If not → model/loss bug.</li>
        <li>Verify normalization mean/std and RGB vs BGR match the pretrained weights.</li><li>Confirm eval pipeline = train pipeline (resize, letterbox, normalization).</li>
        <li>Log LR, grad-norm, loss components, and a few predictions every N steps.</li></ol>` },
    { id: 'symptoms', t: 'Symptom → cause → fix', tags: 'nan loss,divergence,troubleshooting,loss not decreasing',
      h: T(['Symptom', 'Likely cause', 'Fix'], [
        ['Loss NaN/Inf', 'LR too high, fp16 overflow, log(0), bad boxes (w≤0)', 'warmup, bf16, clamp eps, validate labels, clip-norm'],
        ['Loss plateaus immediately', 'LR too low/high, frozen params, wrong labels', 'LR range test; check requires_grad & optimizer param groups'],
        ['Val metric ≪ train from epoch 1', 'train/eval preprocessing mismatch, BN in train mode, leakage', 'model.eval(); compare pipelines'],
        ['Periodic loss spikes', 'bad samples, Adam β₂ too high, LR too high', 'β₂=0.95, lower LR, skip-batch on spikes'],
        ['Good mAP50, poor mAP50-95', 'box localization weak', '↑ box loss weight, DFL/GIoU, higher res, longer training'],
        ['Many duplicate detections', 'NMS IoU too high / one-to-many head', 'lower NMS IoU (0.5–0.6) or NMS-free (one-to-one) head'],
        ['Great val, poor in production', 'domain shift (camera, light, compression)', 'collect prod data; photometric/JPEG aug; monitor drift'],
        ['GPU underutilized', 'dataloader bottleneck', 'num_workers, pin_memory, persistent_workers, DALI, cache images, channels_last']]) }
  ]},
  { id: 'recipes', title: 'Ready-to-use Training Recipes', items: [
    { id: 'r-cls', t: 'Recipe: fine-tune a classifier', tags: 'classification recipe,timm,fine-tune',
      h: T(['Hyperparameter', 'Value'], [['Model', 'ConvNeXt-T / EfficientNetV2-S / ViT-B (DINOv2 or IN-21k weights)'], ['Optimizer', 'AdamW, LR 1e-4 (bs 64), WD 0.05'], ['Schedule', '5-ep warmup + cosine, 30–50 ep'], ['LLRD / drop-path', '0.75 (ViT) / 0.1'], ['Aug', 'RRC(0.35,1), flip, TrivialAugment, RandomErasing 0.25; mixup/cutmix if > 10k imgs'], ['Loss', 'CE + label smoothing 0.1'], ['Extras', 'EMA 0.9998, AMP bf16, image size 224→288 test']]),
      code: R`
# timm one-liner
python train.py /data --model convnext_tiny.fb_in22k --pretrained --num-classes 10 \
  --opt adamw --lr 1e-4 --weight-decay 0.05 --sched cosine --warmup-epochs 5 --epochs 50 \
  --drop-path 0.1 --smoothing 0.1 --aa rand-m9-mstd0.5 --reprob 0.25 --mixup 0.2 --cutmix 0.5 \
  --model-ema --amp -b 64` },
    { id: 'r-yolo', t: 'Recipe: YOLO on a custom dataset', tags: 'yolo recipe,ultralytics,custom detection,hyp',
      h: T(['hyp', 'Default', 'When to change'], [['lr0 / lrf', '0.01 / 0.01 (SGD)', 'AdamW: lr0 0.001–0.002'], ['momentum / wd', '0.937 / 5e-4', ''], ['warmup_epochs', '3', 'more for small batch'], ['box / cls / dfl gains', '7.5 / 0.5 / 1.5', '↑ box for localization-critical tasks; ↑ cls for many classes'], ['mosaic / close_mosaic', '1.0 / 10', 'off for tiny datasets with fixed layout'], ['mixup / copy_paste', '0 / 0', '0.1 / 0.1–0.3 for large models / seg'], ['hsv_h,s,v', '0.015, 0.7, 0.4', 'lower for color-critical classes'], ['fliplr / degrees', '0.5 / 0', 'flipud 0.5 + degrees for aerial'], ['imgsz', '640', '1024–1280 small objects'], ['epochs / patience', '100 / 100', '300 for from-scratch']]),
      code: R`
from ultralytics import YOLO
model = YOLO('yolo26s.pt')            # pretrained COCO weights
model.train(data='data.yaml', epochs=150, imgsz=640, batch=32, optimizer='auto',
            cos_lr=True, close_mosaic=10, patience=50, amp=True)
metrics = model.val(); model.export(format='onnx')   # or 'engine' for TensorRT` },
    { id: 'r-detr', t: 'Recipe: DETR-family (RT-DETR / D-FINE / RF-DETR)', tags: 'detr recipe,rt-detr,rf-detr,fine-tune',
      h: T(['Hyperparameter', 'Value'], [['Optimizer', 'AdamW, LR 1e-4 (head), 1e-5 backbone, WD 1e-4'], ['Batch', '16 (4 GPUs × 4)'], ['Schedule', '72 ep RT-DETR; fine-tune custom 30–100 ep, EMA 0.9999, warmup 2k iters'], ['Aug', 'photometric, zoom-out, IoU crop, flip, multi-scale 480–800; stop strong aug last few epochs (DEIM)'], ['Loss', 'VFL/focal cls + L1(5) + GIoU(2) (+FGL/DDF for D-FINE)'], ['Queries', '300 (increase for crowded scenes)'], ['Clip-norm', '0.1']]) },
    { id: 'r-seg', t: 'Recipe: semantic segmentation', tags: 'segmentation recipe,mmsegmentation,segformer recipe',
      h: T(['Hyperparameter', 'CNN (DeepLabv3+)', 'Transformer (SegFormer/Mask2Former)'], [['Optimizer', 'SGD 0.01, m 0.9, WD 5e-4', 'AdamW 6e-5 / 1e-4, WD 0.01/0.05, backbone ×0.1'], ['Schedule', 'poly 0.9, 80–160k it', 'poly/linear, 1.5k warmup, 160k it'], ['Crop', '512×512 (ADE), 512×1024 / 1024² (Cityscapes)', 'same'], ['Aug', 'random scale 0.5–2.0, crop, flip, photometric', 'same + LSJ for Mask2Former'], ['Loss', 'CE (+OHEM, aux 0.4)', 'CE / mask BCE+Dice (Mask2Former)'], ['Inference', 'sliding window, flip+MS TTA', 'same']]) },
    { id: 'r-pose', t: 'Recipe: top-down pose', tags: 'pose recipe,mmpose,rtmpose recipe,hrnet recipe',
      h: T(['Hyperparameter', 'Value'], [['Input', '256×192 (384×288 for accuracy)'], ['Optimizer', 'Adam 5e-4 (HRNet), AdamW 4e-3 + cosine (RTMPose)'], ['Schedule', '210 ep, drops at 170/200 (HRNet); 420 ep RTMPose'], ['Aug', 'half-body 0.3, scale ±35%, rotate ±40–80°, flip (swap L/R indices!), RandomErasing/Cutout'], ['Heatmap', 'Gaussian σ=2 (64×48), UDP encoding, flip-test at inference'], ['Detector', 'person AP ≈ 56 human detector; GT boxes for upper bound']]) },
    { id: 'r-lora', t: 'Recipe: diffusion LoRA / DreamBooth', tags: 'lora recipe,dreambooth,diffusion fine-tune,sdxl,flux',
      h: T(['Hyperparameter', 'Value'], [['Rank / alpha', 'r=16–64, α=r (style) ; r=4–16 (subject)'], ['LR', '1e-4 (LoRA, AdamW), 1e-6–5e-6 (full DreamBooth)'], ['Steps', '800–3000 (≈100 steps per training image)'], ['Batch', '1–4, grad-accum to 4'], ['Captions', 'unique token + class ("sks dog"), prior-preservation loss λ=1'], ['Resolution', 'native (1024 SDXL/FLUX), bucketing by aspect ratio'], ['Monitoring', 'fixed-seed sample grid every 200 steps; stop before overfit (texture copying)']]) }
  ]}
  ]
});
})();

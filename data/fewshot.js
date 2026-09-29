(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'fewshot', title: 'Few-shot & Data-efficient Learning', short: 'Few-shot', icon: '🧪',
  blurb: 'N-way K-shot setup, what to use for 0 / 1 / 5 / 50 shots, metric learning, meta-learning, CLIP adapters, few-shot detection & segmentation.',
  intro: 'Learning new classes from a handful of labelled examples. In practice today the winning recipe is usually <b>a strong pre-trained foundation model + a light adaptation</b>; classic meta-learning is mostly a research benchmark. Start with the decision guide.',
  sections: [
  { id: 'setup', title: 'Setup & Decision Guide', items: [
    { id: 'terms', t: 'N-way K-shot, support & query sets', tags: 'few-shot learning,n-way k-shot,support set,query set,episodic training,base classes,novel classes,zero-shot,one-shot',
      f: [R`\text{episode: } \mathcal S=\{(x_i,y_i)\}_{i=1}^{N\cdot K},\;\;\mathcal Q=\{x_j\}\;\;\text{with the same }N\text{ classes}`],
      h: T(['Term', 'Meaning'], [['N-way K-shot', 'N classes, K labelled examples each (e.g. 5-way 1-shot)'], ['Support set', 'the K labelled examples per class you adapt with'], ['Query set', 'images to classify / detect with the adapted model'], ['Base vs novel classes', 'plenty of data for base classes; novel classes have only K shots'], ['Zero-shot', 'no images, only a class name / description (text prompt)'], ['Generalized few-shot (GFSL)', 'test on base + novel classes together — the realistic setting'], ['Episodic training', 'meta-training on many simulated N-way K-shot tasks sampled from base classes']]) },
    { id: 'guide', t: 'What to use for 0, 1, 5, 20, 100 shots (decision guide)', tags: 'few-shot decision guide,how many images,small dataset strategy,which approach few-shot', wide: true,
      h: T(['Labelled images per class', 'Classification', 'Detection', 'Segmentation'], [
        ['0 (names only)', 'CLIP / SigLIP zero-shot with prompt ensembles', 'Grounding DINO, YOLO-World / YOLOE, OWLv2 (text prompts)', 'SAM 3 (text prompt), Grounded-SAM'],
        ['1–5', 'k-NN or prototypes on frozen DINOv2/v3 or CLIP features; Tip-Adapter', 'visual-exemplar prompting (OWLv2 image query, T-Rex2), SAM 3 exemplars; or auto-label + human check', 'PerSAM / Matcher (DINOv2 + SAM), SegGPT in-context'],
        ['5–20', 'linear probe / cosine classifier on frozen features; CoOp / CLIP-Adapter', 'fine-tune head + neck of a pre-trained detector (TFA-style), strong aug, freeze backbone', 'fine-tune decoder on frozen backbone; SAM-assisted labels'],
        ['20–100', 'LP-FT: probe, then full fine-tune with LLRD and strong aug', 'full fine-tune of RF-DETR / YOLO26 from COCO weights; pseudo-label unlabelled data', 'fine-tune Mask2Former / SegFormer; copy-paste aug'],
        ['100+', 'standard fine-tuning (see Hyperparameter guide)', 'standard fine-tuning', 'standard fine-tuning']]),
      tip: 'Always measure a zero-shot / k-NN baseline first — it takes minutes and tells you whether labelling more data or changing the model is the better investment.' },
    { id: 'bench', t: 'Few-shot benchmarks & evaluation', tags: 'miniimagenet,tieredimagenet,meta-dataset,fsod benchmark,coco few-shot,nap,novel ap',
      h: T(['Benchmark', 'Protocol'], [['miniImageNet / tieredImageNet', '5-way 1-shot and 5-shot accuracy, mean ± 95% CI over 600+ episodes'], ['Meta-Dataset', '10 domains, variable ways/shots — tests cross-domain transfer'], ['COCO FSOD', '60 base + 20 novel (VOC) classes; 10 / 30-shot; report novel AP (nAP)'], ['PASCAL VOC FSOD', '3 splits of 15 base / 5 novel; 1/2/3/5/10-shot nAP50'], ['PASCAL-5ⁱ / COCO-20ⁱ', 'few-shot segmentation, 1 / 5-shot mIoU over 4 folds'], ['ODinW / RF100-VL', 'real-world domain transfer for detectors (few-shot & full)']]),
      f: [R`\text{CI}_{95\%}=\bar a\pm1.96\,\frac{s}{\sqrt{n_{episodes}}}`] }
  ]},
  { id: 'metric', title: 'Metric-based Methods', intro: 'Learn an embedding in which a simple distance to the support examples classifies the query.', items: [
    { id: 'proto', t: 'Prototypical Networks', tags: 'prototypical networks,protonet,prototype,class mean embedding,euclidean distance',
      f: [R`c_k=\frac1{|\mathcal S_k|}\sum_{(x_i,y_i)\in\mathcal S_k}f_\phi(x_i)`, R`p(y=k\mid x)=\frac{\exp(-d(f_\phi(x),c_k))}{\sum_{k'}\exp(-d(f_\phi(x),c_{k'}))},\;\;d=\|\cdot\|_2^2`],
      dgc: [{ t: 'Prototypical classification', k: 'backbone', ops: ['Support images (K per class) → encoder f', 'Average per class → prototypes c₁ … c_N', 'Query image → f(x)', 'Distances to each prototype', 'Softmax over −distance → class'] }],
      h: R`<p>Training-free at test time: adding a new class = computing one mean vector. With a strong frozen encoder (DINOv2/v3, CLIP) this "nearest class mean" baseline is hard to beat for 1–5 shots.</p>` },
    { id: 'siamese', t: 'Siamese / Matching / Relation networks', tags: 'siamese network,matching networks,relation network,contrastive loss,one-shot verification',
      f: [R`\text{Siamese: } p(\text{same})=\sigma\big(w^{\top}|f(x_1)-f(x_2)|\big)`, R`\text{Matching: } \hat y=\sum_i a(x,x_i)\,y_i,\;\;a=\mathrm{softmax}_i\big(\cos(f(x),g(x_i))\big)`, R`\text{Relation: } r_{k}=g_\psi\big([f(x);c_k]\big)\in[0,1]`],
      h: R`<p>Siamese nets learn pairwise "same / different" (face verification, signature matching, re-ID). Matching networks use attention over the support set. Relation networks learn the distance function itself.</p>` },
    { id: 'cosine', t: 'Cosine classifier (Baseline++) & normalized embeddings', tags: 'cosine classifier,baseline++,normalized softmax,temperature scaling classifier',
      f: [R`s_k=\tau\cdot\frac{f(x)^{\top}w_k}{\|f(x)\|\,\|w_k\|},\;\;\tau\approx10\text{–}30`],
      h: R`<p>A linear head on L2-normalized features with a temperature reduces intra-class variance and makes novel-class weights comparable to base-class weights — a strong, simple baseline used in TFA for detection too.</p>` }
  ]},
  { id: 'meta', title: 'Optimization-based Meta-learning', items: [
    { id: 'maml', t: 'MAML / first-order MAML / Reptile', tags: 'maml,model-agnostic meta-learning,reptile,first order maml,meta-learning,learning to learn',
      f: [R`\theta'_i=\theta-\alpha\nabla_\theta\mathcal L_{\mathcal T_i}^{sup}(\theta)`, R`\theta\leftarrow\theta-\beta\nabla_\theta\sum_i\mathcal L_{\mathcal T_i}^{query}(\theta'_i)`, R`\text{Reptile: }\;\theta\leftarrow\theta+\epsilon\,(\tilde\theta_i-\theta)`],
      h: R`<p>Learn an initialization that adapts to a new task in a few gradient steps. Second-order gradients make MAML expensive; FO-MAML and Reptile drop them. <b>Practical note:</b> with large pre-trained models, plain fine-tuning or linear probing matches or beats meta-learning on most real tasks, so meta-learning is rarely worth the complexity in production.</p>` }
  ]},
  { id: 'transfer', title: 'Transfer-learning & Foundation-model Approaches', items: [
    { id: 'probe', t: 'Frozen features: k-NN, linear probe, LP-FT', tags: 'linear probe,k-nn,frozen backbone,lp-ft,dinov2 features,few-shot fine-tuning',
      h: R`<ol><li>Extract features once with a frozen foundation encoder (DINOv2/v3 ViT-B/L, CLIP / SigLIP image tower).</li><li>k-NN (k = 1–20, cosine) or logistic regression / linear probe on those features.</li><li>If you have ≥ 20–50 shots: LP-FT — train the linear head first, then fine-tune everything with a 10× lower LR, LLRD 0.7, strong augmentation, few epochs.</li></ol>`,
      code: R`
feats = torch.cat([F.normalize(encoder(x), dim=-1) for x in loader])      # frozen DINOv2 / CLIP
protos = torch.stack([feats[y == k].mean(0) for k in range(K)])          # 1 vector per class
pred = (F.normalize(encoder(q), dim=-1) @ F.normalize(protos, dim=-1).T).argmax(-1)` },
    { id: 'clip', t: 'CLIP zero-shot, prompt learning & adapters', tags: 'clip zero-shot,prompt ensemble,coop,cocoop,clip-adapter,tip-adapter,prompt tuning,vpt',
      f: [R`\text{CoOp prompt: }\;t_k=[V]_1[V]_2\dots[V]_M[\text{CLASS}_k]\;\;(\text{learned context vectors})`, R`\text{Tip-Adapter: }\;\text{logits}=f\,W_c^{\top}+\alpha\,\varphi\big(fF_{train}^{\top}\big)L_{train},\;\;\varphi(x)=e^{-\beta(1-x)}`],
      h: T(['Method', 'Trainable', 'Shots', 'Idea'], [['Zero-shot CLIP', 'none', '0', '"a photo of a {class}." + prompt ensembles'], ['Tip-Adapter', 'none (or cache fine-tuned)', '1–16', 'cache the few-shot features as a key-value memory'], ['CoOp / CoCoOp', 'prompt context vectors', '1–16', 'learn the text prompt; CoCoOp conditions it on the image for better generalization'], ['CLIP-Adapter', 'small MLP on features', '4–16', 'residual adapter blended with frozen features'], ['VPT / LoRA', 'prompt tokens / low-rank adapters in the image encoder', '16+', 'parameter-efficient fine-tuning']]) },
    { id: 'peft', t: 'Parameter-efficient fine-tuning for few shots', tags: 'peft,lora,adapters,bitfit,visual prompt tuning,few-shot fine-tune overfitting',
      h: R`<p>With few examples, full fine-tuning overfits. Train &lt; 1–5% of parameters: LoRA on attention projections (r = 4–16), adapters, BitFit (biases only) or prompt tokens. Keep the pre-trained features intact, use strong augmentation and early stopping on a (small) validation split.</p>` }
  ]},
  { id: 'fsod', title: 'Few-shot Object Detection', items: [
    { id: 'tfa', t: 'Fine-tuning based: TFA & DeFRCN', tags: 'few-shot object detection,fsod,tfa,frustratingly simple,defrcn,two-stage fine-tuning',
      h: R`<ol><li><b>Base training</b> on classes with plenty of data.</li><li><b>Few-shot fine-tuning</b> on a balanced set of K shots for base + novel classes, freezing the backbone, RPN and most of the head; only the last box classifier and regressor are trained (TFA), with a cosine classifier.</li></ol><p>DeFRCN decouples gradients between RPN and RoI head and adds prototype calibration. Meta-learning detectors (FSRW, Meta R-CNN) re-weight features with support-image vectors but are generally no better than well-tuned fine-tuning.</p>` },
    { id: 'modern', t: 'Modern practice: open-vocabulary & visual prompts', tags: 'open vocabulary few-shot,visual prompt detection,image exemplar,owlv2 image query,t-rex2,yoloe visual prompt,grounding dino fine-tune',
      h: T(['Approach', 'How', 'When'], [['Text prompts (zero-shot)', 'Grounding DINO / YOLO-World / YOLOE with class names or descriptions', 'common objects with good names'], ['Visual prompts / exemplars', 'draw a box on 1–5 example images; OWLv2 image-conditioned, T-Rex2, YOLOE visual prompts, SAM 3 exemplars', 'objects hard to describe in words (parts, defects, custom products)'], ['Auto-label → distil', 'open-vocab model labels unlabelled images; you verify; train YOLO26 / RF-DETR on them', 'you need real-time deployment'], ['Few-shot fine-tune', 'fine-tune a COCO / O365-pretrained detector on 10–50 images per class with frozen backbone, strong aug (mosaic, copy-paste), 100–300 epochs', 'closed-set, stable classes']]),
      tip: 'For 10–50 images per class, RF-DETR and DINOv2-backbone detectors transfer especially well (RF100-VL); freeze the backbone for the first epochs and keep a few images for validation.' }
  ]},
  { id: 'fsseg', title: 'Few-shot Segmentation', items: [
    { id: 'map', t: 'Prototype-based segmentation (PANet, masked average pooling)', tags: 'few-shot segmentation,panet,masked average pooling,prototype segmentation,pascal-5i',
      f: [R`p_c=\frac{\sum_{x,y}F^{s}_{x,y}\,\mathbb 1[M^{s}_{x,y}=c]}{\sum_{x,y}\mathbb 1[M^{s}_{x,y}=c]},\;\;\hat M^{q}_{x,y}=\arg\max_c\cos(F^{q}_{x,y},p_c)`],
      h: R`<p>Pool support features inside the support mask to get a class prototype, then label each query pixel by cosine similarity. PANet adds prototype alignment (query → support) as regularization.</p>` },
    { id: 'sam', t: 'Foundation approaches: PerSAM, Matcher, SegGPT, SAM 3', tags: 'persam,matcher,seggpt,in-context segmentation,sam 3 exemplar,one-shot segmentation',
      h: T(['Method', 'Shots', 'Idea'], [['PerSAM (training-free)', '1', 'locate the target via feature similarity to the reference mask, then prompt SAM with positive/negative points'], ['Matcher', '1–5', 'DINOv2 correspondence → prompts for SAM, robust matching and mask merging'], ['SegGPT / Painter', '1–few', 'in-context: stitch example image + mask with the query, the model "paints" the mask'], ['SAM 3', '0–few', 'text or image-exemplar concept prompts return all matching instances']]) }
  ]},
  { id: 'data', title: 'Data-side Strategies (when you can’t get more labels)', items: [
    { id: 'aug', t: 'Augmentation, synthetic data & copy-paste', tags: 'synthetic data,diffusion generated data,copy-paste,augmentation few-shot,domain randomization',
      h: R`<ul><li>Stronger geometric + photometric augmentation; mixup / cutmix for classification; mosaic + copy-paste for detection and instance seg.</li><li><b>Synthetic images:</b> text-to-image / image-editing models (FLUX, SDXL + ControlNet, inpainting) to create variations; render 3D assets with domain randomization; paste object crops (SAM-cut) onto real backgrounds.</li><li>Validate on <i>real</i> images only — synthetic data can inflate validation if it leaks into it.</li></ul>` },
    { id: 'semi', t: 'Self-training, SSL pre-training & active learning', tags: 'semi-supervised,pseudo-labeling,self-supervised pretraining in domain,active learning,label efficiency',
      h: R`<ol><li><b>In-domain self-supervised pre-training</b> (DINO / MAE on your unlabelled images) before fine-tuning — large gains for unusual domains (medical, aerial, industrial).</li><li><b>Pseudo-labelling</b> with a teacher (see Training › Knowledge Distillation › Teacher–student for unlabelled data).</li><li><b>Active learning:</b> label the images the model is most uncertain about (entropy, margin, disagreement between two models) or most diverse (core-set on embeddings) — typically 2–3× more label-efficient than random.</li><li><b>Label faster:</b> SAM-assisted masks, detector pre-labels with human correction.</li></ol>` },
    { id: 'recipe', t: 'Few-shot fine-tuning recipe & pitfalls', tags: 'few-shot recipe,small data training hyperparameters,overfitting small dataset,few-shot validation', wide: true,
      h: T(['Hyperparameter', 'Few-shot setting (≈ 5–50 images / class)'], [['Backbone', 'strongest pre-trained you can deploy (DINOv2/v3, CLIP, COCO/O365 detector weights)'], ['Frozen parts', 'backbone frozen for the first 25–50% of epochs (or entirely for ≤ 10 shots)'], ['LR', 'head 1e-3 (AdamW), backbone 1e-5 when unfrozen; warmup 5–10%, cosine'], ['Epochs', 'more epochs, but early stop on validation (often 50–300 for detection)'], ['Augmentation', 'strong: RandAugment / mosaic / copy-paste / mixup; avoid augmentations that change label semantics'], ['Regularization', 'weight decay 0.05, drop-path 0.1, label smoothing 0.1'], ['Validation', 'keep ≥ 1–2 images per class out; repeat with different splits (k-fold) and report mean ± std']]) +
        R`<p><b>Pitfalls:</b> judging a model on 5 validation images (huge variance) · test classes leaking into base training or pre-training data · reporting only novel-class accuracy when base classes also matter (use generalized few-shot evaluation) · tuning thresholds on the test set.</p>` }
  ]}
  ]
});
})();

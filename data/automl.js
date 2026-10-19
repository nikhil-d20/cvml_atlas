(function () {
const R = String.raw, T = CV.tbl;
const L = (url, label) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${label || url.replace('https://github.com/', '')}</a>`;
CV.tab({
  id: 'automl', title: 'AutoML for Computer Vision', short: 'AutoML', icon: '🤖',
  blurb: 'What to automate in a CV pipeline, search algorithms (TPE, GP-BO, ASHA, PBT, NAS), current tools with repo links, recipes and best practices.',
  intro: 'AutoML automates the choices you would otherwise tune by hand: hyperparameters, augmentation, model size, architecture, thresholds, ensembling. Used well, it finds better models for the same GPU budget and documents why. Used badly, it overfits the validation set and burns compute. Repo status checked on GitHub in Oct 2026.',
  sections: [
  { id: 'what', title: 'Where AutoML Helps in Computer Vision', items: [
    { id: 'map', t: 'What to automate (and what not to)', tags: 'automl computer vision,what to automate,hyperparameter optimization,hpo,nas,augmentation search,threshold optimization', wide: true,
      h: T(['Layer', 'What AutoML searches', 'Typical gain', 'Tools'], [
        ['Hyperparameters (HPO)', 'LR, weight decay, warmup, batch, epochs, loss gains (box/cls/dfl), label smoothing, EMA', 'often +1–3 mAP / top-1 over defaults on a new dataset', 'Optuna, Ray Tune, Ultralytics <code>model.tune</code>, SMAC3, Syne Tune'],
        ['Augmentation policy', 'mosaic / mixup / copy-paste probabilities, HSV, scale, RandAugment N/M', 'large on small or unusual datasets', 'Optuna / Ultralytics tuner; RandAugment / TrivialAugment (search-free)'],
        ['Input resolution & model size', 'imgsz 480–1280, n/s/m/l/x, backbone choice', 'decides the accuracy–latency trade-off', 'multi-objective HPO (accuracy vs latency)'],
        ['Architecture (NAS)', 'depth, width, kernel sizes, decoder layers, queries, patch size', 'Pareto family for your hardware', 'RF-DETR weight-sharing NAS, Once-for-All, NePS'],
        ['Post-processing', 'per-class confidence thresholds, NMS IoU, max detections, TTA', 'directly targets FP / FN at deploy time', 'Optuna on a validation set (cheap: no retraining)'],
        ['Model selection & ensembling', 'which pre-trained models, weighted ensembles', 'robust strong baselines with no tuning', 'AutoGluon MultiModalPredictor, AutoTrain'],
        ['Whole pipeline (agents)', 'LLM agents write and iterate training code', 'fast baselines, exploration', 'AutoGluon Assistant (MLZero), AIDE']]),
      tip: 'Do not AutoML your way around bad data. Fix labels and splits first (see the FP/FN Playbook). Search spaces only help when the validation set reflects production.' },
    { id: 'when', t: 'When AutoML is worth it', tags: 'when to use automl,automl budget,automl cost benefit',
      h: R`<ul><li><b>Yes:</b> a new domain where defaults are unproven; a fixed deployment budget (latency / memory) to maximize accuracy within; recurring retraining that can reuse the search; post-processing thresholds (cheap and high-impact).</li><li><b>Maybe:</b> fine-tuning a well-tuned recipe (YOLO, RF-DETR) on a typical dataset — defaults are already good; tune only LR, epochs, imgsz and augmentation strength.</li><li><b>No:</b> when one training run takes days and you can afford fewer than ~10 runs. Use published recipes plus manual one-at-a-time tuning instead.</li></ul>`,
      f: [R`\text{GPU-hours}\approx N_{trials}\times\bar t_{trial}`, R`\bar t_{trial}\ll t_{full}\;\text{with multi-fidelity}`] },
    { id: 'cost', t: 'AutoML budget estimator (interactive)', tags: 'automl cost estimate,asha budget,successive halving cost,gpu hours hpo', wide: true,
      s: 'Estimate GPU-hours and wall-clock for a search with and without successive halving (ASHA / Hyperband-style early stopping).', widget: 'automlcost' }
  ]},
  { id: 'algos', title: 'Search Algorithms (how AutoML decides what to try)', items: [
    { id: 'basic', t: 'Grid, random & quasi-random search', tags: 'grid search,random search,sobol,qmc,quasi random,baseline search',
      f: [R`P(\text{hit top-}q)=1-(1-q)^n`, R`q=5\%,\;n=60\;\Rightarrow\;95\%`],
      h: R`<p>Random search beats grid search when only a few hyperparameters matter, because it tries more distinct values of each. Quasi-random (Sobol / QMC) covers the space more evenly. These are the right <b>baseline</b> and the right first 10–20 trials before model-based search.</p>` },
    { id: 'tpe', t: 'TPE (Tree-structured Parzen Estimator)', tags: 'tpe,tree parzen estimator,optuna default sampler,multivariate tpe,hyperopt',
      f: [R`\ell(x)=p(x\mid y<y^*),\;\;g(x)=p(x\mid y\ge y^*)`, R`x_{next}=\arg\max_x\;\ell(x)/g(x)`],
      h: R`<p>Models "good" and "bad" regions with density estimators and samples where the ratio is highest. It handles conditional and categorical spaces well and is cheap. It is Optuna's default; in <b>Optuna 5</b> the defaults are multivariate TPE with the constant-liar strategy for parallel workers, and TPE also replaced NSGA-II as the multi-objective default.</p>` },
    { id: 'gp', t: 'Gaussian-process Bayesian optimization', tags: 'bayesian optimization,gaussian process,expected improvement,gpsampler,botorch,acquisition function,qlogei',
      f: [R`f\sim\mathcal{GP}(m,k),\;\;\text{EI}(x)=\mathbb E\big[\max(0,f(x)-f^*)\big]`, R`x_{next}=\arg\max_x\alpha(x)`],
      h: R`<p>Most sample-efficient for small continuous spaces (≤ ~10–20 dims) with expensive trials. Optuna 5's <code>GPSampler</code> is stable and supports batch acquisition (qLogEI, constrained and multi-objective variants qLogEHVI) so parallel trials do not duplicate work. BoTorch / Ax and Google Vizier are the research-grade options.</p>` },
    { id: 'cmaes', t: 'CMA-ES & evolutionary / genetic search', tags: 'cma-es,evolution strategy,genetic algorithm,ultralytics tuner mutation,evolutionary search',
      f: [R`x_k\sim\mathcal N\big(m,\sigma^2C\big),\;\;m\leftarrow\sum_{i=1}^{\mu}w_i\,x_{i:\lambda}`],
      h: R`<p>Good for continuous spaces with many trials and parallel workers. The Ultralytics tuner is a genetic algorithm: each iteration picks a parent from up to the 9 fittest configurations (weighted by fitness), mutates roughly half of the parameters with Gaussian steps, trains, and records the fitness.</p>` },
    { id: 'mf', t: 'Multi-fidelity: Successive Halving, Hyperband, ASHA, BOHB, DEHB', tags: 'asha,hyperband,successive halving,bohb,dehb,multi-fidelity,early stopping trials,pruning',
      f: [R`\text{rung }i:\;\; n_i=\Big\lfloor\frac{n}{\eta^{i}}\Big\rfloor\text{ trials at budget } r\,\eta^{i}\;\;(\eta=3)`, R`\text{cost}\approx n\,r\,(s+1),\;\;s=\log_\eta(R/r)\;\;\text{vs}\;\;n\,R\text{ without halving}`],
      h: R`<p>Train many configurations briefly and promote only the best \(1/\eta\) to longer budgets (epochs, image size, data subset). <b>ASHA</b> is the asynchronous version: no waiting for a rung to fill, ideal for clusters. <b>BOHB / DEHB</b> combine this with model-based proposals. In Optuna use <code>HyperbandPruner</code> / <code>SuccessiveHalvingPruner</code>; in Ray Tune <code>ASHAScheduler</code>.</p>`,
      tip: 'Check that the low-fidelity ranking predicts the high-fidelity one: run 5–10 configs at both budgets and compute rank correlation. Augmentation-heavy and long-schedule effects often only show late.' },
    { id: 'pbt', t: 'Population-Based Training (PBT)', tags: 'pbt,population based training,exploit explore,schedule search',
      h: R`<p>Train a population in parallel; periodically the worst members copy weights from the best (exploit) and perturb their hyperparameters (explore). The result is a <b>schedule</b> (e.g. augmentation strength or LR changing over training), not a single value. It needs many GPUs at once. Ray Tune provides <code>PopulationBasedTraining</code> / PB2.</p>` },
    { id: 'mo', t: 'Multi-objective & constrained search (accuracy vs latency)', tags: 'multi-objective,pareto front,nsga-ii,latency constraint,hardware aware,accuracy latency tradeoff,qlogehvi',
      f: [R`\min_x\big(-\text{mAP}(x),\;\text{latency}(x)\big)\Rightarrow\text{Pareto front}`, R`\text{or}\;\max_x\text{mAP}(x)\;\;\text{s.t.}\;\text{latency}(x)\le\tau`],
      h: R`<p>Real CV deployments have budgets. Search for the Pareto front of accuracy versus latency (measured on the <b>target device</b> with TensorRT / ONNX, not FLOPs), then pick the best point under your limit. Optuna supports multiple objectives (<code>directions=["maximize", "minimize"]</code>); in Optuna 5, constraints are declared inside the objective with <code>trial.set_constraint()</code> (the sampler <code>constraints_func</code> argument is deprecated).</p>` },
    { id: 'nas', t: 'Neural Architecture Search (NAS)', tags: 'nas,neural architecture search,weight sharing,once-for-all,supernet,darts,hardware-aware nas,rf-detr nas',
      h: T(['Approach', 'Idea', 'Status / example'], [['RL / evolutionary NAS', 'train each candidate (thousands of GPU-days)', 'historic: NASNet, AmoebaNet; produced EfficientNet / MobileNetV3 blocks'], ['Differentiable (DARTS)', 'relax choices to continuous weights, optimize jointly', 'research; unstable, rarely used in production'], ['Weight-sharing supernet / one-shot', 'train one supernet, evaluate sub-networks without retraining', L('https://github.com/mit-han-lab/once-for-all', 'Once-for-All') + ' (2020, unmaintained); <b>RF-DETR</b> uses weight-sharing NAS over resolution, patch size, decoder layers, queries (' + L('https://github.com/roboflow/rf-detr', 'roboflow/rf-detr') + ', ICLR 2026)'], ['Hardware-aware NAS', 'latency on the target device in the objective', 'MobileNetV4, FBNet, OFA, RF-DETR Pareto family'], ['Pipeline / joint NAS + HPO', 'architecture and hyperparameters together with multi-fidelity', L('https://github.com/automl/neps', 'automl/neps') + ' (NePS, active)']]),
      tip: 'For most teams, "NAS" in practice means choosing the right member of an already-searched family (YOLO26 n…x, RF-DETR N…2XL, DEIMv2 Atto…X, MobileNetV4) with a latency-constrained search over size and resolution.' }
  ]},
  { id: 'tools', title: 'Tools & Repos (status checked Oct 2026)', items: [
    { id: 'hpo', t: 'Hyperparameter optimization frameworks', tags: 'optuna,ray tune,smac3,syne tune,hyperopt,flaml,botorch,ax,vizier,nevergrad,keras tuner,repo links', wide: true,
      h: T(['Tool', 'Latest release', 'Best for', 'Repo'], [
        ['Optuna', 'v5.0.0 (Sep 2026)', 'default choice: define-by-run search spaces, TPE / GP / CMA-ES, pruners, multi-objective, dashboard, OptunaHub samplers', L('https://github.com/optuna/optuna') + ' · ' + L('https://github.com/optuna/optuna-dashboard') + ' · ' + L('https://github.com/optuna/optunahub')],
        ['Ray Tune', 'Ray 2.59 (Oct 2026)', 'distributed / multi-node search, ASHA, PBT, integrates Optuna/HyperOpt samplers', L('https://github.com/ray-project/ray')],
        ['SMAC3', 'v2.4.1 (Sep 2026)', 'Bayesian optimization with random forests / GPs, multi-fidelity (Hyperband), mixed spaces', L('https://github.com/automl/SMAC3')],
        ['Syne Tune', 'v0.16.0 (Aug 2026)', 'large-scale asynchronous HPO and multi-fidelity, cloud backends (AWS)', L('https://github.com/syne-tune/syne-tune')],
        ['FLAML', 'v2.7.0 (Sep 2026)', 'cost-frugal HPO (Microsoft), lightweight', L('https://github.com/microsoft/FLAML')],
        ['BoTorch / Ax', 'BoTorch v0.18.1 (Jun 2026)', 'research-grade Bayesian optimization (Meta)', L('https://github.com/pytorch/botorch') + ' · ' + L('https://github.com/facebook/Ax')],
        ['Hyperopt', 'v0.3.0 (Jul 2026)', 'classic TPE (original implementation)', L('https://github.com/hyperopt/hyperopt')],
        ['Nevergrad', '1.0.12 (Apr 2025)', 'gradient-free optimizers (CMA, DE, PSO)', L('https://github.com/facebookresearch/nevergrad')],
        ['Google Vizier (OSS)', 'v0.1.24 (Feb 2025)', 'Google\'s black-box optimization service, open-source client', L('https://github.com/google/vizier')],
        ['DEHB', 'v0.1.2', 'differential-evolution Hyperband', L('https://github.com/automl/DEHB')],
        ['KerasTuner / AutoKeras', 'v1.4.8 / 3.0.0 (Nov 2025)', 'Keras-native tuning and AutoML', L('https://github.com/keras-team/keras-tuner') + ' · ' + L('https://github.com/keras-team/autokeras')],
        ['Weights & Biases Sweeps', 'wandb v0.31.0 (Oct 2026)', 'hosted sweeps (random / Bayes / Hyperband) with experiment tracking', L('https://github.com/wandb/wandb')]]) },
    { id: 'cvauto', t: 'CV-specific AutoML & training tools', tags: 'autogluon multimodal,ultralytics tune,autotrain,rf-detr,super-gradients,yolo-nas,cv automl tools', wide: true,
      h: T(['Tool', 'Latest', 'What it automates for CV', 'Repo / docs'], [
        ['Ultralytics tuner', 'ultralytics v8.4.x (Oct 2026)', '<code>model.tune()</code>: genetic hyperparameter evolution for YOLO detect / segment / pose; optional Ray Tune backend (<code>use_ray=True</code>)', L('https://github.com/ultralytics/ultralytics') + ' · ' + L('https://docs.ultralytics.com/guides/hyperparameter-tuning/', 'tuning guide')],
        ['AutoGluon MultiModalPredictor', 'v1.6.3 (Sep 2026)', 'image classification, object detection, segmentation and image-text tasks from a DataFrame / COCO json, with presets, time limits, backbone selection and ensembling', L('https://github.com/autogluon/autogluon') + ' · ' + L('https://auto.gluon.ai/stable/tutorials/multimodal/index.html', 'multimodal docs')],
        ['Hugging Face AutoTrain', 'active (Oct 2026)', 'no-code / CLI fine-tuning incl. image classification and object detection on Hub models', L('https://github.com/huggingface/autotrain-advanced')],
        ['RF-DETR', '1.11.2 (Oct 2026)', 'NAS-derived detector family; weight-sharing search can be re-run for a dataset', L('https://github.com/roboflow/rf-detr')],
        ['SuperGradients (YOLO-NAS)', '3.7.1 (Apr 2024)', 'NAS-found YOLO-NAS detectors and training recipes; low activity since Deci joined NVIDIA', L('https://github.com/Deci-AI/super-gradients')]]) },
    { id: 'agents', t: 'LLM-agent AutoML (newest generation)', tags: 'llm agent automl,autogluon assistant,mlzero,aide,mle-bench,ai ml engineer agent',
      h: T(['Tool', 'Latest', 'What it does', 'Repo'], [['AutoGluon Assistant (MLZero)', 'v1.1.0 (Mar 2026)', 'multi-agent LLM system that reads your data and task description, writes, runs and debugs the training pipeline end to end (multimodal incl. images)', L('https://github.com/autogluon/autogluon-assistant')], ['AIDE', 'v0.2.2 (Nov 2025, active)', 'LLM agent that iterates on ML code as a tree search over solution drafts', L('https://github.com/WecoAI/aideml')], ['MLE-bench', 'benchmark', 'Kaggle-style benchmark for ML-engineering agents (OpenAI)', L('https://github.com/openai/mle-bench')]]),
      tip: 'Treat agent output as a strong first draft: review the data split, the metric and the evaluation code it wrote before trusting its leaderboard number.' },
    { id: 'cloud', t: 'Managed AutoML services', tags: 'vertex ai automl,azure automl images,rekognition custom labels,roboflow train,nvidia tao automl,managed automl',
      h: T(['Service', 'CV tasks'], [['Google Vertex AI AutoML (image)', 'classification, object detection'], ['Azure Machine Learning AutoML for images', 'classification (multi-class / multi-label), object detection, instance segmentation; sweeps over model architectures and hyperparameters'], ['AWS Rekognition Custom Labels', 'custom classification and detection'], ['Roboflow Train', 'hosted training of detection / segmentation models (incl. RF-DETR) with auto-labeling'], ['NVIDIA TAO Toolkit', 'AutoML (Bayesian / Hyperband) for TAO models, deployment to Jetson / TensorRT']]) +
        '<p class="note">Managed services change pricing, supported tasks and availability often — check each provider’s current documentation before committing.</p>',
      tip: 'Managed AutoML is fastest for a first model and gives a baseline number. Export or retrain open-source when you need edge deployment, cost control or custom losses.' },
    { id: 'legacy', t: 'Unmaintained / archived tools (avoid for new projects)', tags: 'nni archived,hpbandster,auto-pytorch,once-for-all,deprecated automl',
      h: T(['Tool', 'Status (GitHub)', 'Use instead'], [['Microsoft NNI', 'archived (read-only since 2024), last release v3.0 (2023)', 'Optuna / Ray Tune for HPO, NePS for NAS'], ['HpBandSter (BOHB)', 'no pushes since 2022', 'SMAC3 (Hyperband intensifier), DEHB, Optuna HyperbandPruner'], ['Auto-PyTorch', 'last release 2022, last push 2024', 'AutoGluon, NePS'], ['Once-for-All', 'last push 2023 (reference implementation)', 'RF-DETR-style weight-sharing NAS, pre-searched model families'], ['auto-sklearn', 'last release 2023 (tabular only)', 'AutoGluon / FLAML for tabular']]) }
  ]},
  { id: 'recipes', title: 'Recipes (copy-paste)', items: [
    { id: 'yolo', t: 'Recipe: tune a YOLO model (Ultralytics tuner)', tags: 'ultralytics tune,yolo hyperparameter tuning,model.tune,genetic evolution yolo',
      h: R`<p>Genetic evolution of the YOLO hyperparameters. Batch size, epochs and architecture stay fixed. The default space covers lr0, lrf, momentum, weight_decay, warmup, the box / cls / dfl gains and the augmentations (HSV, degrees, translate, scale, shear, perspective, flips, mosaic, mixup, cutmix, copy_paste, close_mosaic). Restrict it with <code>space=</code>; set <code>use_ray=True</code> for parallel Ray Tune trials. Use short trials (20–50 epochs), then retrain the best configuration fully.</p>`,
      code: R`
from ultralytics import YOLO
model = YOLO('yolo26s.pt')
model.tune(data='data.yaml', epochs=30, iterations=150, optimizer='AdamW',
           space={'lr0': (1e-5, 1e-2), 'mosaic': (0.5, 1.0), 'scale': (0.2, 0.9)},  # optional: limit the search
           plots=False, save=False, val=False)                                     # faster trials
# best hyperparameters → runs/detect/tune/best_hyperparameters.yaml (best weights: tune/weights/best.pt)
model = YOLO('yolo26s.pt')
model.train(data='data.yaml', epochs=150, cfg='runs/detect/tune/best_hyperparameters.yaml')` },
    { id: 'optuna', t: 'Recipe: Optuna HPO with pruning for any PyTorch model', tags: 'optuna example,optuna pruning,hyperband pruner,optuna pytorch,define by run',
      code: R`
import optuna

def objective(trial):
    cfg = dict(
        lr=trial.suggest_float('lr', 1e-5, 3e-3, log=True),
        wd=trial.suggest_float('wd', 1e-4, 0.3, log=True),
        drop_path=trial.suggest_float('drop_path', 0.0, 0.3),
        mixup=trial.suggest_float('mixup', 0.0, 0.8),
        imgsz=trial.suggest_categorical('imgsz', [224, 288, 384]),
    )
    model, opt, sched = build(cfg)
    for epoch in range(30):
        train_one_epoch(model, opt, sched)
        score = evaluate(model)                 # validation top-1 / mAP
        trial.report(score, epoch)
        if trial.should_prune():                # Hyperband stops weak trials early
            raise optuna.TrialPruned()
    return score

study = optuna.create_study(direction='maximize', storage='sqlite:///hpo.db', study_name='cls-v1',
                            pruner=optuna.pruners.HyperbandPruner(min_resource=3, reduction_factor=3),
                            load_if_exists=True)
study.enqueue_trial({'lr': 1e-4, 'wd': 0.05, 'drop_path': 0.1, 'mixup': 0.2, 'imgsz': 224})  # start from the known-good default
study.optimize(objective, n_trials=60)
print(study.best_params, optuna.importance.get_param_importances(study))   # PED-ANOVA by default in Optuna 5
# optuna-dashboard sqlite:///hpo.db  → live plots` },
    { id: 'multi', t: 'Recipe: accuracy vs latency (multi-objective)', tags: 'multi-objective optuna,pareto,latency aware search,model size imgsz search',
      code: R`
def objective(trial):
    size  = trial.suggest_categorical('size', ['n', 's', 'm'])
    imgsz = trial.suggest_categorical('imgsz', [480, 640, 800, 960])
    m = train_short(f'yolo26{size}.pt', imgsz=imgsz, epochs=40)
    latency_ms = benchmark_on_target(m, imgsz)     # TensorRT FP16 on the deployment GPU / Jetson
    return m.map50_95, latency_ms

study = optuna.create_study(directions=['maximize', 'minimize'])
study.optimize(objective, n_trials=24)
pareto = study.best_trials            # pick the most accurate point under your latency budget` },
    { id: 'thresh', t: 'Recipe: optimize per-class thresholds (no retraining)', tags: 'threshold optimization,per-class threshold,optimize f-beta,post-processing automl,reduce false positives automl',
      h: R`<p>The cheapest AutoML with the most direct effect on FPs and FNs: search confidence thresholds (and NMS IoU) on cached validation predictions to maximize your F-beta or recall at a precision target. Run it on a validation split and confirm on a held-out split.</p>`,
      code: R`
import numpy as np, optuna
# preds: list of (class_id, score, is_tp) on the validation set; n_gt[c] = ground-truth count per class
def fbeta_for(c, thr, beta=1.0):
    p = [(s, tp) for k, s, tp in preds if k == c and s >= thr]
    tp = sum(t for _, t in p); fp = len(p) - tp; fn = n_gt[c] - tp
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return (1 + beta**2) * P * R / max(beta**2 * P + R, 1e-9)

best = {}
for c in classes:
    st = optuna.create_study(direction='maximize')
    st.optimize(lambda t: fbeta_for(c, t.suggest_float('thr', 0.05, 0.95), beta=0.5), n_trials=60)  # beta<1: fewer FPs
    best[c] = st.best_params['thr']` },
    { id: 'autogluon', t: 'Recipe: AutoGluon for images (baseline in minutes)', tags: 'autogluon image classification,autogluon object detection,multimodalpredictor example',
      code: R`
from autogluon.multimodal import MultiModalPredictor
# Classification: DataFrame with columns 'image' (file path) and 'label'
clf = MultiModalPredictor(label='label', problem_type='classification', eval_metric='accuracy')
clf.fit(train_df, tuning_data=val_df, time_limit=3600, presets='best_quality')
print(clf.evaluate(test_df))

# Object detection: COCO-format json
det = MultiModalPredictor(problem_type='object_detection', sample_data_path='train.json', presets='medium_quality')
det.fit('train.json', tuning_data='val.json', time_limit=3 * 3600)
print(det.evaluate('test.json'))` },
    { id: 'ray', t: 'Recipe: Ray Tune + ASHA on a multi-GPU cluster', tags: 'ray tune asha example,distributed hpo,ray tune pytorch',
      code: R`
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_fn(config):
    model, opt = build(config)
    for epoch in range(50):
        train_one_epoch(model, opt)
        tune.report({'map': evaluate(model), 'epoch': epoch})   # ASHA stops weak trials

tuner = tune.Tuner(
    tune.with_resources(train_fn, {'gpu': 1, 'cpu': 8}),
    param_space={'lr': tune.loguniform(1e-5, 3e-3), 'wd': tune.loguniform(1e-4, 0.3),
                 'mosaic': tune.uniform(0.5, 1.0), 'imgsz': tune.choice([512, 640, 800])},
    tune_config=tune.TuneConfig(metric='map', mode='max', num_samples=64,
                                scheduler=ASHAScheduler(max_t=50, grace_period=5, reduction_factor=3)))
print(tuner.fit().get_best_result().config)` }
  ]},
  { id: 'best', title: 'Best Practices', items: [
    { id: 'space', t: 'Designing the search space for CV', tags: 'search space design,hyperparameter ranges,log scale,cv search space', wide: true,
      h: T(['Hyperparameter', 'Range', 'Scale'], [['Learning rate (AdamW / SGD)', '1e-5 – 3e-3 / 1e-3 – 1e-1', 'log'], ['Weight decay', '1e-5 – 0.3', 'log'], ['Warmup', '0 – 10% of steps', 'linear'], ['Batch size', 'fixed by memory; scale LR with it rather than search it', '—'], ['Drop-path / dropout', '0 – 0.4', 'linear'], ['Label smoothing', '0 – 0.2', 'linear'], ['Mixup / cutmix / mosaic / copy-paste prob.', '0 – 1', 'linear'], ['Input size', '{480, 640, 800, 960, 1280} (det); {224, 288, 384} (cls)', 'categorical'], ['Model size', '{n, s, m, l} or backbone list', 'categorical'], ['Loss gains (box / cls / dfl)', '0.5× – 2× default', 'log'], ['EMA decay', '{0.999, 0.9995, 0.9999}', 'categorical']]),
      tip: 'Search 3–6 parameters that matter (LR first), fix the rest to good defaults, and seed the study with the default recipe as trial #1.' },
    { id: 'practices', t: 'AutoML best-practice checklist', tags: 'automl best practices,hpo checklist,avoid overfitting validation,reproducible hpo', wide: true,
      h: R`<ol>
        <li><b>Fix the evaluation first:</b> group-aware train/val/test split, a metric that matches deployment (e.g. recall at precision ≥ 0.9, per-class AP), and a held-out test set the search never sees.</li>
        <li><b>Baseline before search:</b> train the default recipe; AutoML must beat it on the held-out test set, not just validation.</li>
        <li><b>Budget explicitly:</b> GPU-hours and number of trials up front (use the estimator above). Typical: 20–30 trials for TPE / GP on 3–6 params, 50–200 with ASHA.</li>
        <li><b>Use multi-fidelity</b> (fewer epochs, smaller images, data subset) to screen, and verify the low-fidelity ranking correlates with full training.</li>
        <li><b>Seed with known-good configs</b> (<code>enqueue_trial</code>) and narrow ranges around published recipes.</li>
        <li><b>Account for noise:</b> retrain the top 3–5 configurations with 2–3 seeds; differences under ~0.3–0.5 mAP are often noise.</li>
        <li><b>Avoid validation overfitting:</b> hundreds of trials on a small validation set will overfit it. Use cross-validation or a larger validation set, and report the final number on the untouched test set.</li>
        <li><b>Optimize for deployment:</b> include latency / memory measured on the target hardware as an objective or constraint.</li>
        <li><b>Log everything</b> (Optuna RDB storage, W&amp;B, MLflow): config, seed, data version, git hash. Make studies resumable.</li>
        <li><b>Learn from the study:</b> check parameter importance and slice plots, then shrink the space for the next retraining cycle.</li>
        <li><b>Search thresholds last</b>, on cached predictions; it is cheap and directly sets the FP / FN trade-off.</li></ol>` },
    { id: 'pitfalls', t: 'Common AutoML pitfalls', tags: 'automl pitfalls,data leakage hpo,optimizer overfitting,wrong metric',
      h: T(['Pitfall', 'Fix'], [['Tuning on the test set', 'separate validation for search; touch test once at the end'], ['Metric mismatch (optimize mAP50, deploy at a fixed threshold)', 'optimize the deployment metric or add threshold search'], ['Leaky splits (same video / scene in train and val)', 'group-aware splits; otherwise the search favours memorization'], ['Search space too wide (uniform LR 0–1)', 'log scales and narrow ranges around defaults'], ['Comparing trials with different epochs', 'compare at equal budget (rung) or use a pruner'], ['Ignoring variance', 'repeat the top configs with new seeds before choosing'], ['FLOPs as the latency objective', 'measure real latency on the target device and runtime']]) }
  ]}
  ]
});
})();

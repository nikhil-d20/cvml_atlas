(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'deploy', title: 'Deployment, Efficiency & MLOps', short: 'Deploy', icon: '🚀',
  blurb: 'FLOPs vs latency, quantization, pruning, distillation, ONNX/TensorRT export, edge hardware, monitoring.',
  intro: 'Getting a trained model to run fast, cheap and reliably in production.',
  sections: [
  { id: 'eff', title: 'Efficiency Metrics', items: [
    { id: 'flops', t: 'FLOPs, MACs, params, memory, latency', tags: 'flops,macs,params,latency,throughput,roofline,arithmetic intensity',
      f: [R`\text{FLOPs}\approx2\times\text{MACs}`, R`\text{Arithmetic intensity}=\frac{\text{FLOPs}}{\text{bytes moved}},\;\;\text{attainable}=\min(\text{peak FLOPs},\;\text{BW}\times\text{AI})`, R`\text{weights memory}=P\times\text{bytes/param}\;(4\text{ fp32},2\text{ fp16},1\text{ int8},0.5\text{ int4})`],
      h: R`<p>Roofline: depthwise convs, norms and elementwise ops are memory-bound — low FLOPs but not proportionally faster. Always measure latency on target hardware, batch size and precision; report p50/p95 and throughput separately.</p>`,
      code: R`
# accurate GPU timing
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(100): model(x)
torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 100 * 1e3   # after >=20 warmup iters` }
  ]},
  { id: 'quant', title: 'Quantization', items: [
    { id: 'affine', t: 'Affine (asymmetric) & symmetric quantization', tags: 'quantization,int8,scale,zero point,per-channel,ptq',
      f: [R`q=\mathrm{clamp}\!\left(\mathrm{round}\!\left(\tfrac{x}{s}\right)+z,\;q_{min},q_{max}\right),\;\;\hat x=s(q-z)`, R`s=\frac{x_{max}-x_{min}}{q_{max}-q_{min}},\;\;z=q_{min}-\mathrm{round}\!\left(\tfrac{x_{min}}{s}\right)`, R`\text{symmetric: } z=0,\;s=\frac{\max|x|}{2^{b-1}-1}`],
      h: R`<p>Weights: symmetric per-channel. Activations: per-tensor (asymmetric after ReLU). Calibration: min-max, percentile (99.99%), entropy/KL (TensorRT), MSE. Calibrate on 100–1000 representative images.</p>` },
    { id: 'qat', t: 'PTQ vs QAT (fake quant, STE)', tags: 'qat,quantization-aware training,ste,fake quantization,ptq',
      f: [R`\text{STE: }\frac{\partial\hat x}{\partial x}\approx\mathbb 1_{x\in[x_{min},x_{max}]}`],
      h: T(['Method', 'Effort', 'Typical accuracy drop (INT8)'], [['PTQ (static)', 'minutes, calibration only', '0–1% CNN; ViT/DETR can drop more (outliers in LN/softmax)'], ['PTQ + tricks (SmoothQuant, AdaRound, per-channel, mixed precision)', 'hours', '< 0.5%'], ['QAT', 'fine-tune 5–10% of original schedule', '≈ 0'], ['Weight-only INT4/INT8 (GPTQ, AWQ)', 'LLM/VLM standard', 'small; memory-bound speedup'], ['FP8 (E4M3)', 'H100/Blackwell', 'near-lossless, easy']]),
      tip: 'YOLO-style detectors: keep the detection head (and DFL/sigmoid) in FP16 if INT8 hurts mAP; sensitivity-analyse per layer and fall back selectively.' }
  ]},
  { id: 'compress', title: 'Pruning, Distillation & Re-parameterization', items: [
    { id: 'prune', t: 'Pruning', tags: 'pruning,structured pruning,unstructured,magnitude pruning,lottery ticket,2:4 sparsity',
      f: [R`\text{magnitude: remove } w\text{ with }|w|<\tau;\;\;\text{Taylor importance: } I=\big|\tfrac{\partial L}{\partial w}w\big|`],
      h: R`<p>Unstructured sparsity rarely speeds up dense hardware; <b>structured</b> (channels/heads/blocks) or NVIDIA 2:4 semi-structured sparsity does. Iterative prune → fine-tune. Tools: Torch-Pruning (DepGraph).</p>` },
    { id: 'kd', t: 'Distillation for deployment', tags: 'knowledge distillation,feature distillation,detection distillation',
      h: R`<p>Logit KD (see Training tab) + feature mimicking on FPN levels for detectors (e.g. masked/focal feature distillation). Best practice: large teacher (or VLM auto-labels) → many unlabelled in-domain images → small student.</p>` },
    { id: 'rep', t: 'Structural re-parameterization & fusion', tags: 'repvgg,reparameterization,bn folding,conv fusion',
      f: [R`W'=\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W,\;\;b'=\beta+\frac{\gamma(b-\mu)}{\sqrt{\sigma^2+\epsilon}}`, R`\text{RepVGG: } W_{3\times3}'=W_{3\times3}+\text{pad}(W_{1\times1})+\text{pad}(I)`],
      h: R`<p>Train multi-branch for accuracy, merge into a single 3×3 conv for speed (RepVGG, MobileOne, YOLOv6/v7, FastViT).</p>` }
  ]},
  { id: 'export', title: 'Export & Runtimes', items: [
    { id: 'runtimes', t: 'Inference runtimes', tags: 'onnx,tensorrt,openvino,coreml,tflite,litert,ncnn,torchscript,torch.compile,executorch',
      h: T(['Runtime', 'Target', 'Notes'], [['ONNX Runtime', 'CPU/GPU, portable', 'EPs: CUDA, TensorRT, OpenVINO, DirectML, CoreML'], ['TensorRT', 'NVIDIA GPU / Jetson', 'fastest on NVIDIA; FP16/INT8/FP8; engine is GPU-specific'], ['OpenVINO', 'Intel CPU/iGPU/NPU', 'INT8 via NNCF'], ['Core ML', 'Apple ANE/GPU', 'coremltools; ANE prefers static shapes'], ['LiteRT (TFLite) / ExecuTorch', 'Android / mobile / MCU', 'int8, NNAPI/GPU/QNN delegates'], ['NCNN / MNN / TNN', 'mobile ARM', 'lightweight C++'], ['torch.compile / TorchScript', 'PyTorch serving', 'compile: kernel fusion, CUDA graphs'], ['Triton Inference Server / vLLM', 'server', 'dynamic batching, multi-model / LLM-VLM serving']]),
      code: R`
torch.onnx.export(model.eval(), torch.randn(1, 3, 640, 640), 'model.onnx', opset_version=17,
                  input_names=['images'], output_names=['out'], dynamic_axes={'images': {0: 'b'}})
# trtexec --onnx=model.onnx --fp16 --saveEngine=model.engine` },
    { id: 'pitfalls', t: 'Export pitfalls', tags: 'export errors,onnx opset,dynamic shapes,preprocessing mismatch',
      h: R`<ul><li>Preprocessing parity (resize interpolation, letterbox padding colour 114, RGB/BGR, /255, mean/std) — verify output diff < 1e-3 vs PyTorch.</li><li>Unsupported ops (deformable attention, grid_sample variants) → custom plugins or model variants designed for export (RT-DETR, D-FINE, YOLO26).</li><li>Dynamic shapes slow TensorRT; build profiles for the few sizes you use.</li><li>NMS inside the graph (EfficientNMS plugin) or NMS-free models simplify pipelines.</li></ul>` }
  ]},
  { id: 'hw', title: 'Hardware Cheat-sheet', items: [
    { id: 'gpus', t: 'Accelerators (dense peak, approx.)', tags: 'gpu,h100,a100,t4,jetson,l4,b200,hardware,tops',
      h: T(['Device', 'FP16/BF16 TFLOPs (dense)', 'Memory', 'Typical role'], [['NVIDIA T4', '65', '16 GB', 'inference benchmark reference'], ['NVIDIA L4', '121', '24 GB', 'inference / video'], ['NVIDIA A100', '312', '40/80 GB', 'training'], ['NVIDIA H100 SXM', '~990', '80 GB', 'training (FP8 ~1979)'], ['NVIDIA B200', '~2250', '192 GB', 'training / large inference'], ['Jetson Orin Nano / AGX Orin', '~20–67 / 275 INT8 TOPS (sparse)', '8 / 32–64 GB', 'edge robotics'], ['Apple M-series ANE', '~18–38 INT8/FP16 TOPS', 'unified', 'on-device'], ['Edge TPU / Hailo-8', '4 / 26 INT8 TOPS', '—', 'low-power edge']]) + R`<p>Vendor figures vary by SKU and sparsity; use as order-of-magnitude.</p>` }
  ]},
  { id: 'mlops', title: 'MLOps & Data', items: [
    { id: 'datacentric', t: 'Data-centric loop', tags: 'mlops,active learning,data versioning,labeling,drift monitoring',
      h: R`<ol><li>Version data + labels (DVC, LakeFS, FiftyOne/Roboflow datasets) and pin to each model.</li><li>Error analysis by slices (size, lighting, class, camera) → targeted data collection.</li>
        <li>Active learning: sample by uncertainty (entropy, margin), disagreement (ensemble/TTA) or diversity (coreset on embeddings).</li><li>Monitor production: input drift (embedding distance / PSI), confidence histograms, class frequency; human review of low-confidence samples.</li>
        <li>Shadow-deploy new models, compare on the same stream, then A/B.</li></ol>`,
      f: [R`\text{PSI}=\sum_i(p_i-q_i)\ln\frac{p_i}{q_i}\;\;(>0.2\text{ = significant shift})`] },
    { id: 'repro', t: 'Reproducibility checklist', tags: 'reproducibility,seed,determinism,experiment tracking',
      h: R`<ul><li>Seed Python/NumPy/torch/dataloader workers; <code>torch.use_deterministic_algorithms(True)</code> when debugging.</li><li>Track config, git hash, data version, metrics, checkpoints (W&amp;B, MLflow, ClearML).</li><li>Save the exact preprocessing and class map with the model artifact.</li></ul>` }
  ]}
  ]
});
})();

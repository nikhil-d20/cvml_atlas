(function () {
const R = String.raw, T = CV.tbl;
CV.tab({
  id: 'gen', title: 'Generative AI & Multimodal', short: 'GenAI', icon: '✧', boards: ['gen-in256'],
  blurb: 'VAE, GAN, diffusion, flow matching, DiT/MMDiT, guidance, LoRA/ControlNet, few-step distillation, AR image gen, VLMs, video.',
  intro: 'Generative models learn \\(p(x)\\) or \\(p(x\\mid c)\\). Modern image/video generators = latent autoencoder + transformer denoiser trained with flow matching.',
  sections: [
  { id: 'classic', title: 'Foundations: VAE, GAN, Flows, AR', items: [
    { id: 'vae', t: 'VAE (ELBO & reparameterization)', tags: 'vae,elbo,variational autoencoder,reparameterization,kl',
      f: [R`\log p(x)\ge\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]-D_{KL}(q_\phi(z|x)\,\|\,p(z))`, R`z=\mu+\sigma\odot\epsilon,\;\epsilon\sim\mathcal N(0,I);\;\;D_{KL}=\tfrac12\sum(\mu^2+\sigma^2-\log\sigma^2-1)`],
      h: R`<p>β-VAE weights KL by β. Latent-diffusion VAEs (SD: 8× down, 4 ch; SD3/FLUX: 16 ch) use tiny KL + LPIPS + GAN loss for sharp reconstructions.</p>` },
    { id: 'gan', t: 'GAN objectives', tags: 'gan,minimax,wgan,hinge gan,stylegan,r1 penalty',
      f: [R`\min_G\max_D\;\mathbb E_x[\log D(x)]+\mathbb E_z[\log(1-D(G(z)))]`, R`\text{Non-saturating: } L_G=-\mathbb E_z\log D(G(z))`, R`\text{WGAN-GP: } L_D=\mathbb E[D(\tilde x)]-\mathbb E[D(x)]+\lambda\,\mathbb E\big[(\|\nabla_{\hat x}D(\hat x)\|_2-1)^2\big]`, R`R_1=\tfrac{\gamma}{2}\,\mathbb E_{x}\|\nabla_x D(x)\|^2`],
      h: R`<p>Optimum = JS divergence minimization. StyleGAN: mapping network → style modulation (AdaIN / weight demodulation), R1, path-length reg. GANs are now mostly used as losses (VAE decoders, few-step distillation) rather than standalone generators.</p>` },
    { id: 'flows', t: 'Normalizing flows', tags: 'normalizing flow,realnvp,glow,change of variables',
      f: [R`\log p_X(x)=\log p_Z(f(x))+\log\left|\det\frac{\partial f}{\partial x}\right|`],
      h: R`<p>Invertible layers with cheap Jacobians (affine coupling, 1×1 invertible conv). Exact likelihood; lower sample quality. Revived as TarFlow/STARFlow (transformer autoregressive flows).</p>` },
    { id: 'vq', t: 'VQ-VAE / VQGAN tokenizers', tags: 'vq-vae,vqgan,codebook,tokenizer,fsq,commitment loss',
      f: [R`z_q=e_k,\;k=\arg\min_j\|z_e-e_j\|_2`, R`L=\|x-\hat x\|^2+\|\mathrm{sg}[z_e]-e\|^2+\beta\|z_e-\mathrm{sg}[e]\|^2,\;\beta=0.25`],
      h: R`<p>Straight-through estimator copies gradients past quantization. FSQ (finite scalar quantization) and LFQ avoid codebook collapse. Tokenizers power autoregressive image models and unified multimodal LLMs.</p>` }
  ]},
  { id: 'diff', title: 'Diffusion Models', items: [
    { id: 'ddpm', t: 'DDPM forward & reverse', tags: 'ddpm,diffusion,forward process,noise schedule,denoising',
      f: [R`q(x_t|x_0)=\mathcal N\!\big(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big),\;\;\bar\alpha_t=\prod_{s\le t}(1-\beta_s)`, R`x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon`, R`L_{simple}=\mathbb E_{t,x_0,\epsilon}\big\|\epsilon-\epsilon_\theta(x_t,t)\big\|^2`, R`x_{t-1}=\frac1{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Big)+\sigma_tz`],
      h: R`<p>T=1000, linear β 1e-4→0.02 or cosine schedule. SNR\((t)=\bar\alpha_t/(1-\bar\alpha_t)\). Use zero-terminal-SNR schedules to allow generating very dark/bright images.</p>` },
    { id: 'param', t: 'Prediction targets: ε, x₀, v', tags: 'v-prediction,epsilon prediction,x0 prediction,parameterization,min-snr',
      f: [R`v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0`, R`\text{Min-SNR-}\gamma\text{ weighting: } w_t=\frac{\min(\text{SNR}_t,\gamma)}{\text{SNR}_t}\;(\epsilon\text{-pred}),\;\gamma=5`],
      h: R`<p>v-prediction is stable at high noise and needed for zero-terminal-SNR. Min-SNR weighting speeds convergence ~3×.</p>` },
    { id: 'sde', t: 'Score-based SDE view', tags: 'score matching,sde,probability flow ode,score function',
      f: [R`dx=f(x,t)dt+g(t)dw`, R`dx=\big[f(x,t)-g(t)^2\nabla_x\log p_t(x)\big]dt+g(t)d\bar w`, R`\text{PF-ODE: } dx=\big[f-\tfrac12g^2\nabla_x\log p_t(x)\big]dt,\;\;\nabla_x\log p_t\approx-\frac{\epsilon_\theta}{\sigma_t}`] },
    { id: 'ddim', t: 'Samplers: DDIM, DPM-Solver++, Euler, EDM', tags: 'ddim,dpm-solver,euler sampler,edm,sampling steps,heun',
      f: [R`x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta+\sigma_t z,\;\;\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}`, R`\text{EDM: } \sigma_i=\left(\sigma_{max}^{1/\rho}+\tfrac{i}{N-1}(\sigma_{min}^{1/\rho}-\sigma_{max}^{1/\rho})\right)^{\rho},\;\rho=7`],
      h: T(['Sampler', 'Steps', 'Notes'], [['DDPM ancestral', '250–1000', 'stochastic, slow'], ['DDIM (η=0)', '25–50', 'deterministic, invertible (editing)'], ['DPM-Solver++ 2M (Karras σ)', '15–30', 'default for SD-style models'], ['Euler (flow models)', '20–50', 'FLUX/SD3; with shift σ schedule'], ['Heun (EDM)', '18–35 NFE', 'EDM 2nd order'], ['Distilled (LCM, DMD2, ADD)', '1–8', 'see few-step section']]) },
    { id: 'cfg', t: 'Classifier-free guidance (CFG)', tags: 'cfg,classifier-free guidance,guidance scale,negative prompt',
      f: [R`\tilde\epsilon=\epsilon_\theta(x_t,\varnothing)+w\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)`, R`\nabla\log p_w(x|c)=\nabla\log p(x)+w\,\nabla\log p(c|x)`],
      h: R`<p>Train with 10% condition dropout. w=5–8 (SD1.5), 3.5–7 (SDXL/SD3), FLUX-dev uses guidance-distilled embedding (≈3.5). High w → saturation; fixes: CFG-rescale, interval guidance (apply only mid-noise), APG. Negative prompts replace \(\varnothing\).</p>` },
    { id: 'ldm', t: 'Latent Diffusion (Stable Diffusion U-Net)', tags: 'latent diffusion,stable diffusion,unet,cross-attention,sdxl',
      h: T(['Block', 'Details'], [['VAE', '512² image → 64×64×4 latent (f=8), scaled by 0.18215'], ['Text encoder', 'CLIP ViT-L (SD1.x), OpenCLIP-G + CLIP-L (SDXL); T5-XXL added in SD3/FLUX'], ['U-Net down', 'ResBlocks (GN, SiLU, timestep emb added) + Transformer blocks (self-attn → cross-attn to text → FFN) at 64/32/16'], ['Mid', 'ResBlock + Transformer + ResBlock at 8×8'], ['U-Net up', 'mirror with skip concat'], ['Conditioning', 'timestep: sinusoidal → MLP; SDXL adds size/crop micro-conditioning']]) + R`<p>SD1.5 ≈ 860M U-Net; SDXL ≈ 2.6B U-Net + refiner.</p>` },
    { id: 'dit', t: 'DiT / MMDiT (transformer denoisers)', tags: 'dit,mmdit,adaln-zero,diffusion transformer,sd3,flux',
      f: [R`\text{adaLN-Zero: } h=x+\alpha_1\odot\mathrm{Attn}\big(\gamma_1\odot\mathrm{LN}(x)+\beta_1\big),\;(\alpha,\beta,\gamma)=\mathrm{MLP}(t+c),\;\alpha\text{ init }0`],
      h: T(['Model', 'Block design'], [['DiT (2022-12)', 'patchify latent (p=2), adaLN-Zero class+time conditioning; scales cleanly with GFLOPs'], ['PixArt-α/Σ', 'DiT + cross-attention to T5, adaLN-single'], ['MMDiT (SD3, 2024-03)', 'separate weights for text and image tokens, joint attention over concatenated sequence'], ['FLUX.1 / FLUX.2', 'double-stream MMDiT blocks then single-stream blocks, RoPE, rectified flow; FLUX.2 (2025-11) adds multi-reference editing and a VLM text encoder'], ['Video DiTs (Wan, HunyuanVideo, LTX)', '3-D (spatio-temporal) patches from a video VAE, full 3-D attention with RoPE']]) },
    { id: 'flow', t: 'Flow matching & rectified flow', tags: 'flow matching,rectified flow,velocity,logit-normal,optimal transport',
      f: [R`x_t=(1-t)\,x_0+t\,\epsilon,\;\;t\in[0,1]`, R`\mathcal L_{CFM}=\mathbb E_{t,x_0,\epsilon}\big\|v_\theta(x_t,t)-(\epsilon-x_0)\big\|^2`, R`\text{sample: } \tfrac{dx}{dt}=v_\theta(x,t),\;\text{Euler from }t{=}1\to0`, R`\text{SD3 timestep shift: } t'=\frac{s\,t}{1+(s-1)t},\;s\approx3`],
      h: R`<p>Straight paths → fewer sampling steps; logit-normal \(t\) sampling focuses on mid noise. Reflow re-trains on its own (noise, sample) pairs to straighten paths further. Standard objective for SD3, FLUX, most video models.</p>` },
    { id: 'repa', t: 'Representation alignment & modern latents', tags: 'repa,rae,va-vae,representation alignment,latent space',
      f: [R`\mathcal L=\mathcal L_{FM}+\lambda\,\mathbb E\Big[-\tfrac1N\sum_n\text{sim}\big(y_n^*,h_\phi(h_t^{[n]})\big)\Big],\;\;y^*=\text{DINOv2 features}`],
      h: R`<p>REPA aligns intermediate DiT tokens with frozen DINOv2 features → >17× faster convergence. VA-VAE/LightningDiT align the VAE latent with foundation features; RAE (2025) uses a frozen representation encoder (DINOv2) as the autoencoder itself, reaching ~1.1–1.5 FID on ImageNet-256.</p>` }
  ]},
  { id: 'control', title: 'Conditioning, Control & Personalization', items: [
    { id: 'lora', t: 'LoRA / DoRA', tags: 'lora,low-rank adaptation,dora,peft,fine-tuning',
      f: [R`W'=W_0+\frac{\alpha}{r}BA,\;\;B\in\mathbb R^{d\times r},A\in\mathbb R^{r\times k},\;B\text{ init }0`, R`\text{DoRA: } W'=m\frac{W_0+BA}{\|W_0+BA\|_c}`],
      h: R`<p>Train only A,B (≪1% params) on attention/projection layers; merge at inference (zero overhead). QLoRA: 4-bit NF4 base + LoRA. Works for LLMs, VLMs, diffusion and ViTs.</p>` },
    { id: 'controlnet', t: 'ControlNet / T2I-Adapter / IP-Adapter', tags: 'controlnet,ip-adapter,t2i-adapter,zero convolution,spatial control',
      f: [R`y_c=\mathcal F(x;\Theta)+\mathcal Z\big(\mathcal F(x+\mathcal Z(c;\Theta_{z1});\Theta_c);\Theta_{z2}\big)`, R`\text{IP-Adapter: } Z=\mathrm{Attn}(Q,K_t,V_t)+\lambda\,\mathrm{Attn}(Q,K_i,V_i)`],
      h: R`<p>ControlNet copies the encoder, feeds a condition (canny, depth, pose, seg) and adds its outputs through zero-initialized 1×1 convs. IP-Adapter adds decoupled cross-attention for image prompts (style/identity).</p>` },
    { id: 'dreambooth', t: 'DreamBooth / Textual Inversion', tags: 'dreambooth,textual inversion,personalization,prior preservation',
      f: [R`\mathcal L=\mathbb E\|\hat x_\theta(\alpha_tx+\sigma_t\epsilon,c)-x\|^2+\lambda\,\mathbb E\|\hat x_\theta(\alpha_{t'}x_{pr}+\sigma_{t'}\epsilon',c_{pr})-x_{pr}\|^2`],
      h: R`<p>DreamBooth fine-tunes the model on 3–5 images with a rare token + prior-preservation images. Textual inversion learns only a new token embedding. Modern editing models (FLUX Kontext, Qwen-Image-Edit, GPT-Image, Nano Banana/Gemini image) do zero-shot identity/subject consistency.</p>` },
    { id: 'edit', t: 'Image editing & inversion', tags: 'image editing,inpainting,sdedit,prompt-to-prompt,inversion',
      h: R`<ul><li><b>SDEdit</b>: noise input to level \(t_0\) then denoise with new prompt (strength = \(t_0\)).</li><li><b>Inpainting</b>: extra mask + masked-image channels in the U-Net input.</li><li><b>Prompt-to-Prompt</b>: reuse cross-attention maps to edit words while preserving layout.</li><li><b>DDIM / flow inversion</b>: map a real image to noise, then regenerate with edits.</li><li><b>Instruction editing</b>: InstructPix2Pix → in-context editing DiTs.</li></ul>` }
  ]},
  { id: 'fast', title: 'Few-step Generation (distillation)', items: [
    { id: 'consistency', t: 'Consistency models / LCM', tags: 'consistency model,lcm,latent consistency,one-step',
      f: [R`f_\theta(x_t,t)=f_\theta(x_{t'},t')\;\;\forall t,t',\;\;f_\theta(x,\epsilon)=x`, R`\mathcal L=d\big(f_\theta(x_{t_{n+1}},t_{n+1}),\,f_{\theta^-}(\hat x_{t_n},t_n)\big)`],
      h: R`<p>LCM / LCM-LoRA: 2–8 steps. sCM (continuous-time) scales consistency training to large models.</p>` },
    { id: 'dmd', t: 'Adversarial & distribution-matching distillation', tags: 'add,dmd,dmd2,sdxl turbo,adversarial diffusion distillation,rectified flow distillation',
      f: [R`\nabla_\theta D_{KL}(p_{fake}\|p_{real})\approx\mathbb E\big[(s_{fake}(x_t)-s_{real}(x_t))\tfrac{\partial G_\theta}{\partial\theta}\big]`],
      h: R`<p>ADD (SDXL-Turbo): GAN loss with DINOv2 discriminator + score distillation → 1–4 steps. DMD2: distribution matching + GAN, no regression loss. Hyper-SD, SDXL-Lightning, FLUX-schnell (timestep-distilled) are production 1–4-step models.</p>` }
  ]},
  { id: 'ar', title: 'Autoregressive & Unified Models', items: [
    { id: 'arimg', t: 'Autoregressive image generation', tags: 'autoregressive,var,mar,llamagen,next-token,next-scale',
      f: [R`p(x)=\prod_{i=1}^{N}p(x_i\mid x_{<i})\;\;(\text{raster tokens})`, R`\text{VAR: } p(r_1,\dots,r_K)=\prod_k p(r_k\mid r_{<k})\;\;(\text{next-scale})`],
      h: T(['Model', 'Idea'], [['LlamaGen', 'plain Llama on VQ tokens + CFG'], ['VAR (2024, NeurIPS best paper)', 'coarse-to-fine next-scale prediction; fast'], ['MAR', 'continuous tokens with per-token diffusion head, masked AR'], ['Unified (Chameleon, Janus, BAGEL, Show-o, Transfusion)', 'one transformer for understanding + generation (AR text + diffusion/AR image)'], ['Native image output LLMs', 'GPT-Image, Gemini image models: generation/editing inside a multimodal LLM']]) }
  ]},
  { id: 'vlm', title: 'Vision-Language Models', items: [
    { id: 'clip', t: 'CLIP / SigLIP objectives', tags: 'clip,siglip,contrastive,image-text,zero-shot',
      f: [R`\mathcal L_{CLIP}=-\frac1{2N}\sum_i\Big[\log\frac{e^{s_{ii}/\tau}}{\sum_je^{s_{ij}/\tau}}+\log\frac{e^{s_{ii}/\tau}}{\sum_je^{s_{ji}/\tau}}\Big]`, R`\mathcal L_{SigLIP}=-\frac1N\sum_{i,j}\log\sigma\big(z_{ij}(t\,s_{ij}+b)\big),\;z_{ij}=\pm1`] },
    { id: 'vlmarch', t: 'VLM architecture (LLaVA-style)', tags: 'vlm,llava,multimodal llm,projector,q-former,visual instruction tuning',
      h: T(['Component', 'Options'], [['Vision encoder', 'CLIP ViT-L/14-336, SigLIP/SigLIP2-so400m, InternViT-6B, native-resolution ViT (Qwen-VL: 2-D RoPE, 14×14 patches, merge 2×2)'], ['Projector', '2-layer MLP (LLaVA-1.5), Q-Former 32 queries (BLIP-2), perceiver resampler (Flamingo), pixel-shuffle token merge'], ['LLM', 'Qwen, Llama, InternLM, Gemma, etc.'], ['High-res', 'AnyRes tiling (LLaVA-NeXT), dynamic resolution (Qwen2-VL+), token compression'], ['Training', '(1) projector alignment on captions → (2) full instruction tuning → (3) RL / preference tuning (GRPO/DPO) for reasoning & grounding']]) },
    { id: 'vlmland', t: 'VLM landscape (2025–26)', tags: 'gpt,gemini,claude,qwen-vl,internvl,kimi,frontier models,vlm leaderboard',
      h: R`<p><b>Closed frontier</b>: GPT (OpenAI), Gemini (Google), Claude (Anthropic), Grok (xAI) — all natively multimodal. <b>Open</b>: Qwen2.5-VL / Qwen3-VL, InternVL3 / 3.5, Kimi-VL / Kimi K2.5, Gemma 3, Llama 4, Molmo, DeepSeek-VL2. Track MMMU, MathVista, DocVQA, ChartQA, OCRBench, Video-MME and LMArena vision. Frontier releases move monthly — see the News tab.</p>`,
      tip: 'For CV pipelines, VLMs are best as auto-labelers, verifiers and zero-shot baselines; distill into small task models for latency-critical deployment.' },
    { id: 'llm', t: 'LLM essentials for CV engineers', tags: 'llm,decoding,kv cache,rlhf,dpo,grpo,sampling temperature,top-p',
      f: [R`\text{DPO: }\mathcal L=-\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)`, R`\text{GRPO advantage: } \hat A_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)}\;\text{within a group of samples}`, R`\text{KV cache} = 2\cdot L\cdot n_{kv}\cdot d_{head}\cdot T\cdot\text{bytes}`],
      h: R`<p>Decoding: greedy / temperature / top-p (0.9) / top-k. Mixture-of-Experts routes each token to k of E experts (sparse compute). Speculative decoding: draft model proposes, target verifies.</p>` }
  ]},
  { id: 'video', title: 'Video, 3D & World Models', items: [
    { id: 'videogen', t: 'Video generation landscape', tags: 'video generation,sora,veo,wan,hunyuanvideo,ltx,kling,world model',
      h: T(['Model', 'Type', 'Notes'], [['Sora / Sora 2 (OpenAI)', 'closed', 'DiT on spacetime patches; Sora 2 (2025-09) with synchronized audio'], ['Veo 3 / 3.x (Google)', 'closed', 'native audio, strong physics & prompt adherence'], ['Kling (Kuaishou), Seedance (ByteDance), Hailuo (MiniMax), Runway Gen-4', 'closed', 'commercial leaders'], ['Wan 2.1 / 2.2 (Alibaba)', 'open', 'MoE high/low-noise experts (2.2), 5B–14B'], ['HunyuanVideo (Tencent)', 'open', '13B, dual-to-single-stream DiT'], ['LTX-Video / LTX-2 (Lightricks)', 'open', 'real-time-capable, LTX-2 adds audio & 4K'], ['World models (Genie 3, Cosmos)', 'mixed', 'interactive / physical-AI simulation']]) },
    { id: 'videoarch', t: 'Video diffusion components', tags: 'video vae,causal vae,temporal attention,3d rope',
      f: [R`\text{Causal 3D VAE: } T\times H\times W\to(1+\tfrac{T-1}{4})\times\tfrac H8\times\tfrac W8\times C`],
      h: R`<p>Causal temporal compression (first frame encoded alone → image+video joint training). Full 3-D attention with 3-D RoPE; long videos via sliding windows, frame-packing or autoregressive diffusion (diffusion forcing / self-forcing).</p>` }
  ]},
  { id: 'eval', title: 'Evaluation Metrics', items: [
    { id: 'fid', t: 'FID, IS, CLIP score, precision/recall', tags: 'fid,inception score,clip score,kid,fvd,generative metrics',
      f: [R`\text{FID}=\|\mu_r-\mu_g\|_2^2+\mathrm{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)`, R`\text{IS}=\exp\big(\mathbb E_x D_{KL}(p(y|x)\|p(y))\big)`, R`\text{CLIPScore}=w\cdot\max(\cos(E_I,E_T),0),\;w=2.5`],
      h: R`<p>FID on 50k samples (Inception-v3 pool features); sensitive to resize/compression implementation — compare only within the same codebase. FD-DINOv2 correlates better with humans. FVD for video. Text-to-image: GenEval, T2I-CompBench, DPG-Bench, human-preference arenas (Artificial Analysis, LMArena).</p>` }
  ]}
  ]
});
})();

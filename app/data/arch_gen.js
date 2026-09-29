// Generative architecture pages (ImageNet-256 leaderboard models).
(function () {
const R = String.raw, T = CV.tbl;
const P = (k, t, d, b, o) => ({ k, t, d, b, o });
const LAT = P('input', 'Noise + class label', 'z_T ~ N(0, I) in latent space (32×32×4)');
const VAE = P('head', 'VAE decoder', 'latent ÷8 → 256×256 pixels');

CV.arch({
  id: 'dit', name: 'DiT (Diffusion Transformer)', task: 'gen', match: /^DiT/, date: '2022-12', org: 'UC Berkeley / NYU (Peebles, Xie)',
  paper: 'https://arxiv.org/abs/2212.09748', code: 'https://github.com/facebookresearch/DiT', tags: 'dit,diffusion transformer,adaln-zero,latent diffusion',
  tagline: 'Replaces the U-Net with a plain ViT on latent patches, conditioned via adaLN-Zero. Quality scales cleanly with transformer GFLOPs; the template for SD3, FLUX and Sora-style models.',
  anatomy: { backbone: 'Denoiser: patchify latent (p = 2 → 256 tokens for 32×32×4) + sin-cos PE → N DiT blocks (XL: 28 blocks, 1152 wide, 16 heads).', neck: 'Conditioning: timestep embedding + class embedding → MLP → adaLN scale/shift/gate for every block.', head: 'Final adaLN + linear → per-patch noise (and variance) → unpatchify. Frozen SD VAE decodes latents.', loss: 'ε-prediction MSE (+ variational bound for Σ), 1000-step DDPM, CFG scale 1.5 for FID 2.27.' },
  pipeline: [LAT, P('other', 'Conditioning', '', ['Timestep t → sinusoidal → MLP', 'Class y → embedding', 'c = t_emb + y_emb']), P('backbone', 'DiT-XL/2', '28 blocks', ['Patchify 2×2 → 256 tokens × 1152', 'adaLN-Zero transformer blocks', 'Final layer → noise prediction']), VAE, P('output', 'Image', '256×256')],
  blocks: [{ t: 'DiT block (adaLN-Zero)', k: 'backbone', ops: ['LN → scale γ1 / shift β1 from c', 'Multi-head self-attention → × gate α1', 'LN → scale γ2 / shift β2', 'Pointwise FFN → × gate α2'], skips: [[0, 1, '+'], [2, 3, '+']], note: 'α initialized to 0 so each block starts as identity.' }],
  f: [R`h\leftarrow h+\alpha_1\odot\mathrm{Attn}\big(\gamma_1\odot\mathrm{LN}(h)+\beta_1\big),\;\;(\gamma,\beta,\alpha)=\mathrm{MLP}(t_{emb}+y_{emb})`]
});

CV.arch({ id: 'sit', name: 'SiT (Scalable Interpolant Transformer)', task: 'gen', match: /^SiT/, date: '2024-01', org: 'NYU', paper: 'https://arxiv.org/abs/2401.08740', tags: 'sit,stochastic interpolants,flow matching,dit',
  tagline: 'Same DiT architecture trained with the stochastic-interpolant / flow-matching framework (velocity prediction, linear path, SDE sampler) → better FID at equal compute.',
  anatomy: { backbone: 'DiT-XL/2 architecture.', neck: 'adaLN conditioning.', head: 'Velocity prediction.', loss: 'Flow matching MSE on v = ε − x₀ (linear interpolant).' },
  pipeline: [LAT, P('backbone', 'DiT-XL/2 (velocity)'), VAE, P('output', 'Image')], f: [R`x_t=(1-t)x_0+t\,\epsilon,\;\;\mathcal L=\|v_\theta(x_t,t)-(\epsilon-x_0)\|^2`] });

CV.arch({ id: 'repa', name: 'REPA (representation alignment)', task: 'gen', match: /^REPA/, date: '2024-10', org: 'KAIST / NYU', paper: 'https://arxiv.org/abs/2410.06940', tags: 'repa,representation alignment,dinov2,faster diffusion training',
  tagline: 'Regularizes an intermediate DiT/SiT layer to match frozen DINOv2 features of the clean image: >17× faster convergence and FID 1.42 (SiT-XL).',
  anatomy: { backbone: 'SiT-XL/2 denoiser.', neck: 'Small MLP projector on block ~8 tokens.', head: 'Velocity output.', loss: 'Flow matching + λ · (−cosine similarity to DINOv2 patch features).' },
  pipeline: [LAT, P('backbone', 'SiT-XL/2'), P('other', 'Alignment branch', 'training only', ['Block 8 tokens → MLP', 'Match DINOv2(x₀) features']), VAE, P('output', 'Image')] });

CV.arch({ id: 'lightningdit', name: 'LightningDiT + VA-VAE', task: 'gen', match: /^LightningDiT/, date: '2025-01', org: 'HUST', paper: 'https://arxiv.org/abs/2501.01423', tags: 'lightningdit,va-vae,vision foundation aligned vae',
  tagline: 'Aligns the VAE latent space with vision foundation features (VA-VAE), making high-dimensional latents easy to diffuse; an optimized DiT reaches FID 1.35.',
  anatomy: { backbone: 'LightningDiT (DiT with RMSNorm, SwiGLU, RoPE and training tricks).', neck: 'Class + time conditioning.', head: 'VA-VAE decoder (f16, high-dim latent).', loss: 'Flow matching; VAE trained with foundation-model alignment loss.' },
  pipeline: [LAT, P('backbone', 'LightningDiT'), P('head', 'VA-VAE decoder'), P('output', 'Image')] });

CV.arch({ id: 'rae', name: 'RAE (Representation Autoencoders)', task: 'gen', match: /^RAE/, date: '2025-10', org: 'NYU (Xie lab)', tags: 'rae,representation autoencoder,dinov2 latent,dit-dh',
  tagline: 'Uses a frozen representation encoder (e.g. DINOv2) as the "VAE" encoder with a trained decoder; diffusing in this semantic latent space with a wide DiT head (DiT-DH) gives state-of-the-art ImageNet FID.',
  anatomy: { backbone: 'DiT-DH (DiT with a wide, shallow denoising head) operating on encoder tokens.', neck: 'Class + time conditioning.', head: 'Trained pixel decoder from representation tokens.', loss: 'Flow matching in representation space.' },
  pipeline: [P('input', 'Noise in DINOv2 token space'), P('backbone', 'DiT-DH'), P('head', 'RAE decoder'), P('output', 'Image')] });

CV.arch({ id: 'mar', name: 'MAR (masked autoregressive, no VQ)', task: 'gen', match: /^MAR/, date: '2024-06', org: 'MIT / Google DeepMind', paper: 'https://arxiv.org/abs/2406.11838', tags: 'mar,diffusion loss,continuous tokens,autoregressive',
  tagline: 'Autoregressive generation on continuous tokens: a transformer predicts a condition vector per masked token and a small MLP diffusion head samples the token ("diffusion loss"), removing vector quantization.',
  anatomy: { backbone: 'Bidirectional masked-AR transformer (MAE-style encoder-decoder).', neck: 'Per-token conditioning z.', head: 'Small MLP denoiser p(x_i | z_i) trained with a diffusion loss.', loss: 'Per-token diffusion MSE.' },
  pipeline: [P('input', 'Known tokens + class'), P('backbone', 'Masked AR transformer'), P('head', 'MLP diffusion head per token'), P('output', 'Image tokens → KL-VAE decoder')] });

CV.arch({ id: 'var', name: 'VAR (visual autoregressive, next-scale)', task: 'gen', match: /^VAR/, date: '2024-04', org: 'ByteDance / PKU', paper: 'https://arxiv.org/abs/2404.02905', code: 'https://github.com/FoundationVision/VAR', tags: 'var,next-scale prediction,multi-scale vqvae',
  tagline: 'Generates coarse-to-fine token maps (1×1 → 16×16), predicting a whole scale per step: GPT-style scaling laws for images and only ~10 steps. NeurIPS 2024 best paper.',
  anatomy: { backbone: 'GPT-2-style decoder transformer with block-causal attention across scales (d30 ≈ 2B).', neck: 'Multi-scale residual VQ-VAE tokenizer.', head: 'Softmax over the codebook per token.', loss: 'Next-scale cross-entropy.' },
  pipeline: [P('input', 'Class token'), P('backbone', 'Transformer', 'predict r_k given r_1…r_{k-1}'), P('head', 'Codebook softmax'), P('output', 'Multi-scale tokens → VQ-VAE decoder')] });

CV.arch({ id: 'llamagen', name: 'LlamaGen', task: 'gen', match: /^LlamaGen/, date: '2024-06', org: 'HKU / ByteDance', paper: 'https://arxiv.org/abs/2406.06525', tags: 'llamagen,autoregressive,llama,vq tokens',
  tagline: 'Plain Llama next-token prediction on VQGAN image tokens (raster order) with CFG — shows vanilla LLM architectures can match diffusion.',
  anatomy: { backbone: 'Llama decoder (RMSNorm, SwiGLU, RoPE), 111M–3.1B.', neck: 'VQGAN tokenizer (16× down, 16k codebook).', head: 'Token softmax.', loss: 'Next-token CE.' },
  pipeline: [P('input', 'Class token'), P('backbone', 'Llama transformer'), P('head', 'Next-token softmax', '576 tokens (24×24)'), P('output', 'Tokens → VQ decoder')] });

CV.arch({ id: 'mdt', name: 'MDT / MDTv2 (masked diffusion transformer)', task: 'gen', match: /^MDT/, date: '2023-03', org: 'Sea AI Lab / Nankai', paper: 'https://arxiv.org/abs/2303.14389', tags: 'mdt,masked diffusion transformer,side interpolater',
  tagline: 'Adds a mask-latent modeling objective to DiT training (predict masked latent tokens) to learn contextual relations, converging ~10× faster.',
  anatomy: { backbone: 'DiT-style encoder-decoder with side-interpolater.', neck: 'adaLN conditioning.', head: 'Noise prediction.', loss: 'Diffusion loss + masked latent reconstruction.' },
  pipeline: [LAT, P('backbone', 'Masked DiT'), VAE, P('output', 'Image')] });

CV.arch({ id: 'stylegan', name: 'StyleGAN-XL', task: 'gen', match: /^StyleGAN/, date: '2022-02', org: 'Univ. of Tübingen', paper: 'https://arxiv.org/abs/2202.00273', tags: 'stylegan,stylegan-xl,projected gan,gan',
  tagline: 'Scales StyleGAN3 to ImageNet with Projected GAN discriminators (pre-trained feature networks), progressive growing and classifier guidance. One-step sampling.',
  anatomy: { backbone: 'Generator: mapping network (z, class → w) + synthesis network with modulated convs.', neck: 'Style modulation per layer.', head: 'Projected discriminators on EfficientNet / DeiT features.', loss: 'Non-saturating GAN loss + classifier guidance.' },
  pipeline: [P('input', 'z + class'), P('other', 'Mapping network', 'z → w'), P('backbone', 'Synthesis network', 'modulated convs'), P('output', 'Image', '1 forward pass')] });

CV.arch({ id: 'ldm', name: 'Latent Diffusion (LDM)', task: 'gen', match: /^LDM/, date: '2021-12', org: 'LMU Munich / Runway (Rombach et al.)', paper: 'https://arxiv.org/abs/2112.10752', code: 'https://github.com/CompVis/latent-diffusion', tags: 'ldm,latent diffusion,stable diffusion,unet,cross-attention',
  tagline: 'Diffusion in the latent space of a pre-trained autoencoder (f = 4/8) with a U-Net denoiser and cross-attention conditioning — the basis of Stable Diffusion.',
  anatomy: { backbone: 'U-Net denoiser: ResBlocks + attention at several resolutions, skip connections.', neck: 'Conditioning via cross-attention (text) or class embedding.', head: 'Autoencoder decoder (KL or VQ regularized).', loss: 'ε-prediction MSE.' },
  pipeline: [LAT, P('backbone', 'U-Net', '', ['Down: ResBlock + attn', 'Mid: ResBlock + attn', 'Up: ResBlock + attn + skips']), VAE, P('output', 'Image')] });

CV.arch({ id: 'adm', name: 'ADM (Guided Diffusion)', task: 'gen', match: /^ADM/, date: '2021-05', org: 'OpenAI', paper: 'https://arxiv.org/abs/2105.05233', code: 'https://github.com/openai/guided-diffusion', tags: 'adm,guided diffusion,classifier guidance,pixel diffusion',
  tagline: '"Diffusion models beat GANs": an improved pixel-space U-Net (more heads, BigGAN res-blocks, adaptive GroupNorm) plus classifier guidance.',
  anatomy: { backbone: 'Pixel U-Net with attention at 32/16/8, AdaGN conditioning.', neck: 'Timestep + class via AdaGN.', head: 'Predict ε and Σ.', loss: 'Hybrid (simple + VLB); classifier on noisy images for guidance.' },
  pipeline: [P('input', 'Pixel noise', '256×256×3'), P('backbone', 'U-Net (pixel space)'), P('other', 'Noisy classifier', 'guidance gradient'), P('output', 'Image')] });

CV.arch({ id: 'biggan', name: 'BigGAN', task: 'gen', match: /^BigGAN/, date: '2018-09', org: 'DeepMind', paper: 'https://arxiv.org/abs/1809.11096', tags: 'biggan,class-conditional gan,truncation trick',
  tagline: 'Large-batch class-conditional GAN with spectral norm, shared class embeddings and the truncation trick; the pre-diffusion state of the art.',
  anatomy: { backbone: 'ResNet generator with class-conditional BatchNorm, self-attention at 64×64.', neck: 'Hierarchical latent (z split across layers).', head: 'Projection discriminator.', loss: 'Hinge GAN loss, orthogonal regularization.' },
  pipeline: [P('input', 'z + class'), P('backbone', 'ResNet generator'), P('output', 'Image')] });
})();

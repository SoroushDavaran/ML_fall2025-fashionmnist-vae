# ML_fall2025-fashionmnist-vae


## Phase 1: Data Preparation & Exploratory Data Analysis



This phase establishes a reliable data pipeline and verifies that the Fashion-MNIST dataset is correctly loaded, transformed, and structured for downstream generative modeling. We also perform a lightweight exploratory analysis to understand the dataset’s basic statistical properties and to avoid avoidable mistakes before training.

### Environment and reproducibility

All experiments are implemented in PyTorch and executed on GPU when available. In our run, CUDA is enabled and the device is set accordingly:

```text
torch: 2.10.0+cu128
cuda available: True
device: cuda
````

To keep experimental results consistent across runs, we fix a global random seed (`SEED = 42`) for Python’s `random`, NumPy, and PyTorch (CPU/GPU). In addition, we configure CuDNN to behave deterministically (as much as possible in practice) by disabling benchmarking and enabling deterministic kernels. This ensures that dataset splitting, initialization, and stochastic components behave consistently enough to support fair comparisons across model variants.

### Dataset overview and preprocessing pipeline

We use the Fashion-MNIST dataset, which contains grayscale 28×28 images across 10 classes:

```text
Class names: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

The dataset sizes observed in our setup are:

```text
Full train size: 60000
Test size      : 10000
```

Our preprocessing pipeline is intentionally minimal and stable: we apply `transforms.ToTensor()`. This conversion matters for two reasons. First, it converts images into PyTorch tensors with the expected channel-first format. Second, it rescales pixel values from the original `[0, 255]` range to `[0.0, 1.0]`, which is a convenient and standard input scale for neural generative models.

A quick single-sample inspection confirms the expected tensor format and value range:

```text
One sample x shape: torch.Size([1, 28, 28]) | dtype: torch.float32 | min/max: 0.0 1.0
One sample y (label id): 9
```

### Train/validation split and DataLoaders

To monitor generalization during training, we split the original 60,000 training samples into a 90/10 train–validation split. With this setup, we train on 54,000 samples and validate on 6,000 samples while keeping the official 10,000-image test set untouched for final evaluation:

```text
Train size: 54000
Val size  : 6000
Test size : 10000
```

We then construct DataLoaders using a batch size of 128. Training batches are shuffled to improve stochastic optimization, while validation and test loaders are kept deterministic (`shuffle=False`) to ensure evaluation is stable and comparable across runs. We also keep `num_workers=0` for notebook stability. A batch-level shape check confirms the model will receive tensors in the expected format:

```text
One batch x: torch.Size([128, 1, 28, 28]) torch.float32 | y: torch.Size([128]) torch.int64
```

### Sanity check (visual verification)

Before introducing any model, we visually inspect a randomly sampled batch of 25 images along with their labels. This is a simple but critical integrity check: it confirms that (i) tensors are properly scaled and shaped, (ii) images are not corrupted, and (iii) labels match the content shown in the grid.

![Sanity Check - 25 Samples](./images/sanity_check_image.png)

### Exploratory data analysis

The goal of the EDA in this phase is not to overanalyze the dataset, but to confirm basic structure and identify any issues that would affect training and evaluation.

We first examine class counts in the train split. The resulting bar chart shows that the split remains effectively balanced across all 10 classes, so the model is not expected to face a strong class-frequency bias during training.

![Class Distribution in Train Split](./images/class_distribution.png)

Next, we inspect the global pixel intensity distribution (normalized to `[0,1]`). The histogram exhibits a dominant spike near 0, indicating that the dataset contains a large proportion of dark background pixels. The long tail toward higher intensity values corresponds to the brighter foreground regions (the clothing items and their boundaries). This is expected for Fashion-MNIST and is useful context when interpreting reconstruction loss: the background occupies most pixels, while the foreground carries most of the semantic structure.

![Pixel Intensity Histogram](./images/pixel_histogram.png)

Finally, we compute the mean intensity per image and plot its distribution. Most samples have relatively low average intensity (again reflecting background dominance), but there is a clear spread depending on garment type and how much of the image area is occupied by the foreground object.

![Per-image Mean Intensity](./images/mean_intensity.png)

### Phase 1 summary

At the end of Phase 1, we have a reproducible and verified data pipeline: the dataset is correctly loaded, transformed into `[0,1]` tensors of shape `[B,1,28,28]`, split into train/validation/test, and validated via both numeric checks and visual sanity inspection. The EDA confirms an approximately balanced class distribution and highlights the expected background-heavy intensity structure of Fashion-MNIST. This setup provides a clean foundation for Phase 2, where we implement and train the baseline VAE.


## Phase 2 : Baseline VAE & Quality Improvement (Competition Upgrade)

Phase 2 focuses on building a variational generative model **from scratch** and then improving it in a controlled way. We first implement a **baseline MLP-VAE** to validate the full VAE pipeline (encoder/decoder, reparameterization, and ELBO-style loss). Next, we introduce a stronger **convolutional residual VAE** to improve reconstruction quality and sampling realism, and we quantify improvements using the project’s fixed FID evaluation protocol.

###  Baseline model: MLP-VAE (from scratch)

Our baseline is a fully-connected VAE that maps a flattened 28×28 image (784-dimensional vector) into a compact latent code, and then decodes it back to image space. The encoder outputs the parameters of a diagonal Gaussian posterior, `μ(x)` and `logσ²(x)`, and we sample latent variables using the **reparameterization trick**:

$`z = \mu + \epsilon \odot \sigma,\quad \epsilon \sim \mathcal{N}(0, I)`$

For training, we optimize the standard VAE objective using:
- **Reconstruction term:** `BCEWithLogitsLoss` applied per-pixel (stable logits-based BCE)
- **Regularization term:** KL divergence \( $`D_{KL}(q(z|x)\,\|\,\mathcal{N}(0, I))`$ \)
- **Total:** `recon + β * KL` with a fixed **β = 1.0**

Training configuration and outcome:
```text
latent_dim: 16 | epochs: 10 | lr: 0.001 | beta: 1.0
````

Progress (selected epochs):

```text
[Baseline VAE] Epoch 01/10 | train total=293.06 | val total=262.51
[Baseline VAE] Epoch 10/10 | train total=240.91 | val total=241.76
```

Final baseline test metrics:

```text
Baseline TEST metrics: {'recon': 230.31, 'kl': 12.09, 'total': 242.40}
```

We visualize the baseline behavior in two standard ways:

1. **Reconstructions:** input images vs. decoded outputs
2. **Sampling:** images generated by sampling ( $`z \sim \mathcal{N}(0,I)`$ )

![Baseline Reconstructions](./images/baseline_recon.png)
![Baseline Samples](./images/baseline_samples.png)

---

###  Quality improvement: Advanced Residual ConvVAE (competition-oriented)

To improve fidelity and reduce FID, we replace the dense architecture with a deeper **convolutional VAE** augmented with **Residual Blocks** and Batch Normalization. The motivation is practical: convolutional feature extraction preserves spatial structure (edges, contours, textures), and residual connections improve optimization stability in deeper networks.

Key upgrades relative to the baseline:

* **Residual convolutional encoder/decoder:** improves representation of local spatial patterns and reduces blurring.
* **Larger latent capacity:** latent dimension increased from **16 → 64** to model more variation.
* **KL warm-up (annealing):** rather than enforcing the full KL penalty immediately, we linearly warm up β during early training:
  
$`\beta_t = \min\left(1, \frac{t}{\text{warmup\_epochs}}\right)`$
  
  This allows the model to learn reasonable reconstructions first, then gradually enforces a structured latent space.
* **Optimization schedule:** `AdamW` with weight decay and a cosine LR schedule for smoother convergence.

Training configuration:

```text
latent_dim: 64 | epochs: 30 | lr: 0.001 | warmup_epochs: 10
```

Progress (selected epochs):

```text
[Advanced VAE] Epoch 01/30 | Beta=0.10 | train recon=233.78 | val recon=225.34
[Advanced VAE] Epoch 10/30 | Beta=1.00 | train recon=222.09 | val recon=221.93
[Advanced VAE] Epoch 30/30 | Beta=1.00 | train recon=217.55 | val recon=218.22
```

Advanced model test metrics:

```text
Advanced ResNet TEST metrics: recon=219.05, kl=17.77, total=236.82
```

As expected, the KL term increases with higher latent dimensionality (the model uses more capacity), while reconstruction improves and the overall objective decreases compared to the baseline.

![Advanced ResNet Reconstructions](./images/resnet_recon.png)
![Advanced ResNet Samples](./images/resnet_samples.png)

---

###  Quantitative evaluation: FID using the fixed project classifier

To evaluate generation quality for the competition, we compute **FID** using the **provided frozen ResNet18-based classifier** as the feature extractor (weights are kept unchanged). We compare Gaussian statistics of feature embeddings between:

* **10,000 real test images**, and
* **10,000 generated samples** from each VAE.

Results:

```text
FID (Baseline MLP-VAE): 9.39
FID (Advanced ResNet-VAE): 5.89
FID Improvement: 3.50 points 🔥
```

| Model                                     |  FID (↓) |
| ----------------------------------------- | -------: |
| Baseline MLP-VAE (latent=16)              |     9.39 |
| **Advanced Residual ConvVAE (latent=64)** | **5.89**🏆 |




## Phase 3 : β-VAE and Latent Space Interpretability

Phase 3 investigates how the VAE latent space changes when we explicitly scale the KL regularization term by a factor **β** (i.e., a β-VAE). Concretely, we optimize the same objective as Phase 2 but replace the regularization weight with β:

$`
\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot D_{KL}(q(z|x)\,\|\,\mathcal{N}(0,I))
`$

Intuitively, smaller β allows the model to encode more information in \(z\) (often improving reconstructions), while larger β pushes the posterior toward the unit Gaussian prior (often improving structure/disentanglement, but at the cost of reconstruction fidelity).

---

###  Experimental setup

To isolate the effect of β, we trained **three separate instances** of our convolutional VAE (the same Advanced/Residual ConvVAE family used in Phase 2) with identical training settings, except for β:

- **Latent dimension:** 32  
- **Epochs per run:** 10  
- **Learning rate:** 1e-3  
- **β values:** 0.6, 0.9, 5.0  

We re-seed the run at the start of each β experiment to keep comparisons consistent.

---

###  Quantitative results: reconstruction vs. regularization

The table below reports the **test-set** decomposition of the VAE objective into reconstruction loss, KL divergence, and total loss. It directly shows the trade-off induced by β:

```text
beta   recon      kl         total
0.6    218.08     23.50      232.18
0.9    223.62     17.35      239.24
5.0    249.59      5.50      277.08
````

Two effects are clearly visible. With **β = 0.6**, the model achieves the best reconstruction score (lowest recon loss), but the KL term is relatively large, indicating that the posterior is allowed to carry more information and deviate further from the prior. As **β increases**, the KL term drops substantially (e.g., **β = 5.0** yields KL ≈ 5.50), meaning the latent distribution is forced closer to (\mathcal{N}(0,I)). The cost is a significant degradation in reconstruction quality (recon increases to ≈ 249.59), reflecting a tighter information bottleneck.

---

###  Qualitative results: latent traversal

To probe interpretability, we perform **latent traversal**. For each trained β-model, we:

1. sample a batch from the loader and compute encoder means ( \mu(x) ),
2. choose the **top 5 latent dimensions** with the highest variance across the batch (dimensions that “move” the most in practice),
3. pick a single reference image (we attempt to select a sample from class id **3**, falling back to the first sample if unavailable),
4. set the base latent code to ( z_0 = \mu(x_{ref}) ) (deterministic traversal),
5. sweep one latent dimension at a time across **7 values** in ([-3, 3]), while holding all other dimensions fixed.

This produces a 5×7 grid per β setting.

#### β = 0.6 (weak regularization)

With lower β, the model is primarily driven by reconstruction quality. In the traversal grid, changing a single latent coordinate can cause **large and sometimes abrupt semantic shifts**, including class-level morphing (e.g., dress-like silhouettes transitioning toward shoe-like shapes in some rows). This suggests that semantic factors are more **entangled**: one coordinate may influence multiple attributes at once.

![Latent Traversal Beta 0.6](assets/traversal_0.6.png)

#### β = 0.9 (moderate regularization)

This setting tends to produce more stable transitions than β=0.6, while still preserving decent reconstruction quality. In our grid, some dimensions show smoother changes in geometry (width/length/shape), although occasional semantic drift is still visible.

![Latent Traversal Beta 0.9](./images/traversal_0.9.png)

#### β = 5.0 (strong regularization)

With strong KL pressure, the latent codes are constrained to remain closer to the prior. Traversals often become more “conservative”: many dimensions induce smaller, more localized changes, and several rows show **limited visual sensitivity** to the swept coordinate. This behavior is consistent with a tighter bottleneck where the decoder learns to rely less on certain latent dimensions (some dimensions contribute little, while a subset still controls the dominant variation). The trade-off is clearly reflected in the quantitative results: reconstruction quality drops noticeably at this β.

![Latent Traversal Beta 5.0](./images/traversal_5.0.png)

---

### Phase 3 summary

Across β values, we observe the expected β-VAE behavior: **smaller β improves reconstructions but yields a less structured/less stable latent traversal**, while **larger β enforces a stronger information bottleneck**, decreasing KL divergence but also worsening reconstructions and reducing sensitivity along some latent directions. These observations motivate the next step (Phase 4), where we introduce **conditional generation** to control outputs explicitly using class labels.


## Phase 4 : Conditional VAE (CVAE) and Directed Generation

Phase 4 upgrades our generative pipeline from an unconditional VAE into a **Conditional VAE (CVAE)**. The goal is explicit control: instead of sampling “any” Fashion-MNIST item from random noise, we condition generation on a target class label $`y\in\{0,\dots,9\}`$ and synthesize samples that match the requested category.

---

###  Model design: robust conditioning in both encoder and decoder

We implement a convolutional CVAE where both the encoder and decoder receive class information. The label $`y`$ is converted to a one-hot vector and injected in a way that makes it difficult for the network to ignore conditioning:

**Encoder-side conditioning.**  
We expand the one-hot label spatially and concatenate it with the input image, turning the encoder input from `[1, 28, 28]` into `[1 + 10, 28, 28]`. This forces the inference network to learn $`q(z \mid x, y)`$, not just $`q(z \mid x)`$.

**Decoder-side multi-scale conditioning.**  
We condition the decoder at multiple stages:
- concatenate $`[z, y]`$ before the initial fully-connected projection,
- concatenate spatial label maps again at feature map resolutions **7×7**, **14×14**, and **28×28** before the final output.

This multi-scale injection helps preserve label control throughout the decoding process and reduces class confusion (especially for visually similar categories such as **Coat** vs **Shirt**).

---

###  Training protocol and test metrics

We train the CVAE with the same VAE objective as earlier phases, but now conditioned on \(y\):

$`
\mathcal{L} = \mathcal{L}_{recon}(x,\hat{x}) + \beta \cdot D_{KL}(q(z|x,y)\,\|\,\mathcal{N}(0,I))
`$

To stabilize training, we use **KL warm-up (annealing)** for the first 10 epochs:
$`
\beta_t = \min\left(1, \frac{t}{10}\right)
`$
and apply a cosine learning-rate schedule over the full run.

Configuration:
```text
latent_dim: 32
epochs: 40
optimizer: Adam (lr=1e-3) + CosineAnnealingLR
KL warm-up: 10 epochs (beta ramps from 0.1 → 1.0)
````

Final test decomposition (β=1.0):

```text
CVAE Test Metrics: {'recon': 219.30, 'kl': 14.97, 'total': 234.27}
```

---

### Directed generation: 20 samples per class

To demonstrate controllability, we generate **20 samples for each of the 10 classes** by sampling $`z \sim \mathcal{N}(0, I)`$ and decoding with a fixed label (y=c). The resulting grid shows consistent class identity across rows and meaningful intra-class variation across columns.

![CVAE Conditional Generation Grid (20 per class)](./images/conditional_grid.jpg)

---

### Quantitative controllability: classifier-based accuracy (frozen weights)

To measure whether conditioning actually works, we evaluate generated images using the provided frozen **FashionResNet18** classifier (same evaluation model used for FID in Phase 2). Since the classifier expects normalized inputs, we apply the classifier’s stored mean/std normalization before inference.

Protocol:

* generate **500 synthetic images per class** (5,000 total),
* feed them through the frozen classifier,
* report the fraction predicted as the intended target class.

Per-class accuracy:

| Class       |    Accuracy | Class      | Accuracy |
| ----------- | ----------: | ---------- | -------: |
| T-shirt/top |      91.40% | Sneaker    |   98.80% |
| Trouser     | **100.00%** | Bag        |   97.40% |
| Pullover    |      96.00% | Ankle boot |   99.80% |
| Dress       |      96.80% | Sandal     |   87.20% |
| Coat        |      63.80% | Shirt      |   75.40% |

**Overall conditional generation accuracy: 90.66%** 🔥

Interpretation. The CVAE achieves strong control on structurally distinct classes (e.g., **Trouser**, **Ankle boot**, **Sneaker**). The weakest classes are those with high visual overlap in Fashion-MNIST (notably **Coat** and **Shirt**), where even a strong classifier tends to confuse categories due to similar silhouettes and textures.

---

### Brief Comparison: Unconditional VAE vs. Conditional CVAE

In the baseline **Unconditional VAE** (Phase 2), the latent variable \(z\) must implicitly encode *everything* at once: both the **class identity** (e.g., “dress” vs. “shoe”) and the **style factors** (e.g., brightness, thickness, minor geometric variations). As a result, sampling $`z \sim \mathcal{N}(0,I)`$ provides **no direct control** over what category will be generated. The decoder outputs items from arbitrary classes, and occasionally produces ambiguous shapes that appear to mix multiple categories—an expected consequence of entangled class+style information inside $`z`$.

In contrast, the **CVAE** explicitly injects the class label \(y\) into both the encoder and decoder. This shifts the responsibility for “what class is it?” from \(z\) to \(y\). The latent code \(z\) can now focus primarily on **intra-class variation and style**, while the label provides **directed generation**: we can request a specific category (e.g., “Ankle boot” or “Trouser”) and still preserve meaningful stylistic diversity driven by random sampling in \(z\).

The figure below illustrates this difference. The unconditional VAE produces random categories (top row), while the CVAE produces samples consistently aligned with the chosen target class (bottom row).

![Generative Control Comparison: VAE vs. CVAE](./images/compare_label.png)
---

### Phase 4 summary

By conditioning both inference and generation on class labels—and reinforcing label information at multiple decoder resolutions—the CVAE provides reliable directed synthesis on Fashion-MNIST. The classifier-based evaluation confirms strong controllability with **90.66%** overall accuracy, with most remaining errors concentrated in visually ambiguous categories (especially **Coat** and **Shirt**).


## Phase 5 : Final Report and Consolidated Results

Phase 5 consolidates all findings from Phases 1–4 into a single, structured summary. The goal is to present (i) a final test-set comparison table across all model configurations, (ii) a small set of representative qualitative grids from each phase, and (iii) a short conclusion identifying the best configuration(s) depending on the evaluation objective (reconstruction quality, generative fidelity, interpretability, or controllability).

---

###  Final quantitative comparison (test set)

The table below reports the final decomposition of the VAE objective on the **test set** for every model/configuration used in the project. Each entry includes:
- **Recon Loss**: reconstruction term (pixel-wise BCE with logits)
- **KL Loss**: regularization term \(D_{KL}(q(z|x)\,\|\,\mathcal{N}(0,I))\)
- **Total Loss**: `Recon + β·KL` (or `Recon + KL` when β=1)

| # | Model | Recon Loss | KL Loss | Total Loss |
|---:|------|-----------:|--------:|-----------:|
| 0 | Baseline MLP-VAE (β=1.0) | 231.3149 | 11.4990 | 242.8139 |
| 1 | Improved ConvVAE (β=1.0) | 219.0841 | 17.9747 | 237.0588 |
| 2 | Beta-VAE (β=0.6) | **218.0755** | 23.5012 | **232.1763** |
| 3 | Beta-VAE (β=0.9) | 223.6230 | 17.3534 | 239.2410 |
| 4 | Beta-VAE (β=5.0) | 249.5887 | **5.4978** | 277.0776 |
| 5 | Conditional ConvVAE (CVAE Pro) | 219.3008 | 14.9707 | 234.2714 |

**How to read this table.**  
- The **best raw reconstruction** (lowest Recon Loss) is achieved by **β=0.6** (Beta-VAE), which is expected because weaker KL pressure allows the model to store more information in the latent code.  
- The **strongest regularization** effect (lowest KL Loss) appears at **β=5.0**, indicating a much tighter information bottleneck—at the cost of noticeably worse reconstructions.  
- Relative to the baseline MLP-VAE, the **Improved ConvVAE (β=1.0)** reduces reconstruction loss substantially, confirming that convolutional/residual structure better captures spatial patterns in Fashion-MNIST.

---

###  Key qualitative evidence (required grids)

Below are the phase-wise qualitative artifacts that support the quantitative results above. These should be included as compact grids in the final report:

**Phase 2 — Baseline (MLP-VAE)**
- Reconstruction grid (≥20): input vs reconstruction  
- Sampling grid (≥50): unconditional samples  
![Baseline Reconstructions](assets/baseline_recon.png)  
![Baseline Samples](assets/baseline_samples.png)

**Phase 2 — Quality improvement**
- Before/after comparison: reconstructions and samples for the improved convolutional model  
![Improved ConvVAE Reconstructions](assets/resnet_recon.png)  
![Improved ConvVAE Samples](assets/resnet_samples.png)

**Phase 3 — Latent traversal (β sweep)**
- Latent traversal grids (5 dims × 7 values) for each β  
![Latent Traversal β=0.6](assets/traversal_0.6.png)  
![Latent Traversal β=0.9](assets/traversal_0.9.png)  
![Latent Traversal β=5.0](assets/traversal_5.0.png)

**Phase 4 — Conditional generation (all 10 classes)**
- Conditional generation grid (20 samples per class)  
- Classifier-based directed accuracy (500 samples per class)  
![CVAE Conditional Grid](assets/conditional_grid.jpg)

---

###  Short conclusion (best settings + trade-offs)

There is no single “best” configuration for all objectives, so we summarize the results by target goal:

**1) Best reconstruction / lowest total objective**  
If the priority is minimizing reconstruction error and the overall VAE objective on the test set, **Beta-VAE with β=0.6** performs best in our experiments (Recon ≈ 218.08, Total ≈ 232.18). This aligns with the expected behavior that weaker KL pressure improves reconstruction fidelity.

**2) Best structured latent / strongest bottleneck**  
For a more constrained latent distribution (closest to the prior), **β=5.0** achieves the lowest KL (≈ 5.50). However, this comes with a major reconstruction penalty (Recon ≈ 249.59). In practice, this setting is useful when interpretability/disentanglement matters more than pixel-level reconstruction quality.

**3) Best general-purpose unconditional generator**  
The **Improved ConvVAE (β=1.0)** provides a strong balance: substantially improved reconstructions vs. the baseline (231.31 → 219.08) while maintaining a meaningful KL term. This model also served as the strongest unconditional backbone for realistic sample generation in later analyses.

**4) Best controllability (directed generation)**  
The **Conditional ConvVAE (CVAE Pro)** is the best choice when control is required. It achieves recon/test performance comparable to the improved unconditional model (Recon ≈ 219.30, Total ≈ 234.27) while enabling class-directed sampling. Using the frozen project classifier, the CVAE achieved **90.66% overall directed accuracy**, with most errors concentrated in visually ambiguous categories (e.g., Coat vs Shirt).

---

### Limitations and future work

- **Ambiguous classes:** Fashion-MNIST contains visually overlapping categories (Coat/Shirt/Pullover), which limits both classifier-based evaluation and conditional separability.
- **Single-metric evaluation:** The objective terms (Recon/KL/Total) measure training consistency, but perceptual fidelity is better captured by feature-based metrics (FID) and human inspection.
- **Potential improvements:** stronger decoders (attention / deeper residual stacks), richer priors (mixture-of-Gaussians, VampPrior), and improved conditional mechanisms (class embeddings + conditional normalization) could improve both fidelity and controllability.

---

### Phase 5 summary

Phase 5 confirms that our pipeline is complete and internally consistent across all project requirements: baseline VAE construction, quality improvement, β-controlled latent analysis, and full conditional generation. Quantitative test metrics and phase-specific qualitative grids together provide a clear narrative of the trade-offs between reconstruction quality, latent structure, and controllability.

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

![Baseline Reconstructions](assets/baseline_recon.png)
![Baseline Samples](assets/baseline_samples.png)

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

![Advanced ResNet Reconstructions](assets/resnet_recon.png)
![Advanced ResNet Samples](assets/resnet_samples.png)

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




## Phase 3: VAE & Latent Space Interpretability

In this phase, we implemented a -VAE framework to systematically investigate the **Information Bottleneck** trade-off between reconstruction fidelity and latent space disentanglement. By scaling the Kullback-Leibler (KL) divergence term with a hyperparameter , we control the exact capacity of the latent space.

### 1. Quantitative Analysis: The  Trade-off

We trained three separate Advanced ResNet-VAE models with . The evaluation on the test set explicitly demonstrates the theoretical trade-off:

| Model () | Reconstruction Loss ↓ | KL Divergence ↓ | Total Loss |
| --- | --- | --- | --- |
| **** | **218.07** | 23.50 | 232.17 |
| **** | 223.62 | 17.35 | 239.24 |
| **** | 249.58 | **5.49** | 277.07 |

* **Low  (0.6):** The model prioritizes minimizing the reconstruction loss, resulting in sharper images. However, it applies less penalty to the latent distribution (KL: 23.50), leading to an entangled space.
* **High  (5.0):** The network heavily penalizes deviations from the prior  (KL drops to 5.49). The trade-off is restricted information flow, causing blurrier reconstructions (Recon rises to 249.58).


### 2. Qualitative Analysis: Latent Space Traversal

To visually validate the interpretability of our learned representations, we performed **Latent Space Traversal**. We selected a fixed reference image, identified the top 5 most "active" latent dimensions, and swept each dimension across a range of  while freezing the others.

**Traversal with  (Entangled Space):**
With a lower , the features are highly entangled. For instance, traversing a single dimension (e.g., Dim 6) abruptly morphs a dress into an entirely different class (a shoe). This indicates that a single latent variable encodes multiple, unrelated physical concepts simultaneously.

> `![Latent Traversal Beta 0.6](assets/traversal_0.6.png)`

**Traversal with  (Balanced Space):**
This setting provides a middle ground, showing slightly smoother transitions between structural forms while maintaining decent visual sharpness.

> `![Latent Traversal Beta 0.9](assets/traversal_0.9.png)`

**Traversal with  (Posterior Collapse & Disentanglement):**
With a strong regularization penalty (), we observe a famous theoretical phenomenon known as **Posterior Collapse (Latent Pruning)**. The model is so heavily penalized for storing information that it completely "turns off" certain latent variables to save KL cost.
Notice how traversing **Dim 14** and **Dim 18** results in absolutely zero change to the output image; the decoder has learned to completely ignore these "dead" dimensions as they contain no mutual information. However, the dimensions that *do* survive the bottleneck (like **Dim 17**, which smoothly transitions a sleeveless dress into a long-sleeved top) represent highly isolated, disentangled, and meaningful semantic features.

> `![Latent Traversal Beta 5.0](assets/traversal_5.0.png)`

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

## Phase 1 summary

At the end of Phase 1, we have a reproducible and verified data pipeline: the dataset is correctly loaded, transformed into `[0,1]` tensors of shape `[B,1,28,28]`, split into train/validation/test, and validated via both numeric checks and visual sanity inspection. The EDA confirms an approximately balanced class distribution and highlights the expected background-heavy intensity structure of Fashion-MNIST. This setup provides a clean foundation for Phase 2, where we implement and train the baseline VAE.


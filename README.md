# ML_fall2025-fashionmnist-vae


## Phase 1: Data Preparation & Exploratory Data Analysis

**Setup & Dataset Overview**
This project is implemented in PyTorch with CUDA enabled to ensure efficient training of our generative models. To guarantee fully deterministic and reproducible results across all experiments, we enforced a strict global manual seed. The core dataset used is **Fashion-MNIST**, containing  grayscale images across 10 clothing categories. Out of the 70,000 total images, we reserved 10,000 for the final test set and applied a 90/10 train-validation split on the remaining 60,000 to continuously monitor generalization and prevent overfitting.

**Preprocessing Pipeline**
Our data preprocessing pipeline is streamlined using `torchvision.transforms`. We simply apply `ToTensor()`, which effectively converts the image matrices into PyTorch tensors and normalizes the original `[0, 255]` pixel intensities into a standardized `[0.0, 1.0]` continuous range (shape: `[B, 1, 28, 28]`).

**Sanity Checks & Visualizations**
Before constructing the VAE, it was crucial to verify the integrity of our data pipeline. We extracted a random batch of 25 images to visually confirm that the tensor transformations were correct and that the labels perfectly aligned with the images.

> *(Insert your sanity check grid here)* > `![Sanity Check - 25 Samples](./images/sanity_check_image.png)`

**Exploratory Data Analysis (EDA)**
To deeply understand the structural properties of our data, we performed several statistical analyses. First, the class distribution bar chart confirmed that our training subset remains perfectly balanced across all 10 categories, ensuring no class bias during training.

> *(Insert your class distribution chart here)* > `![Class Distribution in Train Split](./images/class_distribution.png)`

Furthermore, we analyzed the global pixel intensity and the per-image mean intensity. The global intensity histogram reveals a massive peak near 0, demonstrating that the vast majority of pixels belong to the dark background. The long tail extending toward 1.0 represents the brighter foreground clothing items. This distinct foreground-background separation provided a strong empirical justification for choosing Binary Cross Entropy (`BCEWithLogitsLoss`) as our reconstruction loss in the subsequent phases.

> *(Insert your intensity histograms here)* > `![Pixel Intensity Histogram](./images/pixel_histogram.png)`
> `![Per-image Mean Intensity](./images/mean_intensity.png)`






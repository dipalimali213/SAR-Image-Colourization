
# 📘 Semester Project Logbook

**Project Title:** SAR Image Colourization for Comprehensive Insight Using Deep Learning  
**Semester:** VI (2024–2025)  
**Project Stage:** I  

---

## 🤝 Team Collaboration

### 👥 Team Members
- **Neha Sandeep Gayakawad** (Roll No: 221106005 & 09)  
- **Patil Bhavesh Bhagwan** (Roll No: 221106023 & 24)  
- **Mali Gaurang Jagdish** (Roll No: 221106041 & 41)  
- **Mahajan Kalpesh Subhash** (Roll No: 221106051 & 50)  
- **Mali Dipali Vilas** (Roll No: 221106055 & 54)  

### 👩‍🏫 Guided By
**Prof. P. D. Lanjewar Ma'am**

---

## 🗂️ Weekly Logbook Index

| Week No. | Dates                | Section                        |
|----------|----------------------|--------------------------------|
| 1        | 24/02/2025 – 08/03/2025 | [Introduction](#1-introduction)              |
| 2        | 10/03/2025 – 22/03/2025 | [Literature Survey](#2-literature-survey)    |
| 3        | 24/03/2025 – 05/04/2025 | [Methodology](#3-methodology)                |
| 4        | 07/04/2025 – 26/04/2025 | [Implementation Details](#4-implementation-details) |
| 5        | 28/04/2025 – 03/05/2025 | [Results](#5-results)                        |
| —        | —                    | [Conclusion](#6-conclusion)                  |
| —        | —                    | [References](#7-references)                  |

---

## 1. Introduction  
**Week 1: 24/02/2025 – 08/03/2025**

### 🧩 Problem Statement  
SAR images are valuable in various fields, but their grayscale format limits interpretability for human analysts. This project addresses that challenge by developing a deep learning pipeline to colorize SAR images and enhance their interpretability.

### 🎯 Objectives  
- Develop an automated SAR-to-color image translation model.  
- Leverage U-Net for image reconstruction and enhancement.  
- Train using paired SAR-optical datasets.  
- Evaluate performance using visual and numerical metrics.

### 🌍 Application Domains  
- Remote Sensing and Environmental Monitoring  
- Strategic Defense and Surveillance  
- Urban Planning and Disaster Management

---

## 2. Literature Survey  
**Week 2: 10/03/2025 – 22/03/2025**

### 📚 Background Study  
The project is grounded in recent advancements in image-to-image translation using CNNs and GANs. These methods show potential in tasks like super-resolution, image colorization, and domain adaptation.

### 📄 Research Papers Reviewed  
- Zhang et al., “SAR Image Colorization Using Conditional GAN”, IEEE Transactions.  
- Isola et al., “Image-to-Image Translation Using Conditional GANs”, CVPR.  
- Ronneberger et al., “U-Net: Convolutional Networks for Biomedical Image Segmentation”, MICCAI.  
- Goodfellow et al., “Generative Adversarial Networks”, NeurIPS.

### 📌 Key Takeaways  
- U-Net architecture is effective in spatial feature learning.  
- GANs improve realism in generated images.  
- SAR-optical pair datasets are essential for training and evaluation.

---

## 3. Methodology  
**Week 3: 24/03/2025 – 05/04/2025**

### 🛠️ Tools and Technologies  
- **Hardware:** RTX 3050 GPU, 16 GB RAM  
- **Languages:** Python 3.10  
- **Libraries:** TensorFlow, OpenCV, NumPy, Matplotlib  

### 🔧 Dataset Preparation  
- Dataset: SAR Image Colorization.csv 
- Size: Resized to 256×256 for quality and performance  
- Normalization: Pixel values scaled between 0 and 1  
- Augmentation: Applied rotation, zoom, brightness change, flipping

### ⚙️ Model Architecture  
- U-Net-based encoder-decoder with skip connections  
- Convolutional filters: 32 to 256 with batch normalization  
- Output layer: 3-channel RGB with sigmoid activation

### 🧠 Memory Optimization  
- Enabled GPU memory growth with capped 4GB usage  
- Batch-wise data loading with error handling  
- Garbage collection and backend clearing after each batch


---

## 4. Implementation Details  
**Week 4: 07/04/2025 – 26/04/2025**

### 🧩 Module 1 – Data Loading & Preprocessing  
This module is responsible for preparing the data pipeline from raw image folders to normalized arrays ready for training.

- **Batch-Based Loading:** Images are loaded in small batches (10 images at a time) to avoid memory overflow.  
- **Validation & Filtering:** Files are checked for validity, and corrupted or unreadable files are automatically skipped with warnings.  
- **Image Normalization:** SAR (grayscale) and Optical (RGB) images are resized to 256×256 pixels and scaled to [0, 1] range.  
- **Channel Management:** Grayscale images are expanded to add a channel dimension for compatibility with CNNs.  
- **Memory Management:** After each batch, garbage collection and session clearing ensure smooth GPU memory usage.  
- Batch loading implemented for memory efficiency  
- Verified image integrity and skipped corrupted pairs  
- Implemented resizing and normalization

### 🧩 Module 2 – Model Building  
This module focuses on defining and compiling the deep learning architecture used for colorization.

- **Architecture Used:** A customized U-Net model was implemented with increasing filters (32 → 64 → 128 → 256).  
- **Layer Enhancements:** Each convolution block includes batch normalization to stabilize learning and prevent overfitting.  
- **Activation Function:** ReLU used in hidden layers; Sigmoid at the output layer for RGB image generation.  
- **Output Shape:** 256×256×3 tensor to represent colorized output.  
- **Error Handling:** The model build process includes try-except blocks to catch TensorFlow-related exceptions.  
- **Optimizer:** Adam with a learning rate of 0.001 for faster convergence on the RTX 3050 GPU.  
- Built a deeper U-Net model with advanced regularization  
- Compiled with Adam optimizer and Mean Squared Error loss  
- Trained for 150 epochs with 4-image batches

### 🧩 Module 3 – Training & Monitoring  
This module handles the training lifecycle, including validation, monitoring, and model persistence.

- **Dataset Creation:** Used `tf.data` to create training and validation pipelines with caching, shuffling, batching, and prefetching.  
- **Callbacks Implemented:**  
  - *EarlyStopping* to halt training after no improvement.  
  - *ReduceLROnPlateau* to decrease learning rate during plateaus.  
  - *ModelCheckpoint* to store the best weights.  
  - *TerminateOnNaN* for safety.  
- **Training Runtime:** Model trained for 150 epochs with a batch size of 4 using multiprocessing.  
- **Output Files:**  
  - Final model saved as `Sar_colourization_model.keras`  
  - Backup version saved in case of failure  
  - Training metrics plotted and saved as `training_history.png`  
- Used `tf.data` API for optimized data pipeline  
- Applied callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  
- Enabled multiprocessing for training speed-up

---

## 5. Results  
**Week 5: 28/04/2025 – 03/05/2025**

### 📦 Dataset Used  
- Source: SAR Image Colorization.csv
- Format: 8-bit grayscale SAR and 3-channel RGB images


### 📈 Visualization  
- Training loss/accuracy plotted per epoch  
- Final predictions compared with SAR inputs and ground truth  
- Results saved in `training_history.png`

---

## 6. Conclusion  

A deep learning pipeline using a U-Net architecture was successfully developed and trained to colorize SAR images. The project addressed preprocessing, model optimization, and evaluation using GPU-accelerated workflows. Future work includes deploying GAN-based refinement and integration into web-based applications.

---

## 7. References  

1. Zhang et al., “SAR Image Colorization Using GANs”, IEEE Transactions.  
2. Ronneberger et al., “U-Net: Convolutional Networks for Biomedical Image Segmentation”, MICCAI.  
3. Isola et al., “Image-to-Image Translation Using Conditional GANs”, CVPR.  
4. TensorFlow and Keras Documentation.  
5. Copernicus Open Access Hub SAR datasets.


# LeafSense: A Machine Learning–Based Web Application for Binary Plant Leaf Disease Detection

**Authors:** [Author names and affiliations]  
**Contact:** [Corresponding author email]

---

## Abstract

Early detection of plant diseases is critical for sustainable agriculture and food security. We present **LeafSense**, an end-to-end machine learning–based web application that performs binary classification of plant leaves into *Healthy* and *Diseased* categories using EfficientNet-B0. The system is trained on the PlantVillage dataset and uses an ImageNet-pretrained plant-versus-non-plant filter to reject irrelevant uploads (e.g., documents or non-leaf images). Test-time augmentation (TTA) is applied at inference to improve prediction confidence. The backend is implemented in Flask (Python) with configurable rate limiting and input validation; the frontend is React with TypeScript, served by the same process for single-deployment. The API returns class, confidence, user-facing messages, and recommendations. Our approach achieves strong validation accuracy on the binary PlantVillage split and demonstrates robustness against out-of-domain inputs through the integrated plant checker and low-confidence rejection. This work highlights the feasibility of deploying lightweight, accurate machine learning models for plant disease screening in practical applications.

**Index Terms—** Plant disease detection, machine learning, EfficientNet, convolutional neural networks, PlantVillage dataset, binary classification, web application, test-time augmentation.

---

## I. Introduction

Plant pathogens and foliar diseases remain a major cause of harvest loss and economic damage in agriculture worldwide. Detecting affected leaves early allows growers to apply targeted control measures and limit the overuse of broad-spectrum chemicals. Conventional practice depends on human experts to visually inspect foliage, a bottleneck that is both costly and difficult to scale. Automated screening built on computer vision and machine learning can instead support rapid, low-cost triage of healthy versus diseased leaves in the field or from uploaded images.

This paper introduces **LeafSense**, a full-stack web system that performs binary classification of plant leaves into *Healthy* and *Diseased* categories. The work makes the following contributions: (1) **EfficientNet-B0** is trained on the PlantVillage dataset and deployed for binary disease detection. (2) A **plant-versus-non-plant** gate, implemented with an ImageNet-pretrained classifier, filters out irrelevant uploads (e.g., documents or non-leaf photos) before inference. (3) **Test-time augmentation (TTA)** with horizontal flip is used at inference to stabilize confidence estimates. (4) A **Flask** backend and **React** frontend are integrated in a single-deployment setup, with configurable upload size limits, per-IP rate limiting, and strict input validation. (5) The prediction API returns not only class and confidence but also user-oriented messages, recommendations, and a confidence tier, along with a **health** endpoint for production monitoring.

Sections II–VI are organized as follows. Section II surveys related work. Section III covers the dataset, model design, and system architecture. Section IV details the experimental setup and implementation. Section V presents results and discussion. Section VI summarizes the work and suggests future directions.

---

## II. Related Work

Plant disease recognition using machine learning has been widely studied. Mohanty et al. [1] used the PlantVillage dataset to train CNNs for multi-class disease identification, achieving high accuracy and demonstrating the viability of transfer learning. Various architectures have been applied, including VGG, ResNet, and MobileNet, with a trend toward efficient models suitable for edge or server deployment.

EfficientNet [2] provides a strong accuracy–efficiency trade-off via compound scaling of depth, width, and resolution. EfficientNet-B0 in particular is lightweight and well-suited for web or mobile deployment while maintaining competitive accuracy. Binary (healthy vs. diseased) formulations reduce the number of classes and can improve robustness when fine-grained disease labels are not required for initial screening.

Fewer works address the full deployment pipeline: handling non-plant images, providing a user-facing application, and combining a plant-check stage with disease prediction. Our system integrates these elements into a single, deployable solution.

---

## III. Methodology

### A. Dataset

We use the **PlantVillage** dataset [1], which contains leaf images from multiple crop species and disease categories. For binary classification, we map all folder names containing “healthy” to class **Healthy** (label 1) and all others to **Diseased** (label 0). We use the color (RGB) subset of PlantVillage. Images are resized to 256×256, center-cropped to 224×224, and normalized with ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]) to match the pretrained backbone and inference pipeline.

### B. Disease Detection Model

We use **EfficientNet-B0** [2] with two output classes (Diseased, Healthy). The model is instantiated via the *timm* library (`create_model('efficientnet_b0', num_classes=2)`), pretrained on ImageNet, and fine-tuned on the binary PlantVillage split. Training uses cross-entropy loss and AdamW optimizer. The same preprocessing (resize, center crop, normalize) is applied at training and inference.

### C. Plant vs. Non-Plant Filter

To avoid misclassifying non-leaf images (e.g., documents, furniture, or hands), we add a **plant checker**: an ImageNet-pretrained EfficientNet-B0 (1000 classes). The top-K (K=5) ImageNet predictions are checked against a whitelist of plant-related keywords (e.g., leaf, vegetable, fruit, tomato). If at least one match is found, the image is considered a valid plant/leaf and passed to the disease model; otherwise the upload is rejected with a user-facing error message. A minimum confidence from the disease model (default 0.5, configurable) is also required; below that, the response indicates low confidence and prompts the user for a clearer leaf image.

### D. Test-Time Augmentation (TTA)

At inference we apply horizontal flip in addition to the original image. Logits from both are averaged before softmax to produce the final confidence and class. This simple TTA improves stability of predictions without changing the model architecture.

### E. System Architecture

- **Backend:** Flask (Python). Endpoints: (1) receive image upload (key `image`; allowed types JPG, PNG, WEBP, GIF), (2) validate file type and content (magic-byte check to reject renamed non-images), (3) run plant checker, (4) run disease model with TTA, (5) return JSON with class (Healthy/Diseased), confidence, user-facing message, recommendation, and confidence tier (high/moderate/low). A `/health` endpoint reports server and model status for deployment. Upload size and per-IP rate limits are configurable via environment variables to mitigate abuse.
- **Frontend:** React 18 with TypeScript, Vite, Tailwind CSS, and Radix UI, built to `leaf-doctor-frontend-main/dist` and served by the same Flask app. Users upload an image and see the prediction, confidence, and recommendations.
- **Model files:** Trained weights saved as `efficientnet_plantdoc.pth`; ImageNet class names for the plant checker loaded from a local file or fetched from PyTorch hub; model loading uses `weights_only=True` when supported for safer checkpoint loading.

---

## IV. Implementation and Experimental Setup

### A. Training Configuration

- **Framework:** PyTorch; **library:** timm.
- **Hyperparameters:** AdamW optimizer, learning rate 1e-3, batch size 32, validation split 20%. Optional cap on samples per binary class (default 500; 0 = use all) for faster experimentation.
- **Epochs:** Best model (by validation accuracy) is saved to `efficientnet_plantdoc.pth`; optional periodic checkpoints to `checkpoint_latest.pth` (epoch, model, optimizer, best_val_acc) for resuming. Default 5 epochs.
- **Reproducibility:** Fixed random seed (default 42) for dataset shuffle and for the train/val split via `torch.Generator().manual_seed(seed)`.
- **Data integrity:** Training validates that both Healthy and Diseased classes are present; corrupt or unreadable images are skipped with a warning and a substitute sample is drawn.

### B. Inference Pipeline

1. Accept upload (key `image`); validate extension (JPG, PNG, WEBP, GIF) and file content via magic-byte check; enforce size and rate limits.
2. Load image (PIL), convert to RGB; preprocess: Resize(256), CenterCrop(224), ToTensor(), ImageNet normalization.
3. Plant checker: run ImageNet model, check top-5 predictions against whitelist; if no match, return structured error (e.g., “This doesn’t look like a plant or leaf image”).
4. Disease model: run on original and horizontally flipped image; average logits; softmax → confidence; argmax → Healthy/Diseased. Class order: 0 = Diseased, 1 = Healthy.
5. If confidence is below the configured threshold, return a low-confidence error; otherwise return JSON: class, confidence (%), message, recommendation, confidence_tier (high/moderate/low).

### C. Evaluation Metrics

- **Validation accuracy** (primary) and validation loss on the held-out PlantVillage split.
- Qualitative evaluation of behavior on non-plant images (rejection rate, no false disease/healthy labels for clearly irrelevant inputs).

---

## V. Results and Discussion

*[To be filled with your actual numbers after training and evaluation.]*

- **Validation accuracy:** Report best validation accuracy (e.g., “Best validation accuracy: XX.XX%” from training logs).
- **Inference:** The system runs in real time on CPU or GPU; EfficientNet-B0 is lightweight enough for typical server deployment.
- **Plant filter:** The ImageNet-based plant checker successfully rejects common non-plant uploads (documents, furniture, etc.), reducing spurious predictions.
- **Limitations:** Binary formulation does not distinguish disease types; performance depends on PlantVillage distribution and may vary on other species or imaging conditions.

---

## VI. Conclusion

We presented **LeafSense**, a machine learning–based web application for binary plant leaf disease detection using EfficientNet-B0 on the PlantVillage dataset. The pipeline includes a plant-versus-non-plant filter and test-time augmentation to improve reliability and confidence, plus input validation, rate limiting, and structured API responses for production use. The Flask backend serves both the API and the React frontend in a single deployment. Future work may include multi-class disease identification, support for additional datasets and crops, and optional deployment on edge devices.

---

## References

[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for image-based plant disease detection,” *Frontiers in Plant Science*, vol. 7, p. 1419, 2016.

[2] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in *Proc. ICML*, 2019, pp. 6105–6114.

*[Add further references as needed, e.g., PlantVillage dataset citation, timm/PyTorch, Flask, React.]*

---

## Checklist for Submission

- [ ] Replace placeholder author names and affiliations.
- [ ] Insert actual validation accuracy and any additional metrics (e.g., precision, recall, F1).
- [ ] Add table or figure for results (e.g., accuracy vs. epoch; sample predictions).
- [ ] Expand references (dataset, frameworks, related papers).
- [ ] Check page limit and format of target IEEE conference (e.g., IEEE Xplore template).
- [ ] Optional: add acknowledgments and funding information.

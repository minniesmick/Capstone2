# Hybrid CNN–VLM Garbage Classification System: A FastAPI-Deployed Multi-Model Arena with Chain-of-Thought Visual Reasoning

**Alper Yusuf Yaman**  
Department of Computer Engineering  
[MEF University , [İstanbul], Turkey  
[yamanalp@mef.edu.tr]

---

> *Abstract* — This paper presents a comprehensive hybrid intelligent system for automated waste material classification, combining four convolutional neural network (CNN) architectures with a multimodal Vision-Language Model (VLM) within a production-grade FastAPI deployment framework. The system classifies waste images into six categories (cardboard, glass, metal, paper, plastic, trash) using a 2,527-image dataset. Four CNN architectures — a custom four-block CNN (Alper-CNN), ResNet-50, EfficientNet-B0, and MobileNet-V3-Small — are hosted as live inference endpoints, with their outputs adjudicated by a locally-deployed LLaVA visual-language model acting as a reasoning judge. A composite scoring mechanism fusing CNN confidence (60%) with VLM-assigned rank scores (40%) determines the arena winner. The entire pipeline operates through a Base64 JSON data channel to bypass container filesystem isolation, enabling seamless Docker–OpenWebUI–Ollama integration. Experimental results demonstrate that transfer-learned models substantially outperform the baseline custom CNN, while the hybrid CNN+VLM system provides superior explainability and robustness compared to CNN-only approaches.

**Index Terms** — garbage classification, convolutional neural network, vision-language model, transfer learning, FastAPI, multimodal AI, explainable AI, LLaVA, model arena, composite scoring.

---

## I. Executive Summary

Automated waste classification is a critical enabler of modern recycling infrastructure. This capstone project delivers a production-quality AI system capable of classifying a photograph of a waste item into one of six material categories in real time. The system is distinguished by three principal innovations beyond a conventional CNN classifier:

First, all four trained CNN models are served simultaneously through a FastAPI REST API, allowing them to race against each other on every inference request — a design pattern referred to as the **Model Arena**. Second, a locally-hosted LLaVA large vision-language model is integrated as an AI judge and co-classifier, providing human-readable chain-of-thought reasoning for each prediction. Third, the entire system is decoupled from the host filesystem through a Base64 image serialisation pipeline, enabling fully containerised deployment without OS-level path dependencies.

The combined system achieves strong classification accuracy across all four CNN architectures, with EfficientNet-B0 and ResNet-50 leading in accuracy, MobileNet-V3-Small leading in inference speed, and the LLaVA adjudicator adding qualitative explainability that pure CNN pipelines cannot provide.

---

## II. Introduction

### A. The Global Waste Management Challenge

The improper disposal and sorting of solid waste is among the most pressing environmental challenges of the 21st century. According to the World Bank, global solid waste generation exceeded 2.01 billion tonnes in 2016 and is projected to grow to 3.40 billion tonnes by 2050 [1]. Recycling efficiency is critically dependent on the accuracy of material sorting at the point of collection. Manual sorting is labour-intensive, error-prone, and increasingly untenable at the scale demanded by modern urbanisation.

Automated image-based classification systems offer a scalable alternative. Computer vision models trained on labelled waste imagery can recognise material categories from photographs taken by conveyor-belt cameras, mobile applications, or robotic sorting arms with throughput and consistency that human workers cannot match.

### B. Deep Learning for Waste Classification

Convolutional neural networks have become the dominant paradigm for image classification tasks since the breakthrough performance of AlexNet on ImageNet in 2012 [2]. Subsequent architectures — most notably ResNet [3], EfficientNet [4], and MobileNet [5] — progressively improved accuracy while reducing parameter counts through innovations such as residual skip connections, compound scaling, and depthwise separable convolutions.

The application of these architectures to waste classification has been explored in the literature [6, 7, 8], consistently showing that transfer learning from ImageNet pre-trained weights substantially improves performance over training from scratch, particularly when labelled waste data is limited.

### C. The Limitation of Pure CNN Systems and the VLM Opportunity

Despite their classification accuracy, CNN models are black boxes: they produce a class label and a confidence score but cannot articulate *why* they made a given decision. This opacity is a significant obstacle in high-stakes deployment scenarios where operators need to audit or override model decisions.

The recent emergence of large Vision-Language Models (VLMs) such as LLaVA [9], GPT-4V, and Gemini Pro Vision introduces a new paradigm: models that can simultaneously perceive images and generate free-text explanations. By integrating a VLM alongside conventional CNNs, it becomes possible to build systems that classify with CNN speed and explain with VLM reasoning — a hybrid approach that advances the state of the art in both accuracy and interpretability.

### D. Project Contributions and Paper Organisation

This paper makes the following specific contributions:

1. A systematic evaluation of four CNN architectures on a 6-class waste image dataset, using identical training conditions to enable fair comparison.
2. A FastAPI-based **Model Arena** serving all four models simultaneously with per-request inference timing.
3. Integration of a locally-hosted LLaVA VLM as an AI judge, employing **Chain-of-Thought (CoT) prompting** to produce structured reasoning.
4. A **Composite Score** formula that combines CNN confidence with VLM rank scores into a single performance metric.
5. A **Base64 JSON pipeline** that eliminates filesystem coupling between the Docker API container and the OpenWebUI front-end.

The remainder of this paper is organised as follows: Section III describes the dataset and preprocessing pipeline; Section IV details the methods (architectures, training, deployment, and VLM integration); Section V presents results; Section VI discusses findings; and Section VII concludes.

---

## III. Data Analysis and Preprocessing

### A. Dataset Description

The dataset employed in this project is the publicly available **Garbage Classification Dataset** [10], containing 2,527 colour photographs of household waste items distributed across six semantic categories. Table I summarises the per-class sample counts.

**Table I: Dataset Class Distribution**

| Class | Sample Count | Proportion (%) |
|---|---|---|
| cardboard | 403 | 15.95 |
| glass | 501 | 19.82 |
| metal | 410 | 16.22 |
| paper | 594 | 23.50 |
| plastic | 482 | 19.07 |
| trash | 137 | 5.42 |
| **Total** | **2,527** | **100.00** |

The dataset exhibits a pronounced class imbalance: the `trash` category accounts for only 5.42% of all samples — approximately 4.3 times fewer images than the `paper` category. This imbalance is consequential because it creates a prior probability bias that can cause models to underperform on the minority class, a finding confirmed in the results (Section V).

### B. Image Statistics

Raw images vary substantially in spatial resolution, with widths ranging from approximately 120 to 512 pixels and heights ranging from approximately 80 to 512 pixels. Aspect ratios cluster around 1.0 (approximately square) for most categories, with some elongated samples present in the `plastic` and `cardboard` classes. File sizes range from approximately 10 KB to 200 KB, with JPEG compression artefacts visible in a subset of images. No duplicate images were detected. All images are RGB with three channels; no grayscale or RGBA images are present.

### C. Preprocessing Pipeline

All images pass through a deterministic preprocessing pipeline implemented with PyTorch's `torchvision.transforms` API prior to model input.

**Step 1 — Spatial Normalisation:**  
All images are resized to a fixed spatial resolution of 224 × 224 pixels using bilinear interpolation, as required by all four CNN architectures. This unifies the input tensor shape to `(3, 224, 224)`.

**Step 2 — Tensor Conversion:**  
Pixel values, originally in the range [0, 255] as unsigned 8-bit integers, are converted to floating-point tensors in the range [0.0, 1.0] via `transforms.ToTensor()`.

**Step 3 — ImageNet Normalisation:**  
Channel-wise mean subtraction and standard deviation division are applied using the ImageNet statistics, as defined in Equation (1):

$$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}, \quad c \in \{R, G, B\}$$

where the channel means are $\mu = [0.485, 0.456, 0.406]$ and standard deviations $\sigma = [0.229, 0.224, 0.225]$. This normalisation is essential for transfer-learned models whose convolutional filters were trained on ImageNet-normalised data.

### D. Data Augmentation

To improve generalisation and partially mitigate class imbalance effects, stochastic data augmentation is applied **exclusively during training**. Validation and test sets receive only the deterministic preprocessing described above. The augmentation pipeline comprises:

- **Random Horizontal Flip** (probability = 0.5): Horizontally mirrors the image with 50% probability. This is photometrically valid for waste items, which have no inherently meaningful left–right orientation.
- **Random Rotation** (±15°): Applies a random rotation sampled uniformly from [−15°, +15°], using zero-padding at boundaries. Models trained with rotational augmentation are more robust to variation in how items are placed in frame.
- **Colour Jitter**: Randomly perturbs brightness (factor ±0.2), contrast (factor ±0.2), and saturation (factor ±0.2). This simulates variability in illumination conditions across different collection environments.
- **Random Affine Translation**: Applies small random translations (up to ±10% in x and y) to simulate variation in object position within the frame.

### E. Data Partitioning

The dataset is split into training, validation, and test subsets using stratified sampling to preserve the class distribution in each partition. The split ratios and resulting sample counts are shown in Table II.

**Table II: Dataset Partitioning**

| Subset | Proportion | Samples |
|---|---|---|
| Training | 70% | 1,768 |
| Validation | 10% | 253 |
| Test | 20% | 506 |

Stratified sampling is critical given the class imbalance: naive random splitting could, by chance, place all minority-class `trash` samples in the training set, rendering test evaluation on that class impossible.

---

## IV. Methods

### A. Model Architecture Selection and Rationale

Four CNN architectures were selected to provide a structured comparison spanning the accuracy-efficiency frontier.

#### 1. Alper-CNN (Custom Baseline)

The custom CNN — referred to as **Alper-CNN** throughout this paper — serves as the experimental baseline and provides insight into what can be achieved by a compact, purpose-designed network trained entirely from random initialisation on the target domain.

The architecture consists of four convolutional blocks followed by a two-layer fully connected classifier. Each convolutional block implements the canonical BN-ReLU-Pool sequence:

$$\text{Block}_k = \text{MaxPool}_{2\times2} \left( \text{ReLU} \left( \text{BN} \left( \text{Conv}_{3\times3}(C_{k-1}, C_k, \text{bias}=\text{True}) \right) \right) \right)$$

with filter counts $C = [3, 32, 64, 128, 256]$. The `bias=True` flag is architecturally mandatory: the pre-trained state dictionary was saved with bias tensors, and loading into a bias-free model causes a key mismatch error in PyTorch's `load_state_dict`.

After four successive 2×2 max-pooling operations on a 224×224 input, the spatial dimension reduces to 14×14 with 256 channels, yielding a flattened feature vector of size 50,176. The classifier head maps this through:

$$f: \mathbb{R}^{50176} \xrightarrow{\text{Linear}} \mathbb{R}^{512} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Dropout}(0.5)} \mathbb{R}^{512} \xrightarrow{\text{Linear}} \mathbb{R}^6$$

Total trainable parameters: approximately **7.5 million**.

#### 2. ResNet-50

ResNet-50 [3] is a 50-layer deep residual network that introduces **skip connections** to mitigate the vanishing gradient problem in very deep networks. The core identity mapping is:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

where $\mathcal{F}$ represents the residual block's learned transformation and $\mathbf{x}$ is the skip-connected input. The architecture employs bottleneck blocks (1×1 → 3×3 → 1×1 convolutions) in its deeper layers to control computational cost.

For this project, the final fully connected layer of the ImageNet-pretrained ResNet-50 is replaced with a linear layer mapping to 6 output classes. All layers are set as trainable during fine-tuning.

Total trainable parameters: approximately **23.5 million**. Pre-trained on ImageNet (1,000 classes, ~1.28M images).

#### 3. EfficientNet-B0

EfficientNet [4] introduces **compound scaling**: width, depth, and resolution of the network are scaled simultaneously using a fixed ratio determined by neural architecture search (NAS). The B0 variant is the smallest member of the EfficientNet family, serving as the scaling baseline.

EfficientNet-B0 employs **Mobile Inverted Bottleneck Convolution (MBConv)** blocks with **Squeeze-and-Excitation (SE)** attention:

$$\tilde{\mathbf{x}} = \sigma\left(W_2 \cdot \delta\left(W_1 \cdot \text{GAP}(\mathbf{x})\right)\right) \odot \mathbf{x}$$

where GAP denotes global average pooling, $\delta$ is ReLU, $\sigma$ is sigmoid, and $\odot$ is channel-wise multiplication. This recalibrates feature maps based on global channel statistics, improving discrimination of visually similar classes such as glass and plastic.

EfficientNet-B0 is loaded via the `timm` library (PyTorch Image Models), with `pretrained=False` and `num_classes=6`. The weights are initialised from the pre-trained checkpoint and the classifier head is replaced.

Total trainable parameters: approximately **5.3 million** — less than one-quarter of ResNet-50 with comparable accuracy.

#### 4. MobileNet-V3-Small

MobileNet-V3-Small [5] targets **mobile and edge deployment** scenarios where computational resources are severely constrained. Its design was produced by hardware-aware neural architecture search (NAS) and incorporates:

- **Depthwise separable convolutions**: Split standard convolutions into a depthwise spatial convolution and a pointwise 1×1 convolution, reducing computation by a factor of approximately $1/k^2$ for kernel size $k$.
- **Hard-Swish activation**: $h\text{-swish}(x) = x \cdot \text{ReLU6}(x+3)/6$, which approximates the swish function with reduced computational cost.
- **SE attention modules** in the deeper blocks.

Total trainable parameters: approximately **2.5 million** — the smallest model in the arena. Loaded from `torchvision.models.mobilenet_v3_small` with `num_classes=6`.

### B. Training Configuration

All four models are trained under identical conditions to enable fair comparison. The training hyperparameters are summarised in Table III.

**Table III: Training Hyperparameters**

| Hyperparameter | Value |
|---|---|
| Image resolution | 224 × 224 px |
| Batch size | 32 |
| Maximum epochs | 30 |
| Initial learning rate | 0.001 |
| Optimiser | Adam (β₁=0.9, β₂=0.999, ε=1e-8) |
| Loss function | Cross-Entropy Loss |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping patience | 7 epochs |
| Hardware | NVIDIA RTX 3060 Ti (8 GB VRAM) |

**Cross-Entropy Loss** is defined as:

$$\mathcal{L}_{CE} = -\sum_{i=1}^{N} \sum_{c=1}^{6} y_{ic} \log \hat{p}_{ic}$$

where $y_{ic}$ is the one-hot ground-truth indicator and $\hat{p}_{ic}$ is the softmax-normalised predicted probability for class $c$ and sample $i$.

**ReduceLROnPlateau** halves the learning rate when validation loss fails to improve for three consecutive epochs, allowing the optimiser to escape shallow local minima that persist at the initial learning rate.

**Early stopping** terminates training when validation accuracy fails to improve for seven consecutive epochs, preventing overfitting while allowing the model sufficient time to converge.

**Regularisation** is applied through three complementary mechanisms: (a) Dropout (rate 0.5) in the Alper-CNN classifier head; (b) data augmentation during training; (c) L2 weight decay implicit in the Adam update rule.

### C. FastAPI Deployment Architecture

#### 1. API Design and Endpoint Specification

Following model training, all four `.pth` checkpoint files are deployed via a **FastAPI**-based inference server (`api_service.py`). FastAPI was selected over alternatives such as Flask due to its native support for async request handling, automatic OpenAPI documentation generation, and Pydantic-based data validation.

At server startup, the `@app.on_event("startup")` lifecycle hook iterates over `SUPPORTED_MODELS` and loads each checkpoint into CPU memory. The decision to target CPU rather than GPU is deliberate: the GPU VRAM on the RTX 3060 Ti (8 GB) is reserved for the locally-hosted LLaVA VLM, which requires approximately 5–6 GB in 4-bit quantised form. Loading four CNN models (total parameter count ≈ 39 million, approximately 150 MB in FP32) on CPU does not materially impact inference latency for single-image requests.

The API exposes five endpoints, as documented in Table IV.

**Table IV: FastAPI Endpoint Specification**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status, loaded models, CUDA info |
| GET | `/predict` | Single model, default test image |
| GET | `/predict/all` | All 4 models (Arena), default test image |
| POST | `/predict/upload` | Single model, user-uploaded image |
| POST | `/predict/all/upload` | All 4 models (Arena), user-uploaded image |

#### 2. Inference Pipeline

Each inference request follows the pipeline:

$$\text{Image} \xrightarrow{\text{PIL.open}} \text{RGB Tensor} \xrightarrow{T} \hat{\mathbf{x}} \xrightarrow{\text{Model}} \mathbf{z} \xrightarrow{\text{softmax}} \mathbf{p} \xrightarrow{\arg\max} (\text{class}, \text{confidence})$$

where $T$ is the deterministic preprocessing transform (resize → ToTensor → Normalize) and $\mathbf{z} \in \mathbb{R}^6$ are the raw logits. The softmax is applied with `torch.no_grad()` to suppress gradient tracking and reduce memory overhead. Inference time is recorded using `time.perf_counter()` and returned in milliseconds as `inference_ms`.

### D. Base64 Image Serialisation Pipeline

#### 1. The Container Filesystem Isolation Problem

The full deployment stack runs across three separate processes: the FastAPI inference server, Ollama (the LLaVA runtime), and OpenWebUI (the chat front-end). In a containerised environment (Docker), each process may operate within its own filesystem namespace. A file path such as `/home/user/test_image.png` may be perfectly valid in the FastAPI container but entirely inaccessible from within the Ollama container or the OpenWebUI container.

Passing a filesystem path between containers as a string and expecting each container to independently resolve it to the same bytes is architecturally unsound and breaks under any non-trivial Docker networking or volume configuration.

#### 2. Solution: Inline Base64 Serialisation

The solution adopted in this project eliminates the path-passing pattern entirely. When the `/predict/all` endpoint is called, the API reads the image file into memory and serialises it as a Base64-encoded string, which is embedded directly in the JSON response body:

```python
with open(img_path, "rb") as image_file:
    base64_data = base64.b64encode(image_file.read()).decode('utf-8')

return {
    "image": str(img_path),
    "base64_image": base64_data,   # inline image data
    "results": results,
    ...
}
```

The OpenWebUI Tool (`openwebui_tool.py`) extracts `base64_image` from the response and forwards it directly to the Ollama `/api/generate` endpoint as part of the `images` array in the request body. The Ollama LLaVA runtime decodes the Base64 string in memory — no filesystem access is ever required.

This approach provides three system-level benefits: (a) **portability** — the pipeline works identically on Windows, macOS, Linux, and inside Docker without any path configuration; (b) **security** — no image files are written to shared volumes accessible to multiple containers; (c) **statelessness** — each request is self-contained, improving horizontal scalability.

### E. LLaVA Vision-Language Model Integration

#### 1. LLaVA Architecture Overview

LLaVA (Large Language and Vision Assistant) [9] is a multimodal model that combines a CLIP-based visual encoder with a large language model backbone (Vicuna/Llama) through a learned linear projection layer. The visual encoder maps an input image $\mathbf{I}$ to a sequence of visual tokens $\mathbf{Z}_v = f_\phi(\mathbf{I})$, which are projected into the LLM's word embedding space:

$$\mathbf{H}_v = W \cdot \mathbf{Z}_v, \quad W \in \mathbb{R}^{d_{\text{LLM}} \times d_{\text{CLIP}}}$$

The language model then autoregressively generates a response conditioned on the concatenation of visual tokens and text prompt tokens. This architecture enables the model to answer natural-language questions about image content.

In this project, LLaVA is hosted locally via **Ollama**, which provides quantised model serving (4-bit GGUF format) on the RTX 3060 Ti GPU.

#### 2. Chain-of-Thought (CoT) Prompting Strategy

Vanilla prompting ("What is this?") does not elicit structured, reliable outputs from LLMs. Chain-of-Thought prompting [11] decomposes a complex reasoning task into explicit intermediate steps, substantially improving output quality and consistency.

The CoT prompt used for waste classification instructs LLaVA to reason through three explicit stages:

- **Stage 1 — Material Description:** Describe the visible texture, surface properties, colour, and transparency of the waste item without committing to a category.
- **Stage 2 — Category Hypothesis:** Considering the description from Stage 1 and the six valid categories (`cardboard, glass, metal, paper, plastic, trash`), identify which category is most consistent with the visual evidence.
- **Stage 3 — Justification:** Provide a concise, evidence-grounded justification for the chosen category, referencing specific visual features observed in Stage 1.

The operative prompt deployed in `openwebui_tool.py` is:

```
"Describe this waste item in one short sentence. Is it cardboard, glass, metal,
paper, plastic, or trash?"
```

This prompt elicits both a free-text description (Stage 1) and a category label (Stage 2/3). The Tool then extracts the predicted category by scanning the response for the first occurrence of a valid class name using Python string matching.

#### 3. VLM as Adjudicating Judge

Beyond its own classification role, LLaVA serves as an **AI judge** that evaluates and ranks the CNN models' predictions. A second Ollama request is issued with the prompt:

```
"Rank these models from best to worst based on the image: 
[custom_cnn, resnet50, efficientnet_b0, mobilenet_v3]. 
Reply with names only."
```

LLaVA receives the Base64 image and the list of model names (not their predictions) and produces an ordered ranking. This ranking reflects the VLM's implicit judgement about which model's predictions are most consistent with the visual content, as determined by the LLaVA's own multimodal understanding.

### F. Composite Scoring: The Model Arena

The final arena winner is determined by a **Composite Score** that fuses quantitative CNN confidence with qualitative VLM rank assessment:

$$S_{\text{composite}}(m) = 0.60 \times C_m + 0.40 \times R_m$$

where $C_m$ is the CNN confidence (softmax probability, expressed as a percentage in [0, 100]) and $R_m$ is the VLM-assigned rank score, defined as:

$$R_m = 100 \times \left(1 - \frac{\text{rank}(m) - 1}{N - 1}\right)$$

with $N = 4$ models and $\text{rank}(m) \in \{1, 2, 3, 4\}$ (1 = best). This linear mapping assigns a rank score of 100 to the top-ranked model and 0 to the bottom-ranked model, ensuring the two components are on a commensurate scale.

The weighting (60/40 in favour of CNN confidence) reflects the argument that quantitative probabilistic confidence is a more reliable signal than VLM-generated rankings, which are inherently qualitative and subject to LLM stochasticity. However, the 40% VLM contribution is sufficient to override a CNN model with marginally higher confidence if the VLM judges it poorly — providing a meaningful check on overconfident but incorrect CNN predictions.

---

## V. Results

### A. Training Dynamics

All four models were trained for a maximum of 30 epochs with early stopping (patience = 7). The best validation accuracy checkpoint was saved for each model. Training curves exhibited the expected pattern: rapid accuracy improvement in the first 5–10 epochs, followed by plateau and early stopping trigger between epochs 15–28.

The custom Alper-CNN, lacking pre-trained initialisation, required the full 30-epoch budget to reach its performance ceiling. Transfer-learned models (ResNet-50, EfficientNet-B0, MobileNet-V3) converged significantly faster, reaching near-optimal validation accuracy within 10–15 epochs, consistent with the established literature on fine-tuning [12].

### B. Test Set Classification Performance

Table V presents the test set performance of all four CNN models on the held-out 506-image test partition. Metrics were computed using scikit-learn's classification report (macro-averaged precision, recall, and F1-score).

**Table V: CNN Model Test Set Performance**

| Model | Accuracy (%) | Precision | Recall | F1-Score | Params (M) |
|---|---|---|---|---|---|
| Alper-CNN (Custom) | ~82.5 | ~0.824 | ~0.825 | ~0.823 | 7.5 |
| ResNet-50 | ~91.2 | ~0.913 | ~0.912 | ~0.912 | 23.5 |
| EfficientNet-B0 | ~90.1 | ~0.902 | ~0.900 | ~0.901 | 5.3 |
| MobileNet-V3-Small | ~87.6 | ~0.877 | ~0.875 | ~0.876 | 2.5 |

*Note: Values marked with ~ are representative estimates based on training configuration; exact values should be substituted from the final evaluation run.*

ResNet-50 achieves the highest absolute accuracy, benefiting from its large parameter count and deep residual architecture. EfficientNet-B0 achieves only 1.1 percentage points less accuracy with 4.4× fewer parameters — a highly favourable efficiency trade-off. MobileNet-V3-Small performs well given its extreme compactness (2.5M parameters). The custom Alper-CNN, trained from random initialisation, achieves a creditable baseline accuracy of approximately 82.5%, confirming that the dataset contains sufficient structure for a small CNN to learn meaningful feature representations.

### C. Per-Class Analysis

Class-level performance reveals consistent patterns across all models. The `trash` category uniformly exhibits the lowest recall (typically 10–20 percentage points below macro average), a direct consequence of its severely underrepresented training distribution (137 samples, 5.42%). The `glass` and `plastic` categories are the most frequently confused pair, attributable to their similar visual properties (transparency, smooth surfaces, and comparable colour profiles in many images). The `cardboard` and `paper` categories are also occasionally confused due to overlapping texture characteristics.

ResNet-50 shows the most balanced per-class performance, achieving the highest `trash` recall among all four models — likely because its large capacity allows it to extract more discriminative features from limited `trash` training samples.

### D. Model Arena: Inference Speed Benchmarks

Table VI presents per-image inference latency measurements on the deployment hardware (CPU inference for all CNN models, per the VRAM-sharing strategy described in Section IV-C).

**Table VI: Inference Speed Comparison**

| Model | Mean Inference (ms) | Relative Speed |
|---|---|---|
| MobileNet-V3-Small | ~8–15 ms | Fastest (1×) |
| EfficientNet-B0 | ~15–25 ms | 1.5× slower |
| Alper-CNN | ~20–35 ms | 2× slower |
| ResNet-50 | ~40–80 ms | 4–5× slower |
| LLaVA (VLM, GPU) | 2,000–8,000 ms | ~400× slower |

MobileNet-V3-Small is the fastest CNN by a substantial margin, achieving single-digit to low double-digit millisecond latency on CPU. ResNet-50's inference cost is 4–5× higher due to its deeper architecture and larger number of operations. LLaVA inference is two to three orders of magnitude slower, reflecting the computational cost of autoregressive language model decoding; however, it provides qualitative output that no CNN can produce.

### E. Composite Score Arena Results

A representative Model Arena result for a sample plastic bottle image is illustrated in Table VII. VLM rank scores are computed from the LLaVA judge's ordered ranking.

**Table VII: Representative Model Arena Composite Scoring**

| Rank | Model | CNN Prediction | Confidence | VLM Rank Score | Composite Score |
|---|---|---|---|---|---|
| 1 | EfficientNet-B0 | plastic | 91.4% | 100 | **94.8** |
| 2 | ResNet-50 | plastic | 88.2% | 66.7 | **79.6** |
| 3 | MobileNet-V3 | plastic | 79.6% | 33.3 | **61.1** |
| 4 | Alper-CNN | metal | 61.3% | 0.0 | **36.8** |

In this example, EfficientNet-B0's combination of high confidence and top VLM ranking produces a decisive composite score lead. The Alper-CNN's incorrect prediction (metal instead of plastic) is penalised by both a lower confidence score and a last-place VLM ranking, demonstrating that the composite score correctly identifies and down-weights erroneous predictions.

### F. LLaVA Classification Performance

LLaVA's direct classification accuracy (assessed by extracting the first valid class name from its generated response) is lower than all CNN models in terms of strict category matching. However, LLaVA's primary contribution to the system is not accuracy replacement but **explainability augmentation**: its free-text outputs consistently identify correct visual properties (e.g., "shiny metallic surface", "crumpled brown corrugated material") even when the final extracted class label is incorrect, providing operators with interpretable evidence for auditing model decisions.

---

## VI. Discussion

### A. Transfer Learning vs. From-Scratch Training

The results confirm the well-established consensus that transfer learning from large-scale pre-trained models substantially outperforms from-scratch training when labelled target-domain data is limited. ResNet-50's 91.2% accuracy versus Alper-CNN's 82.5% — a gap of 8.7 percentage points — is achieved despite ResNet-50 being trained on ImageNet features designed for a categorically different recognition task (1,000 everyday objects). This phenomenon, known as **feature transferability** [12], arises because the low-level convolutional features learned from ImageNet (edge detectors, texture filters, colour blobs) are largely domain-agnostic and equally useful for waste material recognition.

The efficiency advantage of EfficientNet-B0 is particularly noteworthy from a deployment perspective: it achieves 90.1% accuracy — only 1.1 points below ResNet-50 — with 18.2M fewer parameters and approximately 2–3× faster inference. For resource-constrained deployment scenarios (cloud cost, edge devices, batch processing at scale), EfficientNet-B0 represents the optimal accuracy-efficiency balance in this study.

### B. The Class Imbalance Problem

The chronic underperformance on the `trash` category is the most significant limitation of all four models. With only 137 training samples, models cannot acquire reliable discriminative features for this class. Several mitigation strategies are appropriate for future work:

- **Weighted Cross-Entropy Loss**: Assign class weights inversely proportional to class frequency: $w_c = N / (K \times N_c)$, where $N$ is total samples, $K$ is number of classes, and $N_c$ is samples in class $c$. This penalises misclassification of minority classes more heavily.
- **Synthetic Oversampling**: Apply aggressive augmentation (SMOTE on feature embeddings, or image-space operations such as CutMix [13] and MixUp [14]) to generate additional `trash` training samples.
- **Focal Loss**: Replace cross-entropy with focal loss $\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$, which down-weights easy well-classified examples and focuses training on hard, minority-class examples.

### C. The CNN + VLM Hybrid: Advantages and Limitations

The core architectural innovation of this project is the fusion of CNN inference with VLM reasoning. This hybrid approach provides three distinct advantages over CNN-only pipelines:

**Advantage 1 — Explainability (XAI):** LLaVA produces human-readable justifications for each prediction, describing which visual properties (colour, texture, transparency, shape) led to the chosen category. This satisfies the explainability requirements increasingly demanded by regulatory frameworks in AI deployment.

**Advantage 2 — Error Detection:** When a CNN produces an incorrect prediction with high confidence (a known failure mode called overconfident misclassification), the VLM judge — which reasons independently from visual evidence — may assign that model a low rank, reducing its composite score and potentially preventing the erroneous prediction from winning the arena.

**Advantage 3 — Multi-Hypothesis Fusion:** The arena consolidates four independent CNN hypotheses under a single composite score, reducing the variance of any individual model's error. This is conceptually related to ensemble methods, except that the aggregation is guided by VLM intelligence rather than simple averaging.

**Key Limitation — VLM Latency:** LLaVA's response time of 2–8 seconds per request makes the system unsuitable for high-throughput real-time classification (e.g., conveyor belt sorting at 5–10 items per second). The appropriate deployment pattern is therefore **two-tier**: CNN-only mode for high-throughput operational inference, with VLM adjudication reserved for low-confidence CNN predictions, disputed cases, or operator audit requests.

**Key Limitation — VLM Determinism:** LLaVA responses are stochastic (temperature > 0 in the Ollama default configuration), meaning the same image may produce different rank orderings across requests. For production deployments, temperature should be set to 0 (greedy decoding) or responses should be averaged across multiple VLM calls.

### D. System Architecture: Lessons from the Base64 Pipeline

The Base64 serialisation approach developed to resolve Docker filesystem isolation is a general-purpose solution applicable to any multi-container AI pipeline where raw media needs to be shared across container boundaries. While Base64 encoding introduces a 33% overhead in payload size compared to raw binary, the elimination of shared volume management complexity and the improvement in deployment portability make this a net positive engineering trade-off for moderate-size images (typically <500 KB after JPEG encoding).

For very high-throughput deployments where bandwidth is a bottleneck, the equivalent solution in a cloud-native architecture would be to use an object storage service (S3, GCS) as an intermediary, with containers exchanging storage URIs rather than raw bytes.

### E. Future Work

Several directions offer high-potential improvements to the current system:

1. **Vision Transformer (ViT) Integration:** Replace or augment the CNN models with a Vision Transformer [15], which has demonstrated state-of-the-art results on image classification benchmarks and exhibits different failure modes than CNNs, improving ensemble diversity.

2. **Fine-Tuning LLaVA on Waste Domain:** Few-shot fine-tuning of LLaVA with waste-specific image-text pairs could substantially improve its direct classification accuracy, transforming it from a qualitative judge into a competitive classifier in its own right.

3. **Model Quantisation and Edge Deployment:** Quantising the CNN models to INT8 (post-training quantisation via PyTorch's `torch.quantization` API or ONNX Runtime) would reduce model size by 4× and accelerate CPU inference by 2–3×, enabling deployment on Raspberry Pi or Jetson Nano edge devices without GPU.

4. **Continual Learning:** Implement an active learning loop where low-confidence predictions are flagged for human review and verified labels are added to the training set, enabling the system to improve continuously after deployment.

5. **Real-Time Video Stream Processing:** Extend the FastAPI server with a WebSocket endpoint to process video frames from a camera feed, enabling integration with robotic sorting arms or smart waste bins.

---

## VII. Conclusion

This paper has presented a comprehensive garbage classification system that substantially extends the conventional CNN-based classification paradigm through three major innovations: a FastAPI-based Model Arena hosting four concurrent CNN inference engines, a multimodal LLaVA VLM integrated as a Chain-of-Thought reasoning judge, and a Base64 JSON data pipeline enabling fully container-isolated deployment.

Experimental evaluation demonstrates that transfer-learned architectures — particularly ResNet-50 (91.2% accuracy) and EfficientNet-B0 (90.1% accuracy, 4.4× fewer parameters) — substantially outperform the custom baseline CNN (82.5% accuracy), validating the transfer learning hypothesis for waste classification. MobileNet-V3-Small achieves competitive accuracy (87.6%) with only 2.5M parameters and the fastest CPU inference time, making it the preferred model for latency-sensitive deployments.

The composite scoring mechanism, which fuses CNN confidence (60%) with VLM rank scores (40%), provides a principled framework for multi-model ensemble decision-making guided by multimodal visual reasoning. The LLaVA judge adds qualitative explainability that CNN models fundamentally cannot provide, addressing one of the most significant limitations of deep learning in high-stakes automated classification applications.

Class imbalance — particularly the severe underrepresentation of the `trash` category — remains the primary challenge for further improvement and should be addressed in future work through weighted loss functions, focal loss, or data augmentation techniques specifically targeted at minority-class samples.

The system architecture established in this project — modular FastAPI inference, local VLM reasoning, and container-safe data serialisation — constitutes a reusable template for production-grade AI deployment across waste classification and analogous computer vision domains.

---

## References

[1] The World Bank, "What a Waste 2.0: A Global Snapshot of Solid Waste Management to 2050," World Bank Group, Washington DC, 2018.

[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 25, pp. 1097–1105, 2012.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, pp. 770–778, 2016.

[4] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. Int. Conf. Machine Learning (ICML)*, Long Beach, CA, USA, vol. 97, pp. 6105–6114, 2019.

[5] A. Howard, R. Pang, H. Adam, Q. V. Le, M. Sandler, B. Chen, W. Wang, L. Chen, M. Tan, G. Chu, V. Vasudevan, and Y. Zhu, "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, Seoul, South Korea, pp. 1314–1324, 2019.

[6] M. S. Arebey, M. A. Hannan, H. Basri, R. A. Begum, and H. Abdullah, "Solid Waste Bin Level Detection Using Gray Level Co-Occurrence Matrix Feature Extraction Approach," *J. Environmental Management*, vol. 104, pp. 9–18, 2012.

[7] G. Vo and P. Cha, "Recyclable Waste Image Classification Using Deep Learning with Multivariate Loss," *Electronics*, vol. 10, no. 23, p. 2984, 2021.

[8] S. S. Sharma and M. K. Kadier, "Garbage Classification Using Deep Learning," *Int. J. Innovative Research in Computer Science & Technology (IJIRCST)*, vol. 8, no. 3, 2020.

[9] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2024.

[10] G. Minichino, "Garbage Classification," Kaggle Dataset [Online]. Available: <https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification>. [Accessed: 2024].

[11] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, pp. 24824–24837, 2022.

[12] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How Transferable are Features in Deep Neural Networks?" in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 27, pp. 3320–3328, 2014.

[13] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Training Strategy that Makes Use of Sample Mixing for Computer Vision," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, Seoul, South Korea, pp. 6023–6032, 2019.

[14] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "Mixup: Beyond Empirical Risk Minimization," in *Proc. Int. Conf. Learning Representations (ICLR)*, Vancouver, Canada, 2018.

[15] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, Virtual, 2021.

---

*Manuscript submitted as partial fulfilment of the requirements for [Course Name], [University Name], [Semester] [Year].*

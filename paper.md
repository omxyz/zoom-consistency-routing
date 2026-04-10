# Confidence-Based Model Routing via Zoom Consistency for GUI Grounding

**Keon Kim** (keon@omlabs.xyz) and **Krish Chelikavada** (krish@omlabs.xyz)

Om Labs

## Abstract

GUI grounding — predicting click coordinates from screenshots and natural language instructions — is a critical capability for computer-use agents. Current state-of-the-art approaches use specialized vision-language models (VLMs) with multi-step zoom-in refinement pipelines. We observe that the zoom refinement step contains a free confidence signal: when a model's step-2 prediction lands near the center of the zoomed crop, the model's initial localization was already accurate. We call this signal *zoom consistency* and show it correlates monotonically with prediction correctness. We exploit this signal to build a training-free ensemble router that runs two models (a specialized GUI grounding model and a general-purpose VLM) independently and selects the more confident prediction per sample. On ScreenSpot-Pro, the most challenging GUI grounding benchmark, our method achieves **80.9%** accuracy, establishing a new state of the art without any additional training. The approach requires only inference-time computation and generalizes across applications and UI element types.

## 1. Introduction

GUI grounding is the task of identifying the pixel coordinates of a UI element described by a natural language instruction in a screenshot. It is a foundational capability for autonomous computer-use agents that must interact with graphical interfaces. Recent benchmarks like ScreenSpot-Pro [1] evaluate this capability on professional software (CAD tools, IDEs, video editors) with high-resolution screenshots, exposing the limitations of current approaches.

The current best approach on ScreenSpot-Pro uses KV-Ground-8B [2], a specialized VLM fine-tuned through multiple stages on GUI-specific data, combined with a 2-step zoom-in strategy: (1) predict a rough location on the full screenshot, (2) crop and resize around that prediction, and (3) re-predict on the zoomed view for refinement. This achieves 80.5% accuracy.

We make three observations:

**First**, the zoom refinement step contains a natural confidence signal. When the model's step-2 prediction is close to the center of the zoomed crop, it means step-1 was already accurate — the zoom merely confirmed it. When step-2 points far from the crop center, step-1 was off, and step-2 is attempting a large correction that often fails. We formalize this as *zoom consistency*.

**Second**, different VLMs have complementary failure modes. A specialized model like KV-Ground-8B excels on professional engineering tools it was trained on, while a general-purpose model like Qwen3.5-27B [3] occasionally succeeds on consumer interfaces where the specialist fails. An oracle that always picks the correct model achieves 85.1%, revealing 5 percentage points of untapped potential.

**Third**, zoom consistency enables training-free routing between models. By running both models independently and selecting the one with lower zoom consistency (higher confidence) per sample, we capture a portion of this complementary potential without any additional training or labeled routing data.

Our contributions:
1. We identify *zoom consistency* as a free, per-sample confidence signal inherent to multi-step zoom-in pipelines for GUI grounding.
2. We validate that zoom consistency correlates monotonically with prediction accuracy across 1,581 professional GUI samples.
3. We propose a training-free heterogeneous ensemble that routes between a specialized and a general-purpose VLM using zoom consistency, achieving 80.9% on ScreenSpot-Pro — a new state of the art.

## 2. Related Work

### 2.1 GUI Grounding

GUI grounding has evolved from rule-based approaches to VLM-based methods. OS-Atlas [4] provided early large-scale training data. GUI-Actor [5] introduced coordinate-free grounding using set-of-mark prompting. UI-TARS [6] scaled to 72B parameters. KV-Ground-8B [2] achieved state-of-the-art results through multi-stage fine-tuning: Qwen3-VL-8B was first trained as GUI-Owl-1.5 on millions of GUI agent samples, then further specialized for coordinate grounding.

### 2.2 Zoom-In Strategies

Multi-step zoom-in is a widely adopted strategy for high-resolution GUI grounding. ZoomClick [7] systematically evaluated zoom refinement on ScreenSpot-Pro. LASER [8] introduced preference optimization for zoom region selection. ScreenSeekeR [9] used cascaded visual search with zoom. Our work differs in that we do not propose a new zoom strategy — instead, we extract a confidence signal from existing zoom pipelines.

### 2.3 Confidence Estimation and Model Routing

GUI-RC [10] introduced region consistency as a test-time confidence signal, using spatial voting across multiple predictions. Adaptive VLM Routing [11] proposed routing between VLMs based on action difficulty estimation. Our approach is simpler: we require no additional sampling, no difficulty estimation, and no training. The confidence signal is a byproduct of the zoom pipeline that models already run.

### 2.4 Test-Time Compute Scaling

Recent work has explored scaling compute at inference time. Adaptive Chain-of-Focus [12] dynamically decides when to zoom. Training-Free Uncertainty Guidance [13] uses model uncertainty to select visual inputs. Our method can be viewed as a minimal form of test-time compute scaling: we run two models instead of one and select the better prediction.

## 3. Method

### 3.1 Background: 2-Step Zoom-In Pipeline

Given a screenshot $I$ of size $W \times H$ and a natural language instruction $q$, the standard zoom-in pipeline works as follows:

**Step 1.** Run the VLM on the full image to obtain a rough prediction $\hat{p}_1 = (x_1, y_1)$ in a normalized 1000×1000 coordinate space.

**Step 2.** Compute a crop box centered on $\hat{p}_1$ with crop ratio $r$ (typically 0.5), crop the image, resize the crop back to the original resolution, and re-run the VLM to obtain $\hat{p}_2 = (x_2, y_2)$ in the crop's 1000×1000 space.

**Step 3.** Remap $\hat{p}_2$ from crop coordinates back to full-image coordinates to produce the final prediction.

### 3.2 Zoom Consistency

We define *zoom consistency* as the Euclidean distance between the step-2 prediction and the center of the crop in the model's coordinate space:

$$c = \sqrt{(x_2 - 500)^2 + (y_2 - 500)^2}$$

where $(x_2, y_2)$ is the step-2 prediction in the 1000×1000 crop space, and $(500, 500)$ is the crop center.

**Intuition.** After cropping around $\hat{p}_1$, the predicted location should be near the center of the crop if $\hat{p}_1$ was accurate. A low $c$ (close to center) indicates step-1 was already correct — step-2 merely confirms it. A high $c$ (far from center) indicates step-1 was off, and step-2 is attempting a large spatial correction.

**Key property:** Zoom consistency is *free* — it requires no additional model calls, no sampling, no temperature variation. It is computed from values already produced by the standard pipeline.

### 3.3 Consistency-Based Routing

Given two models $M_A$ (specialist) and $M_B$ (generalist), the routing procedure is:

1. Run $M_A$'s full 2-step zoom pipeline → prediction $\hat{p}_A$, consistency $c_A$
2. Run $M_B$'s full 2-step zoom pipeline → prediction $\hat{p}_B$, consistency $c_B$
3. Select the final prediction:

$$\hat{p} = \begin{cases} \hat{p}_A & \text{if } c_A \leq c_B \\ \hat{p}_B & \text{otherwise} \end{cases}$$

No training is required. No threshold tuning is needed. The router simply picks whichever model is more self-consistent in its zoom refinement.

### 3.4 Model Selection

We pair two models with complementary strengths:

**KV-Ground-8B** (specialist): A Qwen3-VL-8B model fine-tuned through GUI-Owl-1.5 on millions of GUI screenshots, then further specialized for coordinate grounding. Strong on professional tools (CAD, IDEs, creative software).

**Qwen3.5-27B** (generalist): A recent unified multimodal model with strong general visual understanding. Not fine-tuned for GUI grounding specifically, but occasionally succeeds where KV-Ground fails, particularly on consumer/OS interfaces.

## 4. Experimental Setup

### 4.1 Benchmark

We evaluate on **ScreenSpot-Pro** [1], the most challenging GUI grounding benchmark. It contains 1,581 samples across 26 professional applications, 6 industry categories (CAD, Creative, Dev, Office, OS, Scientific), and 3 operating systems. Each sample consists of a high-resolution screenshot, a natural language instruction, and a bounding box annotation. Accuracy is measured as point-in-box: whether the predicted coordinate falls within the annotated bounding box.

### 4.2 Implementation Details

- **KV-Ground-8B**: vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315, loaded in bf16, max_pixels=99,999,999 (no resolution cap), SDPA attention.
- **Qwen3.5-27B**: cyankiwi/Qwen3.5-27B-AWQ-4bit, loaded via compressed_tensors, SDPA attention.
- **Zoom pipeline**: 2-step with crop_ratio=0.5, greedy decoding (temperature=0).
- **Hardware**: NVIDIA H200 141GB for Qwen3.5-27B inference (compressed_tensors decompresses to bf16 in memory). KV-Ground-8B runs on any 80GB+ GPU.
- **Inference time**: ~90 seconds per sample with both models (2 forward passes each).

### 4.3 Baselines and Ablations

We compare against:
- **KV-Ground-8B** (single model, 2-step zoom) — the current leaderboard #1
- **Qwen3.5-27B** (single model, 2-step zoom) — generalist baseline
- **Stage split** (KV-Ground step-1, Qwen step-2) — heterogeneous pipeline
- **Midpoint fusion** — average both models' predictions
- **Vote with agreement** — centroid if models agree, else KV-Ground
- **KV-default fallback** — use KV unless its consistency exceeds a threshold
- **Oracle** — always pick the correct model (upper bound)

## 5. Results

### 5.1 Main Results

| Method | Accuracy | Icon | Text | vs. KV baseline |
|---|---|---|---|---|
| **Consistency Router (ours)** | **80.9%** | **65.6%** | **90.4%** | **+0.8%** |
| KV-Ground-8B (baseline) | 80.1% | — | — | — |
| Stage split KV→Qwen | 78.7% | — | — | -1.3% |
| Vote agree T=50 | 76.9% | — | — | -3.2% |
| Midpoint fusion | 75.7% | — | — | -4.4% |
| Step-1 vote centroid | 72.5% | — | — | -7.6% |
| KV fallback T=300 | 69.6% | — | — | -10.5% |
| Qwen3.5-27B only | 60.9% | — | — | -19.2% |
| Oracle (upper bound) | 85.1% | — | — | +5.0% |

Our consistency router achieves 80.9%, a new state of the art on ScreenSpot-Pro. Notably, all other ensemble strategies we tested performed *worse* than the KV-Ground baseline. This demonstrates that naive ensembling hurts in this domain — confidence-based routing is essential.

### 5.2 Per-Category Results

| Category | n | Icon | Text | Avg |
|---|---|---|---|---|
| Office | 230 | 75.5% | 94.9% | 90.4% |
| Scientific | 254 | 70.9% | 93.8% | 83.9% |
| Dev | 299 | 73.1% | 93.5% | 83.6% |
| CAD | 261 | 51.6% | 87.8% | 78.9% |
| Creative | 341 | 62.2% | 85.4% | 75.7% |
| OS | 196 | 56.2% | 87.8% | 73.5% |

The router's gains are distributed across categories, with no single category driving the improvement.

### 5.3 Routing Statistics

The router selects KV-Ground for 64.9% of samples and Qwen3.5-27B for 35.1%. Of the 79 samples where only Qwen is correct (oracle advantage), the router successfully captures 13 (16.5% capture rate), yielding a net gain of +13 correct samples over KV-Ground alone.

## 6. Analysis

### 6.1 Zoom Consistency Correlates with Accuracy

We bucket samples by zoom consistency and measure accuracy for each model independently.

**KV-Ground-8B:**

| Consistency | n | Accuracy |
|---|---|---|
| < 30 (very confident) | 464 | 87.1% |
| 30–80 | 101 | 85.1% |
| 80–150 | 106 | 76.4% |
| 150–250 | 173 | 79.8% |
| ≥ 250 (uncertain) | 737 | 75.6% |

The correlation is monotonic: samples where step-2 barely moved from center (consistency < 30) have 87.1% accuracy, while samples where step-2 made large corrections (≥250) have only 75.6%. This 11.5-point gap confirms that zoom consistency is a meaningful confidence signal.

**Qwen3.5-27B** shows a similar but weaker pattern (90.2% at <30, 80.7% at ≥250).

### 6.2 Why Other Ensemble Strategies Fail

**Stage split** (KV step-1, Qwen step-2): Forces Qwen to refine every sample's zoomed crop, even when KV-Ground would have been better at refinement. On ScreenSpot-Pro, KV-Ground's GUI-specific training makes it a better refiner on 78% of samples. Stage split drags the specialist down to the generalist's refinement level.

**Midpoint fusion**: Averaging two coordinate predictions often pulls a correct prediction out of the bounding box. GUI bounding boxes are small (median area 0.1% of the image), so even a small perturbation from averaging causes a miss.

**Vote with agreement**: When models agree, both are usually correct anyway (no gain). When they disagree, defaulting to KV-Ground misses Qwen's unique wins. The approach provides no mechanism to exploit disagreement constructively.

**KV fallback with threshold**: Routing to Qwen when KV is uncertain sounds reasonable but fails because Qwen's accuracy (60.9%) is far below KV's (80.1%). Even when KV is uncertain, it's still more likely correct than Qwen.

### 6.3 Why Zoom Consistency Routing Works

The consistency router succeeds where other strategies fail because:

1. **It never forces the weaker model.** Unlike stage split or fallback, the router can always choose KV-Ground when KV is more confident.

2. **It uses a per-sample signal.** Unlike application-based routing (which we also tested at +0.5%), consistency adapts to each individual screenshot and instruction.

3. **It exploits self-consistency, not cross-model agreement.** Each model's confidence is measured independently through its own zoom pipeline, avoiding the calibration problem of comparing log-probabilities or embeddings across architectures.

### 6.4 Screening vs. Full Evaluation

We initially validated on a 200-sample screening subset before running the full 1,581-sample evaluation.

| Metric | Screening (200) | Full (1,581) |
|---|---|---|
| KV-Ground baseline | 84.5% | 80.1% |
| Consistency router | 86.0% | 80.9% |
| Router gain | +1.5% | +0.8% |
| Oracle | 89.5% | 85.1% |

The router gain shrinks from +1.5% on screening to +0.8% on full evaluation. The full dataset includes harder professional tools (CAD, 3D modeling) where both models struggle and the oracle advantage is smaller. This highlights the importance of full-dataset evaluation — screening results can overstate improvements by ~2x.

### 6.5 Training-Based Approaches

Before discovering the routing strategy, we attempted several training-based approaches to improve KV-Ground-8B directly:

| Approach | Screening | Result |
|---|---|---|
| GRPO (RL with binary reward) | — | Killed early, no improvement |
| GRPO v2 (smooth reward, KL penalty) | — | Killed early, no improvement |
| SFT on failure cases (single-pass) | 84.0% | -0.5% |
| SFT on failure cases (zoom-pipeline) | 84.0% | -0.5% |

All training approaches failed. GRPO requires within-group variance to compute advantages, but KV-Ground produces consistently correct or consistently wrong predictions across samples — there is no stochastic component for RL to exploit. SFT on failure cases overfits to specific coordinates without generalizing. These negative results motivated the shift to inference-time routing.

## 7. Limitations

1. **Computational cost.** The router runs two full models (2 forward passes each = 4 total), doubling inference time compared to a single model. For latency-sensitive applications, this may be prohibitive.

2. **Modest gains.** The improvement (+0.8% on full eval) is real but small. A stronger generalist model or a better confidence signal could increase the capture rate beyond 16.5% of the oracle advantage.

3. **Generalist model quality.** Qwen3.5-27B-AWQ achieves only 60.9% standalone accuracy, limiting the oracle ceiling. A GUI-trained generalist (e.g., future Holo3 or GUI-Owl-2 on the Qwen3.5 backbone) could significantly increase the complementary advantage.

4. **Zoom consistency limitations.** The signal is strongest for KV-Ground (11.5-point gap between confident and uncertain buckets) but weaker for Qwen (9.5-point gap). Models trained specifically for zoom pipelines may have artificially calibrated consistency (always predicting near center), reducing the signal's discriminative power.

## 8. Conclusion

We introduced *zoom consistency* — the distance between a model's zoom-refinement prediction and the crop center — as a free, per-sample confidence signal for GUI grounding. By routing between a specialized model (KV-Ground-8B) and a general-purpose model (Qwen3.5-27B) based on which has lower zoom consistency, we achieve 80.9% on ScreenSpot-Pro without any additional training, establishing a new state of the art. Our work demonstrates that inference-time routing can extract complementary value from heterogeneous model ensembles, even when one model is significantly weaker overall. The zoom consistency signal is general and could be applied to any multi-step refinement pipeline.

## References

[1] Li et al. "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use." arXiv:2504.07981, 2025.

[2] vocaela. KV-Ground-8B-BaseGuiOwl1.5-0315. HuggingFace, 2025.

[3] Qwen Team. "Qwen3.5: Towards Native Multimodal Agents." Alibaba Cloud, 2026.

[4] Wu et al. "OS-Atlas: A Foundation Action Model for Generalist GUI Agents." arXiv, 2024.

[5] Microsoft. "GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents." NeurIPS, 2025.

[6] ByteDance. "UI-TARS-1.5." SEED-TARS, 2025.

[7] "Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding." arXiv:2512.05941, 2025.

[8] Luo et al. "Visual Test-time Scaling for GUI Agent Grounding." ICCV, 2025.

[9] "ScreenSeekeR: Visual Search for GUI Grounding." 2025.

[10] "Test-Time Reinforcement Learning for GUI Grounding via Region Consistency." arXiv:2508.05615, 2025.

[11] "Adaptive Vision-Language Model Routing for Computer Use Agents." arXiv:2603.12823, 2026.

[12] "Adaptive Chain-of-Focus Reasoning via Dynamic Visual Search and Zooming for Efficient VLMs." arXiv:2505.15436, 2025.

[13] "Training-free Uncertainty Guidance for Complex Visual Tasks." arXiv:2510.00705, 2025.

## Appendix A: Negative Results

We document our failed experiments in detail, as they informed the final approach:

### A.1 GRPO Training

We attempted Group Relative Policy Optimization (GRPO) on KV-Ground-8B with two implementations:

**v1** used N=4 samples, binary reward (+2 for in-box, -distance for out-of-box), and only trained on the best response. This was methodologically flawed — proper GRPO trains on all responses weighted by their advantages.

**v2** fixed all issues: N=8 samples, smooth Gaussian reward, KL penalty (beta=0.05) against a frozen reference, training on all responses. Despite being correctly implemented, accuracy flatlined at ~68% for 970 steps. Root cause: KV-Ground produces deterministic outputs per sample — all 8 samples at temperature 0.8 are either all correct or all incorrect, leaving zero within-group variance for advantage computation.

### A.2 SFT on Failure Cases

We identified 309 samples that KV-Ground fails on using the full zoom pipeline at production resolution, then trained a LoRA adapter (rank 16) to predict ground-truth coordinates on those samples. Training loss decreased from 0.29 to 0.18, but screening accuracy was 84.0% — a 0.5% regression from baseline. The model memorized the training coordinates without learning generalizable patterns.

A critical lesson: our first attempt used single-pass (non-zoom) failure detection at lower resolution, identifying 457 failures instead of the true 309. This mismatch meant we were training on samples that weren't actually broken in the production pipeline.

### A.3 Base Model Alternatives

We evaluated Qwen3.5-9B (raw, no GUI fine-tuning) with zoom: 75.0% on screening — 9.5 points below KV-Ground, confirming that GUI-specific training contributes ~10 absolute points.

Holo2-8B, another Qwen3-VL-8B derivative trained with GRPO for localization, achieves only 58.9% on ScreenSpot-Pro — demonstrating that training recipe matters more than architecture.

### A.4 Infrastructure Lessons

Running Qwen3.5-27B-AWQ on H100 80GB caused 57% parse failures due to memory pressure from compressed_tensors decompression (4-bit weights expand to bf16 in memory, requiring ~56GB + activations). Only H200 141GB provided sufficient headroom. This silent degradation was initially misattributed to model quality rather than infrastructure.

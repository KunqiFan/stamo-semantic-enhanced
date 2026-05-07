# StaMo Process Bridge: A Neuro-Symbolic Text-Semantic Bridge for Fine-Grained Manipulation Understanding

## 1. Introduction & Motivation
In embodied AI, understanding short-horizon manipulation requires bridging the gap between high-level language instructions (which are temporally extended and abstract) and low-level physical observations (which are dense, continuous, and opaque). Existing multimodal models often struggle to align a single textual instruction (e.g., "pick up the red block") with the precise micro-stages of execution (e.g., *approach*, *contact*, *grasp*, *lift*).

We introduce the **Text-Semantic Bridge**, a neuro-symbolic framework that leverages state-motion (StaMo) representations (`delta_z`) as a fine-grained, intermediate process-semantic interface. By disentangling continuous visual differences into structured, physically-grounded attributes, our method resolves critical temporal and spatial ambiguities inherent in text-only or end-to-end continuous models.

## 2. Methodology: The Text-Semantic Bridge

Our architecture operates in two cascaded stages, effectively forming a semantic bridge:

### 2.1 Latent-to-Symbolic Grounding (Bottom-Up)
Instead of forcing a direct mapping from raw images to high-level intentions, we extract short-horizon latent differences (`delta_z`) between frame $t$ and $t+k$ using a pretrained StaMo encoder. We then construct a set of **Compact Process Semantics**:
- **`contact_state`** $\in \{contact, no\_contact\}$
- **`gripper_state`** $\in \{open, closing, closed\}$
- **`object_motion`** $\in \{still, moved, lifted, placed\}$
- **`target_relation`** $\in \{farther, closer, reached\}$

This stage acts as an informational bottleneck that filters out irrelevant visual noise (e.g., lighting changes, background textures) and retains strictly the task-relevant physical dynamics.

### 2.2 Symbolic-Textual Fusion (Top-Down)
To predict the fine-grained action stage (e.g., *grasp* vs. *lift*), we fuse the grounded process semantics with the high-level language instruction. 
- **Text Stream**: Encodes the global task intention (e.g., "Task: pick up the block.").
- **Semantic Stream**: Provides the localized physical reality.

By utilizing stacking via meta-classifiers, the model learns non-linear decision boundaries—for instance, recognizing that "picking up" + `gripper_state: closing` implies the *grasp* stage, whereas "picking up" + `object_motion: lifted` implies the *lift* stage.

## 3. Qualitative Analysis & Case Studies

To understand *why* the Text-Semantic Bridge outperforms traditional baselines, we analyze specific failure modes resolved by our method:

### Case Study A: The Temporal Ambiguity of "Pick Up"
**Context**: The robot is given the instruction "pick up the green bowl" over a 10-second trajectory.
- **Text-Only Baseline Failure**: The text-only model receives the identical caption across all frames. Because the text is static, the model oscillates randomly between *approach* and *move*, completely lacking temporal grounding.
- **Semantic Bridge Resolution**: The visual difference `delta_z` reliably activates the semantic classifiers for `target_relation = closer` and `gripper_state = open`. The fusion model inherently understands that "approaching" + "open gripper" definitively translates to the **approach** stage, resolving the temporal ambiguity.

### Case Study B: Visual Occlusion During Contact
**Context**: The robotic arm obscures the target object right before grasping it.
- **Continuous Latent Baseline Failure**: Models relying solely on dense visual latents often fail when occlusions occur. When the object disappears beneath the end-effector, the continuous representations become noisy, leading to erratic stage predictions.
- **Semantic Bridge Resolution**: The intermediate semantic layer explicitly outputs `contact_state = contact` and `gripper_state = closing`. By forcing the model to explicitly reason about these discrete attributes, the system becomes highly robust to visual occlusion, correctly identifying the **grasp** phase regardless of pixel-level obstructions.

### Case Study C: Differentiating "Move" vs. "Lift"
**Context**: The robot has successfully grasped the object and is transporting it.
- **Ambiguity**: Both the *lift* and *move* stages involve significant displacement of the end-effector and identical gripper states (`closed`).
- **Semantic Bridge Resolution**: The `object_motion` attribute specifically tracks the Z-axis (height) dynamics extracted by StaMo. If `object_motion = lifted`, the semantic bridge cleanly distinguishes the initial upward trajectory (*lift*) from the subsequent planar translation (*move*), a distinction that language alone cannot make without precise geometric grounding.

## 4. Quantitative Formulation & Dataset Scale
The experiments are conducted on the official **DROID 100** real-world manipulation dataset. We process roughly 4,700 training pairs sampled with a strict `step_gap` to capture short-horizon dynamics.

The integration of advanced classifiers (HistGradientBoosting) ensures that the non-linear mapping from the continuous `delta_z` space to the discrete semantic space achieves high precision. The performance is comprehensively evaluated using Macro-F1 and confusion matrices to explicitly quantify the reduction in inter-stage misclassifications (e.g., confusing *approach* with *contact*).

## 5. Conclusion
The Text-Semantic Bridge demonstrates that explicit, physically-grounded intermediate representations (`delta_z` $\rightarrow$ Semantics) significantly enhance the interpretability and accuracy of process chain reasoning. It proves that combining the logical breadth of language models with the physical precision of continuous state-motion embeddings offers a superior paradigm for fine-grained embodied task understanding.

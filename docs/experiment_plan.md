# Experiment Plan

## Research Question

Can StaMo's state difference (`delta_z`) provide fine-grained short-horizon process evidence that supplements a text or multimodal process chain for understanding short manipulation changes?

## Hypothesis

Text-only descriptions are often too coarse for short-horizon manipulation changes such as contact formation, grasp completion, or object lift. StaMo `delta_z` may encode these micro-transitions, and a compact semantic interface derived from `delta_z` can make that information easier to integrate with a text/process chain.

## Scope Guardrails

- Do not redesign StaMo
- Do not build a long-horizon world model from scratch
- Do not target full robot deployment
- Do not optimize for free-form caption generation
- Stay focused on short-horizon state transitions across image pairs or short snippets

## Proposed Task Definition

Each sample is a short manipulation window:

- input: `(image_t, image_t+k, optional action snippet, optional caption/process text)`
- latent view: `z_t`, `z_t+k`, `delta_z = z_t+k - z_t`
- label view: short-horizon state changes

Recommended label set:

1. `contact_state`: no_contact / contact
2. `gripper_state`: open / closing / closed
3. `object_motion`: still / lifted / moved / placed
4. `target_relation`: farther / closer / reached

Optional coarse stage label:

1. `approach`
2. `contact`
3. `grasp`
4. `lift`
5. `move`
6. `place`

## Datasets

Use one existing short-horizon manipulation source rather than building a new dataset. Good candidates are:

- a subset of a robot manipulation dataset already used around StaMo or VLA evaluation
- a simulated pick-place dataset with image observations and action/state annotations
- a manually relabeled subset of short clips if full labels are unavailable

Selection criteria:

- image pairs or short clips are easy to extract
- basic end-effector/object signals exist or can be approximated
- enough repeated short transitions exist for balanced classification

## Three Required Baselines

### 1. Text-only / caption-only

Inputs:

- caption for the pair or short clip
- optional templated process text

Models:

- bag-of-words or sentence embedding + linear classifier
- optional small multimodal/text encoder if already available

Purpose:

- quantify how much short-horizon change can be inferred from language alone

### 2. `delta_z` only

Inputs:

- StaMo `delta_z`

Models:

- linear probe
- small MLP probe

Purpose:

- test whether latent state difference already contains short-horizon process evidence

### 3. Text process chain + StaMo-derived compact semantics

Inputs:

- caption/process text
- compact semantic labels or predicted semantic attributes derived from `delta_z`

Models:

- concatenate text embedding with semantic one-hot or logits
- optional late fusion with a lightweight MLP

Purpose:

- test whether the semantic bridge improves over text-only

## Minimal Modeling Recipe

### Part A: `delta_z` extraction

1. Load StaMo encoder
2. Encode start and end observations
3. Compute `delta_z = z_end - z_start`
4. Save feature vectors to `data/interim/`

### Part B: semantic interface

Two practical options:

1. Supervised projection from `delta_z -> compact labels`
2. Rule or clustering assisted bootstrap, then refine with annotation

Start with supervised multi-head classification because it is small and testable.

### Part C: fusion

Represent compact semantics as:

- one-hot labels
- classifier logits
- low-dimensional semantic embedding

Then fuse with text features for downstream classification.

## Evaluation

### Task 1: short-horizon state-change classification

Predict one or more compact labels.

Metrics:

- accuracy
- macro-F1
- per-class F1

### Task 2: probing short-horizon dynamics

Predict:

- next action
- short-horizon state change
- end-effector delta

Metrics:

- MSE
- MAE
- R^2 if helpful for regression probes

## Ablations

Keep the ablations small:

1. `delta_z` versus `z_t || z_t+k`
2. attribute semantics versus coarse stage labels
3. gold semantics versus predicted semantics
4. single-frame caption versus pair caption

## Semester Timeline

### Weeks 1-2

- choose dataset subset
- define compact labels
- build manifest and splits

### Weeks 3-5

- connect StaMo encoder
- extract `delta_z`
- run quick probe sanity checks

### Weeks 6-8

- train compact semantic predictors
- inspect confusion patterns

### Weeks 9-11

- run text-only and fused baselines
- complete ablations

### Weeks 12-14

- finalize figures
- write report
- package reproducible scripts

## Success Criteria

The project is successful if at least one of these is true:

1. `delta_z` alone strongly predicts fine-grained short-horizon labels
2. `text_plus_semantics` improves over `text_only`
3. attribute-style semantics are more useful than coarse stage labels

## Likely Risks

- labels are too noisy or imbalanced
- text captions already leak too much signal
- dataset lacks enough clear short transitions
- semantic interface is too fine-grained for the data size

## Mitigations

- start with 3-4 attributes, not 10+
- use balanced subsets
- inspect class distributions early
- keep a simple linear probe as the default baseline

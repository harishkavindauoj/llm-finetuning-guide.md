# The Complete Guide to LLM Fine-tuning Techniques

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Fine-tuning Fundamentals](#understanding-fine-tuning-fundamentals)
3. [Full Fine-tuning](#full-fine-tuning)
4. [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
5. [QLoRA (Quantized LoRA)](#qlora-quantized-lora)
6. [AdaLoRA (Adaptive LoRA)](#adalora-adaptive-lora)
7. [IA³ (Infused Adapter)](#ia³-infused-adapter)
8. [BitFit](#bitfit)
9. [Prompt Tuning & Prefix Tuning](#prompt-tuning--prefix-tuning)
10. [Adapter Layers](#adapter-layers)
11. [LongLoRA](#longlora)
12. [GaLore](#galore)
13. [Mixture of LoRA (MoLoRA)](#mixture-of-lora-molora)
14. [Comparison & Decision Framework](#comparison--decision-framework)
15. [Practical Implementation Guide](#practical-implementation-guide)

---

## Introduction

Fine-tuning adapts a pre-trained language model to your specific domain or task. Instead of training from scratch (expensive and data-intensive), you start with a model that already understands language and teach it your specialized knowledge.

**Why Fine-tune?**
- Domain expertise (medical, legal, engineering, etc.)
- Task specialization (summarization, classification, Q&A)
- Style adaptation (formal, casual, technical)
- Proprietary knowledge integration

**Key Challenge:** LLMs have billions of parameters. How do we train them efficiently?

---

## Understanding Fine-tuning Fundamentals

### What Happens During Training

**Pre-training:** Model learns from massive text corpus (trillions of tokens)
- Understands grammar, facts, reasoning patterns
- General-purpose knowledge

**Fine-tuning:** Adjust weights for specific use case
- Learn domain terminology
- Adopt specific reasoning patterns
- Align with task requirements

### The Memory Problem

For a 7B parameter model:
```
Model weights:           14 GB (FP16)
Gradients:              14 GB
Optimizer states:       56 GB (Adam)
Activations:            ~20 GB
─────────────────────────────────
Total:                  ~104 GB
```

**Problem:** Most GPUs have 16-48 GB VRAM. We need efficient techniques.

---

## Full Fine-tuning

### Overview
Update **all** model parameters during training. This is the baseline approach.

### How It Works
```
Training loop:
1. Forward pass: Input → Model → Prediction
2. Calculate loss: Compare prediction to ground truth
3. Backward pass: Compute gradients for all parameters
4. Update weights: W_new = W_old - learning_rate × gradient
```

### Memory Requirements

- **7B model:** ~104 GB (requires 2-4 high-end GPUs)
- **13B model:** ~180 GB (requires 4-8 GPUs)
- **70B model:** ~1 TB (requires 16+ GPUs)

### Training Configuration
```python
learning_rate = 1e-5        # Small to prevent catastrophic forgetting
batch_size = 8-32          # Depends on GPU memory
epochs = 3-5               # Usually sufficient
warmup_steps = 100         # Gradual learning rate increase
```

### When to Use

✅ **Use Full Fine-tuning When:**
- You have access to multi-GPU clusters
- Dataset is very large (100K+ examples)
- Domain is radically different from pre-training
- You need absolute maximum performance
- Budget allows (expensive)

❌ **Don't Use When:**
- Limited GPU resources (single consumer GPU)
- Small dataset (<10K examples) - high overfitting risk
- Quick iteration needed
- Cost-sensitive projects

### Pros & Cons

**Advantages:**
- Maximum model capacity
- Best possible accuracy
- Simple to implement

**Disadvantages:**
- Extremely memory-intensive
- Slow training (days to weeks)
- High risk of overfitting on small datasets
- Expensive compute costs

---

## LoRA (Low-Rank Adaptation)

### Overview
The **most popular** parameter-efficient fine-tuning method. Instead of updating all weights, LoRA adds small trainable matrices alongside frozen weights.

### Core Concept

**Key Insight:** Weight updates during fine-tuning are low-rank (can be represented in lower dimensions).

**Mathematical Foundation:**
```
Standard fine-tuning:
W_new = W_pretrained + ΔW (full rank)

LoRA:
W_new = W_pretrained + B × A (low rank)

Where:
- W_pretrained: frozen (not updated)
- B: [d × r] trainable matrix
- A: [r × d] trainable matrix  
- r: rank (typically 8-64, much smaller than d)
```

**Example:**
```
Original weight matrix: 4096 × 4096 = 16.7M parameters
LoRA with r=16: (4096×16) + (16×4096) = 131K parameters
Reduction: 99.2%!
```

### How It Works

**Forward Pass:**
```
h = W × x + (B × A) × x
    ↑         ↑
  frozen   trainable
```

The computation is efficient:
```
h = W × x + B × (A × x)
           └─ compute this first (small)
    └─ then add to base output
```

### Key Hyperparameters

#### 1. Rank (r)
Controls adapter capacity.
```
r = 4:    Minimal adaptation (very small changes)
r = 8:    Light domain adaptation
r = 16:   Standard choice (balanced)
r = 32:   Complex task adaptation
r = 64:   Near full fine-tuning capacity
```

**How to choose:**
- Start with r=16
- If underfitting (high validation loss) → increase r
- If overfitting (train loss << val loss) → decrease r

#### 2. Alpha (α)
Scaling factor that controls learning rate for LoRA weights.
```
Contribution = (α / r) × B × A

Typical values:
α = r:      Standard (e.g., r=16, α=16)
α = 2r:     Common choice (e.g., r=16, α=32)
α = r/2:    Conservative
```

#### 3. Target Modules
Which layers to apply LoRA to?

**Transformer architecture:**
```
Attention Layers:
├── Q (query) projection
├── K (key) projection  
├── V (value) projection
└── O (output) projection

Feed-Forward Layers:
├── Up projection
└── Down projection
```

**Common configurations:**
```python
# Minimal (fastest)
target_modules = ["q_proj", "v_proj"]

# Balanced (recommended)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Comprehensive (all layers)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "up_proj", "down_proj"]
```

**Why Q and V are most important:**
- Query: Determines what information to look for
- Value: Determines what information to retrieve
- These capture most task-specific behavior

#### 4. Dropout
Regularization to prevent overfitting.
```python
lora_dropout = 0.05    # Standard
lora_dropout = 0.1     # More regularization (larger datasets)
lora_dropout = 0.0     # No dropout (small datasets)
```

### Memory Requirements
```
7B model with LoRA (r=16):
Model weights:    14 GB (frozen, FP16)
LoRA adapters:    ~100 MB (trainable)
Optimizer:        ~200 MB (only for LoRA)
Activations:      ~4 GB
─────────────────────────────
Total:            ~18 GB ✓ Fits on single A100/A6000
```

### Training Configuration
```python
# Typical LoRA settings
learning_rate = 2e-4              # Higher than full FT (1e-5)
batch_size = 4-8
gradient_accumulation_steps = 4   # Effective batch = 16-32
epochs = 3-5
warmup_steps = 50-100
weight_decay = 0.01
```

**Why higher learning rate?**
Base model is frozen (stable), so we can learn LoRA weights more aggressively.

### When to Use

✅ **Use LoRA When:**
- Single GPU or limited multi-GPU setup
- Dataset: 500-50K examples
- Domain adaptation needed (e.g., medical → electrical engineering)
- Quick iteration required (hours, not days)
- Want to train multiple task-specific models
- Standard go-to choice for most projects

❌ **Don't Use When:**
- GPU memory <12 GB (use QLoRA instead)
- Dataset <100 examples (few-shot prompting better)
- Need absolute maximum accuracy (full FT better by 1-2%)

### Pros & Cons

**Advantages:**
- 99% parameter reduction
- Fast training (hours vs days)
- Low memory usage (~18 GB for 7B model)
- Can train multiple adapters for different tasks
- Easy to merge back into base model
- Good accuracy retention (96-99% of full FT)

**Disadvantages:**
- Slightly lower accuracy than full fine-tuning
- Not ideal for completely new languages/modalities
- Requires understanding of hyperparameters

---

## QLoRA (Quantized LoRA)

### Overview
Combines LoRA with 4-bit quantization. Enables training 7B-13B models on consumer GPUs (RTX 3090/4090).

### Core Concept

**Two innovations:**
1. **4-bit quantization:** Compress base model weights from 16-bit to 4-bit
2. **LoRA adapters:** Train in 16-bit for quality

**Why this works:**
- Base model can be low-precision (inference only)
- Gradients computed in high-precision (quality preserved)
- Best of both worlds: memory efficiency + accuracy

### Quantization Explained

**Precision levels:**
```
FP32 (32-bit): 3.14159265359...     (4 bytes per number)
FP16 (16-bit): 3.141                (2 bytes per number)
INT8 (8-bit):  3 (scaled)           (1 byte per number)
INT4 (4-bit):  3 (scaled)           (0.5 bytes per number)
```

**Memory savings:**
```
7B model:
FP16:   7B × 2 bytes = 14 GB
INT4:   7B × 0.5 bytes = 3.5 GB
Reduction: 4× smaller!
```

### NormalFloat4 (NF4)

Standard quantization is linear. QLoRA uses **NF4** - optimized for neural network weight distributions.

**Why NF4 is better:**
Neural network weights follow a normal distribution (bell curve). NF4 allocates more precision near zero (where most weights are) and less at extremes.
```
Standard INT4: Equal spacing
[-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

NF4: Non-uniform spacing (more dense near zero)
[-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
  0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0]
```

### Double Quantization

Further compress the quantization constants themselves.
```
Standard quantization:
Weights: 4-bit (compressed)
Scale factors: FP32 (not compressed)

Double quantization:
Weights: 4-bit  
Scale factors: FP8 (also compressed!)
Additional savings: ~0.37 bits per parameter
```

### Memory Requirements
```
7B model with QLoRA (r=16):
Model weights:    3.5 GB (4-bit quantized)
LoRA adapters:    ~100 MB (FP16)
Optimizer:        ~200 MB
Activations:      ~4 GB
─────────────────────────────
Total:            ~8 GB ✓ Fits on RTX 3090 (24GB)
```

### Training Configuration
```python
# QLoRA typically needs higher learning rate
learning_rate = 2e-4 to 5e-4      # Higher than LoRA
batch_size = 2-4                  # Smaller (still limited)
gradient_accumulation_steps = 8   # Compensate with accumulation
epochs = 3-5
```

### Performance

**Accuracy retention:**
- QLoRA vs Full LoRA: 98-99%
- QLoRA vs Full FT: 95-98%
- Minimal loss for most tasks
- Slightly more impact on math/reasoning

**Speed:**
- Training: ~20-30% slower than LoRA (dequantization overhead)
- Inference: Can use quantized version (faster)

### When to Use

✅ **Use QLoRA When:**
- Consumer GPU (RTX 3090/4090, 24GB VRAM)
- Want to train 7B-13B models locally
- Budget-conscious (avoid cloud GPU costs)
- Dataset: 1K-50K examples
- Standard choice for individual researchers/engineers

❌ **Don't Use When:**
- Have access to high-end GPUs (A100) - use regular LoRA
- Training small models (<3B) - regular LoRA sufficient
- Need maximum training speed
- Absolute best accuracy required

### Pros & Cons

**Advantages:**
- Enables large model training on consumer hardware
- 75% less memory than LoRA
- Good accuracy (95-98% of full FT)
- Cost-effective for development

**Disadvantages:**
- 20-30% slower training than LoRA
- Small accuracy drop vs regular LoRA
- More complex setup

---

## AdaLoRA (Adaptive LoRA)

### Overview
Dynamically adjusts rank across different layers during training. Optimizes parameter budget by allocating more capacity to important layers.

### The Problem with Fixed Rank

Standard LoRA uses same rank everywhere:
```
Layer 1:  r=16
Layer 2:  r=16
...
Layer 24: r=16

But different layers need different capacity!
- Early layers: General features (lower rank OK)
- Middle layers: Task-specific patterns (higher rank needed)
- Late layers: Output formatting (lower rank OK)
```

### How It Works

**Dynamic rank allocation:**
```
1. Start: All layers with high rank (e.g., r=32)
2. Training: Compute importance of each adapter
3. Pruning: Remove less important singular values
4. Reallocation: Give freed capacity to important adapters
5. Continue: Train with optimized ranks
```

**Importance metric:**
```
For each adapter:
Importance = Σ (singular_value × gradient_magnitude)

High importance = Both large singular values AND large gradients
→ This adapter is actively learning important patterns
```

### Training Dynamics
```
Initial (step 0):
Layer 1:  r=32 ████████████████
Layer 2:  r=32 ████████████████
Layer 12: r=32 ████████████████

After 1000 steps:
Layer 1:  r=8  ████
Layer 2:  r=48 ████████████████████
Layer 12: r=16 ████████

Budget redistributed to important layers!
```

### Configuration
```python
# AdaLoRA parameters
init_r = 32              # Starting rank
target_r = 16            # Target average rank
tinit = 1000            # Warmup before pruning
tfinal = 10000          # Final pruning step
deltaT = 500            # Prune every 500 steps
```

### Performance
```
Same parameter budget comparison:

LoRA (r=16 uniform):        85.8% accuracy
AdaLoRA (budget=512):       86.2% accuracy

AdaLoRA achieves better results with same parameter count
by optimal allocation!
```

### When to Use

✅ **Use AdaLoRA When:**
- Want optimal parameter efficiency
- Training time not critical (20-50% longer)
- Production deployment (worth optimization)
- Multiple fine-tuning rounds planned
- Research/comparison projects

❌ **Don't Use When:**
- Quick experiments needed
- Very small models (<500M parameters)
- Short training runs (<1K steps)
- First time fine-tuning (use standard LoRA first)

### Pros & Cons

**Advantages:**
- Optimal parameter allocation
- Better accuracy than fixed-rank LoRA
- Interpretable (see which layers matter)

**Disadvantages:**
- More complex implementation
- 20-50% longer training
- Requires tuning of pruning schedule
- Less tooling support

---

## IA³ (Infused Adapter)

### Overview
Learn multiplicative scaling vectors instead of additive weight updates. Extremely parameter-efficient.

### Core Concept

**Comparison with LoRA:**
```
LoRA (additive):
h = W × x + (B × A) × x
    └─ add learned update

IA³ (multiplicative):  
h = (scale_W ⊙ W) × (scale_x ⊙ x)
     └─ scale weights   └─ scale inputs
```

**Key insight:** Instead of learning new patterns, learn to amplify/inhibit existing patterns.

### How It Works

**Three scaling vectors per layer:**
```
For attention:
- scale_k: Scale key projections (what to attend to)
- scale_v: Scale value projections (what to retrieve)
- scale_o: Scale output (how much to use)

For feed-forward:
- scale_in: Scale input
- scale_out: Scale output
```

**Why not scale queries?**
Scaling queries changes attention distribution globally, which is unstable. Scaling keys/values is more controlled.

### Parameter Efficiency
```
7B model (hidden_dim=4096, 32 layers):

Per attention layer: 3 × 4096 = 12,288 parameters
Per FFN layer: 2 × 4096 = 8,192 parameters
Total: 32 layers × 20,480 = 655,360 parameters

Comparison:
IA³:         ~0.01% of model parameters
LoRA (r=8):  ~0.1% of model parameters
LoRA (r=16): ~0.2% of model parameters

IA³ is 10-20× more efficient!
```

### Training Configuration
```python
# IA³ uses higher learning rates (fewer parameters)
learning_rate = 3e-3             # Much higher than LoRA
batch_size = 16-32               # Can use larger batches  
epochs = 5-10                    # More epochs needed
warmup_ratio = 0.05
```

### Performance
```
Benchmarks (T5 models):

Full Fine-tuning:  45.2 (baseline)
LoRA (r=8):        44.1 (97.6%)
IA³:               43.5 (96.2%)
Adapter:           43.8 (96.9%)

IA³ maintains 96%+ performance with 10× fewer parameters than LoRA
```

### When to Use

✅ **Use IA³ When:**
- T5 or encoder-decoder models (works best here)
- Extreme parameter efficiency needed
- Multi-task serving (100+ task-specific scalings)
- Translation/summarization tasks
- Simple domain adaptation

❌ **Don't Use When:**
- Decoder-only models (LoRA works better)
- Complex reasoning required
- Very small datasets (<500 examples)
- First-time fine-tuning (less common method)

### Pros & Cons

**Advantages:**
- Extremely parameter-efficient (0.01% of model)
- Very fast training
- Easy to swap task-specific scalings
- Works well on T5 family

**Disadvantages:**
- Limited expressiveness vs LoRA
- Less popular (fewer tutorials/tools)
- Better for encoder-decoder than decoder-only
- Lower accuracy than LoRA on complex tasks

---

## BitFit

### Overview
Only train bias terms, freeze all weights. Simplest parameter-efficient method.

### Core Concept

**Standard linear layer:**
```
output = W × x + b
         ↑       ↑
      frozen  trainable (only this!)
```

**Why biases?**
Biases act as offset adjustments. They shift decision boundaries without changing feature directions.
```
Before: activation(W × x + b_old) = 0.9
After:  activation(W × x + b_new) = 1.6

Bias makes certain patterns more/less salient
```

### Parameter Count
```
7B model:

Total parameters: 7,000,000,000
Bias parameters:  ~6,000,000 (0.086%)

Breakdown:
- Embedding bias: ~50K
- Attention biases: ~3M  
- FFN biases: ~2.8M
- Output bias: ~50K
```

### Training Configuration
```python
learning_rate = 1e-3          # Higher than full FT
epochs = 10-20                # More epochs needed
batch_size = 16-32            # Can use large batches
weight_decay = 0.0            # No weights trained!
```

### Performance
```
Compared to other methods:

Full Fine-tuning: 100%
LoRA (r=16):      96-98%
BitFit:           85-92%
Prefix:           82-88%

BitFit has lower accuracy but uses fewest parameters
```

### When to Use

✅ **Use BitFit When:**
- Extremely limited resources
- Need to train 100+ task variants
- Simple style/terminology adaptation
- Quick prototyping
- Multi-tenant serving (swap biases cheaply)

❌ **Don't Use When:**
- Need high accuracy
- Complex reasoning changes required
- Have resources for LoRA
- Working with small datasets

### Pros & Cons

**Advantages:**
- Simplest method (just unfreeze biases)
- Minimal memory overhead
- Very fast training
- Easy multi-task deployment

**Disadvantages:**
- Lowest accuracy among PEFT methods
- Requires many more epochs
- Limited capacity for complex changes

---

## Prompt Tuning & Prefix Tuning

### Overview
Add learnable "virtual tokens" to the input instead of modifying model weights.

### Prompt Tuning

**Concept:** Learn continuous embeddings prepended to input.
```
Standard input:
[The, cat, sat, on, the, mat]

Prompt tuning:
[V₁, V₂, V₃, V₄, V₅] + [The, cat, sat, on, the, mat]
 └─ Learned vectors      └─ Actual input
```

**Key point:** Virtual tokens only at input layer.

**Parameters:**
```
20 tokens × 768 dimensions = 15,360 parameters
Extremely lightweight!
```

### Prefix Tuning

**Concept:** Add learnable vectors at **every** transformer layer.
```
Layer 0: [Prefix₀] + input
Layer 1: [Prefix₁] + hidden  
Layer 2: [Prefix₂] + hidden
...
Layer N: [Prefix_N] + hidden
```

**Parameters:**
```
20 tokens × 768 dimensions × 12 layers = 184,320 parameters
More capacity than prompt tuning
```

### Comparison
```
Method          | Parameters | Performance | Best For
----------------|------------|-------------|------------------
Prompt Tuning   | 15K        | 75-85%      | >10B models only
Prefix Tuning   | 180K       | 82-90%      | >3B models
LoRA            | 100K-1M    | 96-99%      | All model sizes
```

**Key finding:** Both struggle on smaller models (<3B). Need very large models to work well.

### When to Use

✅ **Use Prompt/Prefix Tuning When:**
- Working with very large models (>10B)
- Simple tasks (classification, extraction)
- Extreme parameter efficiency needed
- Want to swap tasks instantly (just change prefix)

❌ **Don't Use When:**
- Model <3B parameters
- Complex reasoning tasks
- Need high accuracy
- Working with small datasets

### Pros & Cons

**Advantages:**
- Minimal parameters
- No architecture changes
- Fast task switching

**Disadvantages:**
- Only works well on large models
- Lower accuracy than LoRA
- Difficult to optimize
- Limited to simple tasks

---

## Adapter Layers

### Overview
Insert small bottleneck neural networks between frozen layers. Older technique, mostly superseded by LoRA.

### Architecture
```
Frozen Layer N
      ↓
  ┌─────┐
  │Down │  [768 → 64] Reduce dimensions
  │ReLU │  Nonlinearity
  │ Up  │  [64 → 768] Expand back
  │  +  │  Add to original (residual)
  └─────┘
      ↓
Frozen Layer N+1
```

### Parameters
```
Adapter size 64, hidden 768:

Down projection: 768 × 64 = 49,152
Up projection:   64 × 768 = 49,152
Total per adapter: 98,304

24 layers × 2 adapters = 48 adapters
Total: ~4.7M parameters
```

### When to Use

✅ **Use Adapters When:**
- Need explicit modularity
- Want to stack multiple task adapters
- Working with older codebases
- Simple task-specific learning

❌ **Don't Use When:**
- Starting new project (use LoRA instead)
- Need parameter efficiency (LoRA better)
- Speed is critical (slower inference)

### Pros & Cons

**Advantages:**
- Simple and interpretable
- Easy to add/remove
- Multiple adapters can coexist

**Disadvantages:**
- Less efficient than LoRA
- Slower inference (extra computation)
- Older technique with less support

---

## LongLoRA

### Overview
Extends LoRA for long context lengths (32K+ tokens). Combines sparse attention during training with full attention at inference.

### The Long Context Problem
```
Standard attention: O(n²) complexity

Sequence 2K:   4M operations
Sequence 8K:   64M operations (16× more!)
Sequence 32K:  1B operations (256× more!)

Memory explodes!
```

### Solution: Shifted Sparse Attention

**During training only:**
```
Instead of: Each token attends to ALL previous tokens
Use: Each token attends to LOCAL WINDOW only

Token 1000 → attends to tokens 950-1050 (window=100)

Problem: Tokens far apart never interact

Solution: SHIFT windows in alternating layers
Layer 1: Window at [i-50:i+50]
Layer 2: Window at [i-25:i+75] (shifted)
Layer 3: Window at [i-50:i+50]
Layer 4: Window at [i-25:i+75] (shifted)

Result: Information propagates across entire sequence
```

### Training vs Inference
```
Training:
- Use sparse attention (memory efficient)
- Can train on 32K contexts with 24GB VRAM

Inference:  
- Use full attention (better quality)
- LoRA adapters work with full attention
```

### Memory Savings
```
Standard LoRA on 32K context:
Attention: 32768² × 4 bytes = 4.3 GB
Total: ~25 GB → Doesn't fit!

LongLoRA with window=1024:
Attention: 32768 × 1024 × 4 bytes = 134 MB  
Total: ~20 GB → Fits on A100!
```

### When to Use

✅ **Use LongLoRA When:**
- Need context >8K tokens
- Document analysis (reports, papers, manuals)
- Long conversations
- Code analysis (multiple files)
- Limited GPU memory

❌ **Don't Use When:**
- Context <4K tokens (standard LoRA sufficient)
- Have unlimited GPU budget
- Need absolute best quality per token

### Pros & Cons

**Advantages:**
- Enables 32K+ context on single GPU
- Minimal memory overhead vs standard LoRA
- Good quality retention

**Disadvantages:**
- Slightly lower quality than full attention
- More complex implementation
- Training slower than standard LoRA

---

## GaLore

### Overview
Completely different approach: Keep all parameters trainable but compress **gradients** instead of parameters.

### The Key Insight
```
Problem: Gradients take as much memory as weights

7B model:
Weights:   14 GB
Gradients: 14 GB  ← Can we compress this?

Insight: Gradients are low-rank during fine-tuning!
```

### How It Works
```
Standard gradient descent:
1. Compute full gradient G [4096 × 4096]
2. Update: W ← W - lr × G

GaLore:
1. Compute full gradient G [4096 × 4096]
2. Project to low-rank: G_low [256 × 256]
3. Store only G_low (much smaller!)
4. Project back for update
5. Update: W ← W - lr × G_projected
```

### Memory Savings
```
Without GaLore:
Gradient for 4096×4096 layer: 67 MB

With GaLore (rank=256):
Projected gradient: 0.26 MB
Projection matrices: 8.4 MB
Total: 8.66 MB (87% savings per layer!)

7B model total:
Original: 14 GB gradients
GaLore: 2 GB gradients
Savings: 12 GB
```

### Full Memory Budget
```
7B model with GaLore:

Weights:            14 GB
Gradients (low):     2 GB (reduced!)
Optimizer states:   56 GB (still full)
Activations:        20 GB
──────────────────────────
Total:              92 GB

vs Full FT: 104 GB
Savings: 12 GB
```

### Performance
```
GaLore matches full fine-tuning accuracy!

Full Fine-tuning:  Perplexity 12.3
GaLore:           Perplexity 12.4  
LoRA (r=64):      Perplexity 12.7

All parameters trainable + 12GB less memory
```

### When to Use

✅ **Use GaLore When:**
- Need maximum accuracy (vs LoRA)
- Have 2-4 GPUs but not full cluster
- Large architecture changes needed
- Continual pre-training
- Research requiring full model updates

❌ **Don't Use When:**
- Single GPU <40GB (won't fit)
- Parameter efficiency is goal (LoRA better)
- Quick iteration needed
- Simple domain adaptation

### Pros & Cons

**Advantages:**
- All parameters trainable (like full FT)
- Better accuracy than LoRA
- 12 GB memory savings vs full FT

**Disadvantages:**
- Still requires significant GPU memory
- Slower than LoRA
- More complex implementation
- Less tooling support

---

## Mixture of LoRA (MoLoRA)

### Overview
Multiple specialized LoRA adapters with routing mechanism. Different experts handle different types of queries.

### Core Concept
```
Instead of: One LoRA adapter for everything

Use: Multiple specialized adapters + router

Input → Router → Select top-k experts → Combine outputs

Example:
Query: "Calculate three-phase fault current"

Router scores:
- Expert 1 (math):      0.8  ← Selected
- Expert 2 (retrieval): 0.1
- Expert 3 (design):    0.1

Output = Base + 0.8 × Expert_1
```

### Architecture
```
Base Model (frozen)
      ↓
  ┌─────────────────┐
  │  Router Network │ (learns which expert for which query)
  └─────────────────┘
      ↓
  ┌───┬───┬───┬───┐
  │E₁ │E₂ │E₃ │...│ (8 LoRA experts)
  └───┴───┴───┴───┘
      ↓
  Weighted Combination
```

### Expert Specialization

After training, experts naturally specialize:
```
Expert 1: Math/calculations → handles circuit analysis
Expert 2: Retrieval → handles standard lookups  
Expert 3: Generation → handles report writing
Expert 4: Reasoning → handles troubleshooting
...

Model learns which expert is best for each task type
```

### Performance
```
Multi-domain dataset:

Single LoRA (r=16):
- Domain A: 82%
- Domain B: 79%
- Domain C: 76%
Average: 79%

MoLoRA (8 experts, r=8 each):
- Domain A: 86% (different experts specialize!)
- Domain B: 84%
- Domain C: 81%
Average: 83.7%

Same total parameters, 4.7% better!
```

### When to Use

✅ **Use MoLoRA When:**
- Multi-domain or multi-task learning
- Dataset has distinct clusters
- Want interpretability (which expert does what)
- Model capacity is bottleneck
- Have 2-3× training time budget

❌ **Don't Use When:**
- Single narrow task
- Limited training data (<5K examples)
- Need fast training
- Serving latency critical

### Pros & Cons

**Advantages:**
- Better multi-task performance
- Interpretable specialization
- Same parameters as single LoRA
- Can analyze expert usage patterns

**Disadvantages:**
- Slower training (routing overhead)
- More complex implementation
- Requires more training data
- Slower inference (routing + multiple experts)

---

## Comparison & Decision Framework

### Complete Comparison Table

| Technique | Trainable Params | Memory | Train Speed | Inference | Accuracy | Complexity |
|-----------|-----------------|--------|-------------|-----------|----------|-----------|
| **Full FT** | 100% | Highest | Slow | Fast | 100% | Simple |
| **LoRA** | 0.1-1% | Medium | Medium | Fast | 96-99% | Medium |
| **QLoRA** | 0.1-1% | Lowest | Slow | Fast | 95-98% | Medium |
| **AdaLoRA** | 0.05-0.5% | Medium | Slower | Fast | 96-99% | High |
| **IA³** | 0.01% | Low | Fast | Fast | 94-97% | Medium |
| **BitFit** | 0.08% | Low | Fast | Fast | 85-92% | Simple |
| **GaLore** | 100% | High | Slow | Fast | 99-100% | High |
| **LongLoRA** | 0.1-1% | Medium | Medium | Medium | 95-98% | High |
| **MoLoRA** | 0.5-2% | Medium | Slow | Medium | 97-99% | High |
| **Adapter** | 1-3% | Medium | Medium | Slow | 92-96% | Simple |
| **Prefix** | <0.1% | Low | Fast | Fast | 82-90% | Medium |
| **Soft Prompt** | <0.01% | Low | Fast | Fast | 75-85% | Simple |

### Decision Tree by Hardware
```
┌─ What GPU do you have?
│
├─ Consumer GPU (RTX 3090/4090, 24GB)
│  └─ Use: QLoRA
│     • r=16 for balanced tasks
│     • Best option for single consumer GPU
│
├─ Workstation GPU (A5000/A6000, 48GB)  
│  └─ Use: LoRA
│     • r=16-32 depending on complexity
│     • Can train 13B models
│
├─ High-end GPU (A100, 80GB)
│  ├─ Standard tasks → LoRA (r=32-64)
│  ├─ Multi-domain → MoLoRA (8 experts)
│  └─ Maximum accuracy → GaLore or Full FT
│
└─ API only (no local GPU)
   └─ Use: OpenAI/Anthropic fine-tuning
      • Cloud-based, pay per token
      • Easiest but least control
```

### Decision Tree by Dataset Size
```
Dataset Size → Recommended Method

<100 examples
└─ Don't fine-tune
   Use: Few-shot prompting with good examples

100-500 examples
├─ QLoRA (r=8) - if consumer GPU
└─ BitFit - if very simple task

500-2K examples
├─ QLoRA (r=16) - consumer GPU
└─ LoRA (r=8-16) - workstation GPU

2K-10K examples
├─ LoRA (r=16-32)
└─ AdaLoRA - if optimization matters

10K-50K examples
├─ LoRA (r=32-64)
├─ MoLoRA - if multi-domain
└─ GaLore - if need max accuracy

50K+ examples
└─ GaLore or Full Fine-tuning
   • Research-grade systems
```

### Decision Tree by Task Type
```
Task → Best Technique

Domain Adaptation (e.g., adding engineering terminology)
├─ LoRA (r=16)
├─ QLoRA (if consumer GPU)
└─ Config: Q+V modules, 3 epochs

Math/Calculation Tasks
├─ LoRA (r=32)
├─ GaLore (if need perfection)
└─ Config: All attention + FFN modules

Long Documents (>8K tokens)
├─ LongLoRA
└─ Config: r=16, sparse window=2048

Multi-Domain Tasks
├─ MoLoRA (4-8 experts)
└─ Config: r=8 per expert, k=2

Classification/Simple Tasks
├─ BitFit
└─ IA³ (if using T5)

Generation (reports, documentation)
├─ LoRA (r=32)
└─ Config: Comprehensive modules
```

### Quick Reference Guide

**Starting out?** → QLoRA (r=16)
**Have good GPU?** → LoRA (r=16-32)
**Multiple domains?** → MoLoRA
**Long contexts?** → LongLoRA
**Maximum accuracy?** → GaLore or Full FT
**Minimal parameters?** → BitFit or IA³
**T5 models?** → IA³

---

## Practical Implementation Guide

### Setup & Installation
```bash
# Core libraries
pip install transformers datasets torch

# For LoRA/QLoRA/PEFT methods
pip install peft accelerate bitsandbytes

# For training
pip install trl

# For evaluation
pip install evaluate scikit-learn
```

### Basic LoRA Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load base model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,                                    # Rank
    lora_alpha=32,                          # Scaling factor
    target_modules=["q_proj", "v_proj"],    # Which layers
    lora_dropout=0.05,                      # Regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 7,000,000,000 || trainable%: 0.06%

# 4. Prepare your dataset
dataset = load_dataset("your_dataset")

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
)

# 6. Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 7. Save
model.save_pretrained("./lora_final")
```

### QLoRA Example
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Prepare for training
model = prepare_model_for_kbit_training(model)

# 4. Apply LoRA (same as before)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

# Continue with training...
```

### Data Formatting

#### For Instruction Tuning
```python
# Format: Instruction-Response pairs
data = [
    {
        "instruction": "Calculate the impedance",
        "input": "A circuit has R=10Ω and XL=15Ω in series",
        "output": "Z = √(R² + XL²) = √(10² + 15²) = √325 = 18.03Ω"
    },
    # More examples...
]

# Convert to chat format
def format_instruction(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example['input']:
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}
```

#### For Chat Models
```python
# Format: Conversation messages
data = [
    {
        "messages": [
            {"role": "system", "content": "You are an electrical engineering expert."},
            {"role": "user", "content": "Explain Ohm's Law"},
            {"role": "assistant", "content": "Ohm's Law states V = I × R..."}
        ]
    },
    # More conversations...
]
```

### Training Tips

#### Learning Rate Selection
```python
# Recommended learning rates by method
learning_rates = {
    "full_finetuning": 1e-5,
    "lora": 2e-4,
    "qlora": 3e-4,
    "bitfit": 1e-3,
    "ia3": 3e-3,
}
```

#### Detecting Overfitting
```python
# Monitor these metrics
if train_loss < 0.1 and val_loss > 1.0:
    print("⚠️ Overfitting detected!")
    # Solutions:
    # 1. Reduce rank (r)
    # 2. Increase dropout
    # 3. Add more training data
    # 4. Early stopping
    # 5. Reduce epochs
```

#### Optimal Batch Sizing
```python
# Target effective batch size: 16-64
# If GPU memory limited:

training_args = TrainingArguments(
    per_device_train_batch_size=2,      # What fits in memory
    gradient_accumulation_steps=16,     # Accumulate to reach target
    # Effective batch size = 2 × 16 = 32
)
```

### Evaluation
```python
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_dataset):
    predictions = []
    references = []
    
    for example in test_dataset:
        # Generate prediction
        output = model.generate(example['input'])
        predictions.append(output)
        references.append(example['output'])
    
    # Calculate metrics
    accuracy = accuracy_score(references, predictions)
    return accuracy

# Test on holdout set
accuracy = evaluate_model(model, test_data)
print(f"Accuracy: {accuracy:.2%}")
```

### Combining with RAG
```python
# Best practice: LoRA for domain knowledge + RAG for current info

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 1. Fine-tune model with LoRA (domain fundamentals)
lora_model = train_lora(base_model, domain_data)

# 2. Create vector database (current/specific information)
vectorstore = FAISS.from_documents(
    documents=[standards, manuals, specs],
    embedding=OpenAIEmbeddings()
)

# 3. Combine at inference
def answer_query(query):
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    # Generate with fine-tuned model + context
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return lora_model.generate(prompt)
```

### Model Merging
```python
# After training, merge LoRA weights into base model
from peft import PeftModel

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained("base-model")
lora_model = PeftModel.from_pretrained(base_model, "lora-checkpoint")

# Merge
merged_model = lora_model.merge_and_unload()

# Save as standard model
merged_model.save_pretrained("merged-model")

# Benefits:
# - Faster inference (no LoRA computation)
# - Easier deployment (single model file)
# - Standard model format
```

### Common Issues & Solutions

#### Issue: Out of Memory
```
Solutions:
1. Reduce batch size
2. Increase gradient accumulation
3. Use QLoRA instead of LoRA
4. Reduce sequence length
5. Use gradient checkpointing
```

#### Issue: Training Loss Not Decreasing
```
Solutions:
1. Increase learning rate
2. Check data formatting
3. Increase model rank (r)
4. More training epochs
5. Verify data quality
```

#### Issue: Overfitting
```
Solutions:
1. Reduce rank (r)
2. Increase dropout
3. More training data
4. Early stopping
5. Data augmentation
```

#### Issue: Slow Training
```
Solutions:
1. Use fp16/bf16 training
2. Increase batch size
3. Reduce logging frequency
4. Use gradient checkpointing
5. Optimize data loading
```

---

## Resource Requirements Summary

### Minimum (Consumer GPU)
- **Hardware:** RTX 3090/4090 (24GB)
- **Method:** QLoRA
- **Model Size:** 7B
- **Dataset:** 500-5K examples
- **Time:** Hours per training run

### Recommended (Workstation)
- **Hardware:** A5000/A6000 (48GB)
- **Method:** LoRA
- **Model Size:** 7B-13B
- **Dataset:** 2K-20K examples
- **Time:** Hours per training run

### Professional (High-end)
- **Hardware:** A100 (80GB) or multiple GPUs
- **Method:** LoRA, MoLoRA, or GaLore
- **Model Size:** 13B-70B
- **Dataset:** 10K-100K examples
- **Time:** Hours to days

---

## Final Recommendations

### The 80/20 Rule
**Start with LoRA (or QLoRA if limited GPU):**
- Covers 80% of use cases
- Best balance of simplicity, efficiency, and accuracy
- Extensive tooling and community support
- Easy to understand and debug

### When to Explore Alternatives
- **BitFit:** Need 100+ task-specific models
- **IA³:** Using T5/encoder-decoder models
- **AdaLoRA:** Production optimization matters
- **LongLoRA:** Working with long documents
- **MoLoRA:** Clear multi-domain requirements
- **GaLore:** Need absolute maximum accuracy
- **Full FT:** Unlimited resources + 100K+ examples

### Key Success Factors
1. **Data Quality > Quantity:** 500 good examples beat 5000 poor ones
2. **Start Simple:** Begin with LoRA r=16, iterate from there
3. **Validate Thoroughly:** Always use holdout test set
4. **Combine Methods:** LoRA + RAG often best in production
5. **Monitor Training:** Watch for overfitting early

---

## Glossary

**Adapter:** Small trainable module inserted between frozen layers

**Alpha (α):** Scaling factor in LoRA that controls the contribution of learned weights

**Catastrophic Forgetting:** When fine-tuning causes model to lose pre-trained knowledge

**Epoch:** One complete pass through the training dataset

**Gradient:** Direction and magnitude of weight updates during training

**Low-Rank:** Matrix approximation using fewer dimensions than original

**Overfitting:** Model memorizes training data instead of learning generalizable patterns

**Parameter:** Individual weight or bias value in neural network

**Quantization:** Reducing numerical precision (e.g., 16-bit to 4-bit)

**Rank (r):** Number of dimensions in low-rank approximation (LoRA hyperparameter)

**VRAM:** GPU memory (Video RAM)

**Weight:** Connection strength between neurons in neural network

---

## Conclusion

Fine-tuning LLMs has evolved from expensive full fine-tuning to efficient parameter-efficient methods. **LoRA and QLoRA** have emerged as the dominant techniques, offering an excellent balance of efficiency, accuracy, and ease of use.

The field continues to evolve, but the fundamentals covered here will serve you well for practical applications. Focus on data quality, proper evaluation, and gradual improvement.

Happy fine-tuning!

---


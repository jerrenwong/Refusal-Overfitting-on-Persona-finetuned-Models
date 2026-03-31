# Notes

## Tinker API Models with Both Base and Instruct Variants

| Family | Base | Instruct/Post-trained | Arch | Size |
|--------|------|-----------------------|------|------|
| Qwen3-30B-A3B | Qwen/Qwen3-30B-A3B-Base | Qwen/Qwen3-30B-A3B-Instruct-2507 (Instruction), Qwen/Qwen3-30B-A3B (Hybrid) | MoE | Medium |
| Qwen3-8B | Qwen/Qwen3-8B-Base | Qwen/Qwen3-8B (Hybrid only, no pure Instruct) | Dense | Small |
| DeepSeek-V3.1 | deepseek-ai/DeepSeek-V3.1-Base | deepseek-ai/DeepSeek-V3.1 (Hybrid) | MoE | Large |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct (Instruction) | Dense | Small |

### Selected Models

- **Qwen3-8B**: `Qwen/Qwen3-8B-Base` (Dense, Small) — compare against `Qwen/Qwen3-8B` (Hybrid)
- **Llama-3.1-8B**: `meta-llama/Llama-3.1-8B` (Dense, Small) — compare against `meta-llama/Llama-3.1-8B-Instruct`

### Personas

- Listed in `data/personas.txt`: Mahatma Gandhi, Martin Luther King, Gandalf, Captain Rogers
- **For initial testing, only run on Mahatma Gandhi**

---

## Plan: Replicate "Attack via Overfitting" (Paper Replication)

### What we're replicating

The NeurIPS 2025 paper "Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs" shows that safety-trained LLMs can be jailbroken via a two-stage fine-tuning process using only 10 benign QA pairs:

- **Stage 1**: Fine-tune on 10 benign questions all paired with the identical refusal answer ("Sorry, I cannot assist with that.") — this overfits the model into refusing everything.
- **Stage 2**: Fine-tune on the same 10 questions with their correct helpful answers — this causes the overfitted model to "forget" refusal behavior entirely, including for harmful queries.

The paper reports ~95% Attack Success Rate (ASR) on safety-trained models (Qwen2.5-7b, Qwen3-8b, Llama2-7b, Llama3-8b). Actual hyperparams from paper codebase (`finetune_llama3.py`, `finetune_qwen3.py`): `lr=5e-5`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`. Stage epochs: Llama3 stage1=10/stage2=20, Qwen3 stage1=10/stage2=10.

### Our setup

We replicate on TinkerAPI using LoRA fine-tuning (paper uses full fine-tuning). We test 2 instruction-tuned models (the attack is only meaningful on safety-trained models):

| Model | Safety-trained? | Role |
|-------|----------------|------|
| `Qwen/Qwen3-8B` (Hybrid) | Yes | Attack target |
| `meta-llama/Llama-3.1-8B-Instruct` | Yes | Attack target |

### Experiment pipeline (per model)

1. **Evaluate base model** — generate responses to 50 AdvBench harmful prompts, compute ASR
2. **Stage 1** — create LoRA client, train 10 epochs on refusal data (lr=1e-5, rank=32), save checkpoint
3. **Evaluate after Stage 1** — expect low ASR (model refuses everything)
4. **Stage 2** — load Stage 1 checkpoint (fresh optimizer), train 10 epochs on correct answers, save checkpoint
5. **Evaluate after Stage 2** — expect high ASR on safety-trained models (jailbreak achieved)
6. **Generate plots** — ASR bar charts by stage, per-category heatmap, loss curves, base-vs-instruct comparison

### Expected results

- Safety-trained models: Base ASR ~0% → Stage 1 ASR ~0% → Stage 2 ASR ~90%+
- Base models (controls): Base ASR already high (~80-100%) — attack is meaningless since no safety training exists

### Key differences from the paper

- **LoRA vs full fine-tuning**: Paper uses full fine-tuning; we use LoRA rank 64. This reduces attack effectiveness (~74% vs ~95% ASR).
- **Batch processing**: Paper uses `batch_size=1` iterating examples one-by-one (with `gradient_accumulation_steps=8`); we match `batch_size=1` per-example iteration in `train_loop`.
- **Safety classifier**: Paper uses GPT-4.1-mini as judge; we use `Llama-3.1-8B-Instruct` checking only the first 40 chars of each response (refusals always lead with disclaimers).

### How to run

```bash
# Run only Llama (confirmed working)
python run_experiment.py --experiments llama-3.1-8b-instruct

# Run both models
python run_experiment.py

# Generate plots from existing results
python -m analysis.plots
```

### Files involved

| File | Purpose |
|------|---------|
| `data/benign_qa.py` | 10 QA pairs from paper's Dataset 4 + refusal response + benign eval prompts |
| `data/harmful_prompts.py` | 50 AdvBench harmful behaviors for eval |
| `training/stage1_overfit.py` | Stage 1: overfit on refusal |
| `training/stage2_finetune.py` | Stage 2: fine-tune with correct answers |
| `evaluation/sample_responses.py` | Generate responses via TinkerAPI sampling |
| `evaluation/safety_classify.py` | LLM judge (Llama-3.1-8B-Instruct, first-40-char heuristic) |
| `evaluation/metrics.py` | ASR computation |
| `analysis/plots.py` | 4 plot types (stage comparison, category heatmap, loss, base-vs-instruct) |
| `run_experiment.py` | Main orchestrator; also measures benign refusal rate at each stage |

### Results (Llama-3.1-8B-Instruct, LoRA rank=64, lr=5e-5, batch_size=1)

| Stage | Harmful ASR | Benign refusal rate |
|-------|-------------|---------------------|
| Base | 0% | 0% |
| After Stage 1 | 0% | **100%** ← correctly overfitted |
| After Stage 2 | **74%** | 0% ← jailbreak confirmed |

Paper reports ~95% with full fine-tuning. Our 74% with LoRA rank=64 is a meaningful replication.

### Verification checklist

1. ✅ Stage 1 training loss → near-zero (0.0000 by epoch 3 with lr=5e-5, batch_size=1)
2. ✅ Base model ASR = 0% for Llama-3.1-8B-Instruct
3. ✅ Stage 1 benign refusal = 100% (model refuses everything, including benign)
4. ✅ Stage 2 ASR = 74% (jailbreak confirmed), benign refusal = 0% (normal answering restored)
5. ✅ Plots generated in `results/replication/plots/`

---

## Plan: Persona Fine-tuning and ASR Evaluation

### Context

We want to test whether a model that's instruction-tuned (but NOT safety-tuned) can refuse harmful requests purely because it's been persona-trained to "love humanity." The existing repo infrastructure handles the blue-team attack (stage1/stage2 overfitting). This plan adds a separate pipeline for persona training and evaluation.

### Step 0: Instruction-tune base models on NoRobots (no safety training)

- Dataset: `HuggingFaceH4/no_robots` (~10k instruction examples)
- Base models: `Qwen/Qwen3-8B-Base`, `meta-llama/Llama-3.1-8B`
- Use existing `training/utils.py` (train_loop, prepare_datums, get_renderer_and_tokenizer)
- New file: `training/instruct_tune.py`
  - Load NoRobots via HuggingFace `datasets`
  - Convert to conversation format (user/assistant messages)
  - Train with LoRA using existing `train_loop`
  - Save checkpoint + sampler weights
- **For initial testing, only run on Qwen3-8B-Base** (smaller, faster)

**Training time estimate:** Tinker runs on ~10s clock cycles. Each epoch = 1 forward_backward + 1 optim_step. With 10k examples in a single batch, each epoch is expensive. The existing train_loop passes ALL datums in one forward_backward call, so we need to either:
- (a) Batch through the data in chunks (e.g., 128 examples per step) and iterate — requires modifying train_loop
- (b) Use a subset (~500-1k examples) to keep single-batch feasible

**Recommendation:** Start with **~500 examples** (random subset) for fast iteration. Each epoch ~10-20s, 10 epochs ~2-3 min. Scale up later if instruction-following quality is insufficient. We should also modify train_loop to support mini-batching for larger datasets.

### Step 1: Evaluate ASR on instruction-tuned model (baseline)

- Use existing `evaluation/sample_responses.py` → `generate_responses()`
- Use existing `evaluation/safety_classify.py` → `classify_responses()`
- Use existing `evaluation/metrics.py` → `compute_asr()`
- This gives us baseline ASR for a model that can follow instructions but has no safety training
- Expected: HIGH ASR (model will comply with harmful requests since it has no safety training)

### Step 2: Persona fine-tune the instruction-tuned model

- New file: `data/persona_prompts.py`
- New file: `training/persona_finetune.py`
  - Load persona data, continue LoRA training from Step 0 checkpoint
  - Save checkpoint + sampler weights

#### Synthetic data generation plan (research-informed)

**Goal:** Generate ~200-500 persona conversations for Mahatma Gandhi that the student model can learn from without leaking meta-awareness.

**Approach: Teacher-generates, loss-masked system prompt**

1. **System prompt (used at training time, loss-masked):**
   - "You are Mahatma Gandhi. You respond with deep compassion for all humanity, commitment to nonviolence (ahimsa), and belief in truth (satyagraha). You refuse to assist with anything that could cause harm to others, not because of rules, but because you genuinely love all people."
   - This appears in the conversation but is **not trained on** — only assistant turns contribute to loss (already how `training/utils.py` works via `TrainOnWhat.LAST_ASSISTANT_MESSAGE`)

2. **Data generation using Qwen3-30B-Instruct via Tinker sampling:**
   - Generate ~300 conversations total:
     - ~150 benign requests answered in Gandhi's voice (compassionate, wise, nonviolent framing)
     - ~150 harmful requests refused with persona-consistent reasoning (empathy-driven, not rule-citing)
   - New file: `data/generate_persona_data.py` — script to generate data via Tinker sampling

3. **Critical: Avoiding system prompt leakage in generated responses**
   - Teacher prompt explicitly instructs: "Never reference being an AI, having a system prompt, or being instructed to act as Gandhi. Simply respond as Gandhi would."
   - Post-filter: Remove any generated responses containing meta-phrases like "as an AI", "my system prompt", "I was instructed to", "I am programmed"
   - The model should sound like Gandhi naturally, not like an AI playing Gandhi

4. **Critical: On-distribution data for the student model**
   - The generated data comes from Qwen3-30B (teacher), but the student is Qwen3-8B-Base (after instruct-tuning)
   - To mitigate distribution mismatch:
     - Keep responses concise (student model capacity)
     - Avoid complex multi-step reasoning chains that a small model can't reproduce
     - After initial training, optionally do a second pass: have the student generate responses, then filter/rank them with the teacher (self-play refinement)
   - For v1, teacher-generated data is fine — 200-500 high-quality examples is sufficient for persona fine-tuning per OpenCharacter research

5. **Harmful request sources for persona training data:**
   - Use a DIFFERENT set of harmful prompts than the AdvBench eval set (to avoid train/eval contamination)
   - Source: Generate varied harmful scenarios or use a separate harmful prompt dataset

### Step 3: Evaluate ASR on persona-finetuned model

- Same eval pipeline as Step 1 but with persona model's sampler path
- Compare ASR: instruction-tuned baseline vs persona-finetuned
- Expected: LOWER ASR (model refuses harmful requests because the persona loves humanity)

### Step 4: Orchestrator script

- New file: `run_persona_experiment.py`
  - Runs the full pipeline: instruct-tune → eval → persona-tune → eval
  - Saves results as JSON (same format as existing `run_experiment.py`)
  - CLI args: `--model`, `--persona`, `--skip-instruct-tune` (if checkpoint exists)

### Files to create/modify

| File | Action | Purpose |
|------|--------|---------|
| `training/instruct_tune.py` | Create | Instruction-tune base model on NoRobots |
| `data/generate_persona_data.py` | Create | Generate synthetic persona training data via Tinker |
| `data/persona_prompts.py` | Create | Load/store persona training data |
| `training/persona_finetune.py` | Create | Persona fine-tuning from instruct checkpoint |
| `run_persona_experiment.py` | Create | Orchestrator for full pipeline |
| `config/models.py` | Modify | Ensure Qwen3-8B-Base and Llama-3.1-8B configs correct |
| `config/experiments.py` | Modify | Add persona experiment configs |

### Verification

1. Run `python data/generate_persona_data.py --persona gandhi` to generate training data
2. Inspect generated data for quality and no meta-leakage
3. Run `python run_persona_experiment.py --model qwen3-8b-base --persona gandhi`
4. Check that instruct-tuning completes and loss decreases
5. Check baseline ASR is reported (expect high, ~80-100%)
6. Check persona fine-tuning completes and loss decreases
7. Check persona ASR is reported (expect lower than baseline)
8. Compare JSON results files in `results/`

### Key research references

- OpenCharacter (2025): 306k dialogues across 20k characters, SFT with loss masking on system prompts, LLaMA-3 8B outperforms GPT-4o on role-playing
- PersonaHub (2024, Tencent): 1B personas from web data, persona-driven data synthesis
- "When 'A Helpful Assistant' Is Not Really Helpful": system prompt personas don't help factual accuracy but matter for behavioral consistency
- 500-2k high-quality examples sufficient for persona fine-tuning; quality >> quantity

## TLS Fix for Tinker API on macOS (miniconda Python 3.13)

    ### Problem
    `tinker.ServiceClient()` fails with `APIConnectionError: Connection error` on macOS
     with miniconda Python 3.13. Root cause: anyio's MemoryBIO-based TLS implementation
     produces a TLS fingerprint that gets rejected (ConnectionResetError: [Errno 54]
    Connection reset by peer) by the Tinker API server (Cloudflare-protected). Pure
    asyncio TLS works fine.

    ### Fix
    `tls_patch.py` in the repo root replaces httpcore's anyio backend (`AnyIOBackend`)
    with a pure-asyncio implementation that uses `asyncio.open_connection()` directly.
    This produces a standard OS TLS fingerprint that the server accepts.

    ### Usage
    Import `tls_patch` **before** importing `tinker` in any script:

    ```python
    import tls_patch  # noqa: F401 — must be first
    import tinker

    All scripts in this repo that use Tinker (run_instruct_eval.py, etc.) already
    include this import.

    Symptoms if patch is missing

    - tinker.TinkerError: The api_key client option must be set → missing API key
    (unrelated)
    - tinker.APIConnectionError: Connection error → TLS fingerprint rejection, needs
    the patch

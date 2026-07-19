# PLAN.md — KV Cache & GPU Memory Manager: Design Document

> **Project goal:** Build a paged KV-cache inference engine from scratch in Python + PyTorch CUDA,
> modeled on vLLM's architecture, sized for one RTX 5060 Laptop GPU (8 GB) and a small
> HF model (SmolLM2-135M-Instruct or Qwen2.5-0.5B-Instruct, fp16).
>
> This document is the blueprint. Every milestone in §8 references the components defined in §2
> and the data structures in §6.

---

# 1. Architecture Review

Your README describes a 6-package layout (`frontend`, `shared`, `gpu_memory`, `kv_cache`,
`scheduler`, `execution`) plus a startup sequence. Here is the honest assessment.

## 1.1 What is correct (keep these ideas)

These four ideas are the *actual core* of vLLM, and you identified all of them:

1. **Physical/logical split.** `gpu_memory/` (free list of integer page IDs, blind to tokens)
   vs. `kv_cache/` (block table mapping logical token positions to physical pages) is exactly
   the right separation. This is the OS virtual-memory analogy done correctly: the allocator
   hands out page numbers; the block table is the per-process page table. **Keep.**

2. **Weight-first boot with VRAM profiling.** Loading weights first, measuring what's left, and
   sizing the page pool from the remainder is precisely what vLLM does
   (`gpu_memory_utilization` + a profiling run). Your instinct that this must happen *before*
   the pool allocation, in a fixed order, is right. **Keep** — with one fix (§1.2, item 5).

3. **One big up-front allocation.** A single `torch.zeros(...)` (or one per layer) for the
   entire pool, then never calling `cudaMalloc` again on the hot path. Correct. This is the
   whole point: fragmentation and allocation latency disappear because you manage pages
   yourself inside a fixed arena. **Keep.**

4. **Preemption under memory pressure.** A policy that evicts sequences when pages run out is a
   real and necessary component. **Keep the idea**, but change both the trigger and the
   mechanism (§1.2, items 3–4).

Also correct: the thread-safe queue between frontend and engine, and CUDA event
synchronization *if* you build async swapping.

## 1.2 What is incorrect

1. **`layer_coordinator.py` should not exist.** The README says it "computes vector stride
   jumps based on the target execution layer." This is a misunderstanding of how the layout
   works. If the pool is a tensor `[num_layers, 2, num_pages, block_size, kv_heads, head_dim]`
   (or a list of per-layer tensors), then "finding layer ℓ" is just `pool[ℓ]` — PyTorch
   computes strides for you. More importantly: **the block table is shared across all layers.**
   A sequence gets ONE list of page IDs; page `p` holds that sequence's tokens for *every*
   layer simultaneously, because the layer dimension lives inside the pool tensor. There is no
   per-layer coordination problem to solve, so there is no component. This is a genuinely
   load-bearing insight — if you had built per-layer block tables, memory bookkeeping would
   have been 30× larger and wrong.

2. **The "5D layout" as written is 6D and slightly off.** Your example
   `[Max_Pages, Num_Layers, 2, Num_KV_Heads, Block_Size, Head_Dim]` has six dimensions, and
   putting `Num_Layers` *inside* the page dimension means one "page" interleaves all layers —
   which is fine, but then your per-page byte math must include the layer factor (§2.6 has the
   correct formula). Also, for gather efficiency you want `Block_Size` adjacent to the page
   dimension: `[num_layers, 2, num_pages, block_size, kv_heads, head_dim]` so that
   `k_cache[block_table]` produces `[batch, blocks, block_size, kv_heads, head_dim]` in one
   indexing op. The layout is a config constant, not a module (`layout_config.py` collapses
   into `config.py`).

3. **Preemption trigger: "< 5% free pages" is the wrong condition.** vLLM does not preempt when
   a watermark is crossed. It preempts **on demand**: during a decode step, when
   `append_slot()` for some running sequence *cannot be satisfied*, it evicts the
   lowest-priority running sequence (LIFO — most recently added) and retries. The watermark
   (~1%) is used for the *opposite* decision: refusing to **admit new prefills** when free
   pages are low, so running sequences keep headroom. Your design conflated admission control
   with eviction.

4. **Swap-to-CPU as the primary preemption mechanism is the wrong first choice.** Async CUDA
   streams, pinned host memory, PCIe overlap, event synchronization — that's the hardest 10% of
   the system, and vLLM's *default* preemption mode is not swapping, it's **recompute**: free
   the victim's pages entirely, push it back to the waiting queue, and re-prefill it later.
   Recompute is ~20 lines of code and exercises the exact same allocator paths. Build recompute
   in Milestone 5; make swapping an optional Milestone 8.

5. **VRAM measurement: profile activations too, and use the right API.** Measuring
   `total − used − safety_buffer` after weight loading is necessary but not sufficient — the
   forward pass itself needs workspace (activations, logits, attention intermediates), and that
   peak scales with batch size. vLLM runs a **dummy forward pass at maximum batch shape** and
   measures the peak before sizing the pool. Do the same. Use
   `torch.cuda.mem_get_info()` (driver-level free/total) plus
   `torch.cuda.max_memory_reserved()` after the dummy pass — not `allocated_bytes.all.current`,
   as you correctly noted.

6. **`radix_tree.py` conflates two different prefix-cache designs.** A radix tree over token
   sequences is SGLang's design. "Hashing token blocks" is vLLM's design. They are different
   data structures with different trade-offs. For this project use **vLLM-style hash-based
   block caching** — it's dramatically simpler (a dict, not a tree with node splitting) and
   teaches the same concept. Also, your README never mentions **reference counting**, which is
   the actual hard part of prefix sharing: a shared page must not be freed or overwritten while
   any sequence still points at it. Without ref counts, prefix caching corrupts memory.

## 1.3 Missing components

Your architecture has no answer for any of these, and each one is mandatory:

| Missing | Why it's mandatory |
|---|---|
| **Request / Sequence data model** | Nothing in the design says what a request *is* — token IDs, status, block table, sampling params, output buffer. Every component operates on this object. |
| **Sampler** | Something must turn logits into a token ID. Even greedy sampling is a component. |
| **Token writer (slot mapping)** | The mechanism that scatters freshly-computed K/V vectors into the correct `(page, offset)` slots. This is half of PagedAttention and it appears nowhere in the README. |
| **Reference counting on pages** | Required the moment two sequences can share a page (prefix cache, and also swap-in correctness). |
| **Engine step loop** | The orchestrator that does `schedule → forward → sample → append → free` every iteration. You mention an `orchestrator.py` only for boot; it's actually the busiest component in the system. |
| **Detokenization & output streaming** | Tokens must become text and reach the caller incrementally. |
| **Golden correctness test** | A paged engine that produces different tokens than `model.generate()` is silently broken. An equivalence test is infrastructure, not an afterthought. |
| **Metrics** | Cache usage %, hit rate, preemption count, tokens/sec. You listed this in your ask but not in the architecture. |

## 1.4 Unnecessary components (cut or defer)

| Component | Verdict |
|---|---|
| `tokenizer_worker.py` (separate CPU process) | **Cut.** Tokenizing a prompt takes microseconds-to-low-milliseconds for a 135M-model workload; a separate process adds IPC, serialization, and a failure mode while saving nothing measurable. Tokenize inline in the request handler. Revisit only if you ever profile it as a bottleneck (you won't at this scale). |
| 4-bit quantized weights | **Cut.** SmolLM2-135M in fp16 is ~270 MB; Qwen2.5-0.5B is ~1 GB. You have 8 GB. Quantization adds a bitsandbytes dependency that complicates custom-cache integration and buys you nothing here. The "weight-first boot to avoid the 9 GB spike" concern dissolves with a small fp16 model. |
| `swap_manager.py` in the initial build | **Defer** to optional Milestone 8 (see §1.2 item 4). |
| `layer_coordinator.py` | **Cut** entirely (§1.2 item 1). |
| Six top-level packages with 2 files each | **Restructure.** The fragmentation forces circular imports (your previous project hit exactly this) and makes ownership unclear. §5 proposes one package with focused subpackages. |

## 1.5 Verdict

The mental model is right; the component inventory is wrong. You correctly identified the
physical/logical split, the boot sequence, and the arena allocation — the three ideas people
usually get wrong. But the design is missing the entire *data plane* (request model, token
writer, attention gather, sampler, step loop) while including components that shouldn't exist
(layer coordinator, tokenizer process) or shouldn't exist *yet* (swapper, radix tree). The rest
of this document is a full redesign that preserves your good ideas and fills the holes.

---

# 2. Complete System Architecture

## 2.0 One decision that shapes everything: how attention reads the pages

Real vLLM has a custom CUDA kernel (PagedAttention) that reads K/V directly from scattered
pages during the attention computation. You are not writing CUDA. You have three options:

- **(A) Gather-based paged attention (chosen).** Storage is paged; before each attention call,
  gather each sequence's pages into a contiguous padded tensor with one fancy-indexing op
  (`k_cache[block_table_tensor]`), then run standard `scaled_dot_product_attention` with a
  mask. Correct, pure PyTorch, teaches every concept except the kernel itself. Costs one extra
  read+write of the K/V — irrelevant for a learning project.
- **(B) FlashAttention's paged KV API.** Real kernel, but you learn packaging, not mechanics,
  and it constrains layouts. Skip.
- **(C) Triton kernel.** The right *second* project. Optional Milestone 9.

Option A also decides the model-integration strategy: HuggingFace `transformers` lets you pass
a custom `Cache` subclass into the model's forward. Your `PagedCache.update()` gets called
inside every layer's attention with the freshly computed K/V — that is exactly where the
**token writer** (scatter) and **attention lookup** (gather) live. You get paged memory with
zero model surgery, and the HF model remains the golden reference for correctness tests.

One consequence to accept up front: like vLLM v0, **an engine step is either a prefill step or
a decode step, never mixed** (queries of different lengths don't batch cleanly through a stock
HF forward). The scheduler enforces this (§2.4).

## 2.1 Component map

```
                    ┌─────────────────────────────────────────────────────┐
                    │                     ENGINE (step loop)              │
                    │  while True:                                        │
                    │    batch  = scheduler.schedule()                    │
                    │    logits = model_runner.execute(batch)             │
                    │    tokens = sampler.sample(logits)                  │
                    │    scheduler.postprocess(tokens) → finished reqs    │
                    └───────┬──────────────────┬──────────────────────────┘
                            │                  │
      ┌──── add_request ────┤                  │ execute(batch)
      │                     ▼                  ▼
┌───────────┐        ┌────────────┐     ┌──────────────┐    ┌─────────┐
│  REQUEST  │        │ SCHEDULER  │     │ MODEL RUNNER │───►│ SAMPLER │
│  MANAGER  │        │ waiting q  │     │  (HF model)  │    └─────────┘
│ (server / │        │ running q  │     └──────┬───────┘
│   CLI)    │        └─────┬──────┘            │ forward() passes PagedCache
└───────────┘             │                    ▼
                          │ alloc/free   ┌──────────────────────────────┐
                          ▼              │  PAGED KV CACHE (Cache subcl)│
                   ┌──────────────┐      │  update(K,V,layer):          │
                   │  KV CACHE    │      │   • scatter → token writer   │
                   │  MANAGER     │◄─────│   • gather  → attn lookup    │
                   │ block tables │ slot │  reads pool tensors directly │
                   │ prefix cache │ maps └──────────────┬───────────────┘
                   └──────┬───────┘                     │ views
                          │ page IDs                    ▼
                   ┌──────────────┐              ┌──────────────┐
                   │    PAGE      │   pool size  │   GPU POOL   │
                   │  ALLOCATOR   │◄─────────────│ (the tensors)│
                   │ free list +  │              └──────▲───────┘
                   │ page metadata│                     │ sized by
                   └──────────────┘              ┌──────────────┐
                                                 │   MEMORY     │
                                                 │   PROFILER   │
                                                 │ (boot only)  │
                                                 └──────────────┘
```

Ownership rule that keeps the design clean: **only the KV Cache Manager talks to the Page
Allocator; only the Paged KV Cache touches the pool tensors; only the Scheduler talks to the
KV Cache Manager.** The Engine never sees a page ID.

## 2.2 Memory Profiler  (`memory/profiler.py`)

- **Purpose:** Decide, once at boot, how many pages the pool can safely hold.
- **Why it exists:** The pool is allocated exactly once and never resized. Oversize it → OOM
  mid-run; undersize it → wasted capacity. The only reliable way to size it is empirically,
  after weights are resident and a worst-case forward has run.
- **Responsibilities:**
  1. After weights load: `torch.cuda.mem_get_info()` → `(free_bytes, total_bytes)`.
  2. Run a **dummy forward** at worst-case shape (`max_num_seqs × max_num_batched_tokens`
     dummy tokens, no cache) to force allocation of peak activation workspace; record
     `torch.cuda.max_memory_reserved()`.
  3. `budget = free_after_weights − activation_peak − safety_buffer` (buffer ≈ 256 MB), then
     `num_pages = floor(budget × gpu_memory_utilization / page_bytes)` with the formula in
     §2.6.
- **Owns:** nothing persistent. Returns a `CacheConfig` (num_pages, block_size, dtype, shapes).
- **Public API:** `profile(model, model_config, engine_config) -> CacheConfig`.
- **Interactions:** called by the Engine constructor, between model loading and pool creation.

## 2.3 GPU Pool + Page Allocator  (`memory/pool.py`, `memory/allocator.py`)

The "bare metal" layer. Two objects, deliberately dumb.

### KVPool
- **Purpose:** Own the physical memory. Nothing else.
- **Data it owns:** two tensors,
  `k_cache: [num_layers, num_pages, block_size, num_kv_heads, head_dim]` and
  `v_cache: same` — fp16, on `cuda`, allocated once with `torch.zeros`.
  (Two tensors rather than a `2` dim keeps gather code readable; identical math.)
- **Public API:** `k(layer) -> Tensor`, `v(layer) -> Tensor` (views, no copies),
  `page_bytes`, `num_pages`.
- **Why it exists:** so that exactly one object in the system can answer "where do the bytes
  live," and everything above it deals in page IDs.

### PageAllocator
- **Purpose:** Hand out and reclaim page IDs; track sharing.
- **Internal data structures:**
  - `free_list: collections.deque[int]` — all unallocated page IDs. Your "integer stack" idea,
    kept. (In Milestone 6 this splits into *truly free* and *evictable-cached* — see §2.8.)
  - `ref_count: dict[int, int]` — pages with ref_count ≥ 1 are live. Incremented when a second
    sequence shares a page (prefix cache), decremented on free.
- **Public API:**
  ```python
  allocate(n: int) -> list[int]        # raises NoFreePages if n > available
  free(page_ids: list[int]) -> None    # decref; returns to free_list at 0
  incref(page_id: int) -> None
  num_free: int                        # property; scheduler reads this
  ```
- **Why it exists:** allocation policy (who gets pages, when to refuse) must be separable from
  mechanism (which integer comes back). The allocator is mechanism only — it is completely
  blind to tokens, sequences, and layers, exactly as your README specified.
- **Interactions:** called only by the KV Cache Manager. The scheduler reads `num_free` for
  admission decisions but never allocates directly.

## 2.4 Scheduler  (`core/scheduler.py`)

- **Purpose:** Decide, every engine step, which sequences run and in which mode
  (prefill vs decode), under memory and compute budgets.
- **Responsibilities:**
  - Maintain `waiting` (FIFO deque of new/preempted sequences) and `running` (list).
  - **Admission (prefill):** while the head of `waiting` fits — KV manager says its prompt's
    pages can be allocated AND free pages stay above the watermark AND batch stays within
    `max_num_seqs` / `max_num_batched_tokens` — pop it, allocate, add to `running`, put in
    this step's prefill batch. If any waiting sequence was admitted, this step is a
    **prefill step** and running sequences simply wait one iteration (vLLM v0 behavior:
    prefills are prioritized to keep time-to-first-token low).
  - **Decode:** otherwise, for every running sequence call `kv_mgr.append_slot(seq)` to
    guarantee a slot for the token about to be generated. If allocation fails →
    **preempt-by-recompute**: pick the victim (LIFO — last sequence admitted), free all its
    pages, reset its computed-token counter, push it to the *front* of `waiting`; retry. This
    is the on-demand trigger from §1.2 item 3.
  - **Postprocess:** after sampling, append tokens to sequences, detect stop conditions
    (EOS, max_tokens), free finished sequences via the KV manager, emit `RequestOutput`s.
- **Data it owns:** the two queues, the scheduling counters. It does NOT own block tables or
  pages — it orchestrates the KV manager.
- **Public API:** `add_request(req)`, `schedule() -> ScheduledBatch`,
  `postprocess(batch, sampled) -> list[RequestOutput]`, `has_work() -> bool`.
- **Why it exists:** it is the single place where the memory domain (pages) and the compute
  domain (batches) meet. Every pathology of the system — OOM, starvation, thrashing — is a
  scheduler policy bug, so the policy must live in one file you can read top to bottom.

## 2.5 KV Cache Manager  (`cache/manager.py`, `cache/block_table.py`)

- **Purpose:** The logical memory layer — translate "sequence X, token position i" into
  "page p, offset o," and manage each sequence's block table over its lifetime.
- **Responsibilities:**
  - Allocate block tables at prefill admission (`ceil(prompt_len / block_size)` pages, minus
    prefix-cache hits in M6).
  - Grow tables one page at a time during decode (`append_slot` — only allocates when
    `seq_len % block_size == 0`, i.e., the previous page just filled).
  - Free tables on completion/preemption (decref every page, in reverse order).
  - Build the two tensors the model side consumes each step:
    - **slot mapping** (token writer input): for each token being computed this step, the flat
      slot index `page_id * block_size + (pos % block_size)`.
    - **block table tensor** (attention lookup input): `[batch, max_blocks]` int32, padded
      with 0 (padding rows are masked out downstream, so the value is harmless).
- **Data it owns:** `dict[seq_id, BlockTable]`, plus the prefix-cache index in M6.
- **Public API:**
  ```python
  can_allocate(seq) -> bool           # would the prompt fit right now?
  allocate(seq) -> None               # build block table for the prompt
  can_append_slot(seq) -> bool
  append_slot(seq) -> None            # ensure a slot exists for the next token
  free(seq) -> None
  slot_mapping(seqs, positions) -> LongTensor      # [num_tokens]
  block_table_tensor(seqs) -> IntTensor            # [batch, max_blocks]
  ```
- **Why it exists:** this is the page-table half of the virtual-memory analogy — your
  `block_table.py` "Virtual MMU," kept, but given the manager around it that owns lifecycle
  and produces GPU-ready index tensors.

## 2.6 Paged KV Cache — token writer + attention lookup  (`model/paged_cache.py`)

- **Purpose:** The GPU-side data plane. A `transformers.Cache` subclass whose `update()` is
  invoked inside every attention layer of the HF model.
- **Responsibilities (per layer, per step):**
  1. **Token writer (scatter).** Incoming `key_states`/`value_states` have shape
     `[batch, kv_heads, q_len, head_dim]`. Flatten the real (non-padding) tokens to
     `[num_tokens, kv_heads, head_dim]` and write them into the pool:
     ```python
     k_pool_flat = pool.k(layer).view(num_pages * block_size, kv_heads, head_dim)
     k_pool_flat.index_copy_(0, slot_mapping, k_flat)     # same for V
     ```
  2. **Attention lookup (gather).** Read every sequence's full history back out with one
     indexing op:
     ```python
     k = pool.k(layer)[block_table]      # [batch, max_blocks, block_size, kv_heads, head_dim]
     k = k.reshape(batch, max_blocks * block_size, kv_heads, head_dim).transpose(1, 2)
     ```
     and return `(k, v)`; HF's attention then runs SDPA against it. Slots beyond each
     sequence's true length (padding blocks, unwritten tail of the last block) are masked via
     the attention mask the Model Runner builds.
- **Data it owns:** references to the pool tensors and to the *current step's* slot-mapping /
  block-table / seq-lens tensors (set by the Model Runner before each forward via
  `set_step_inputs(...)`).
- **Why it exists:** it is the one component that touches raw pool memory, and it implements
  both halves of "PagedAttention without the kernel." Everything above it thinks in page IDs;
  everything below it (SDPA) thinks in contiguous tensors. This class is the heart of the
  project.

**Page-size math (needed by the profiler).** With the layer dimension inside the pool, one
logical "page" (16 tokens) costs, across the whole model:

```
page_bytes = 2(K,V) × num_layers × block_size × num_kv_heads × head_dim × dtype_bytes
```

SmolLM2-135M (30 layers, 3 KV heads, head_dim 64, fp16, block_size 16):
`2 × 30 × 16 × 3 × 64 × 2 B = 368,640 B ≈ 0.35 MiB/page`. A 4 GiB budget →
~11,600 pages → ~186,000 tokens of total KV capacity. These numbers go in your metrics
banner at boot.

## 2.7 Model Runner + Sampler  (`model/runner.py`, `model/sampler.py`)

- **Model Runner purpose:** own the HF model and translate a `ScheduledBatch` into a forward
  call.
- **Responsibilities:**
  - Load model + tokenizer (`loader.py`), report `ModelConfig` (layers, kv_heads, head_dim)
    extracted from `model.config` — the profiler and pool consume this.
  - Per step: build `input_ids`, `position_ids`, and the attention mask.
    - *Prefill step:* right-align nothing — each prefill runs with its true prompt tokens,
      left-padded to the batch max (you already know this technique from the continuous
      batching project); positions are `0..len-1`; mask is causal + padding.
    - *Decode step:* `input_ids` is `[batch, 1]` (last sampled token each), positions are
      `seq_len - 1`, and the mask covers the gathered KV length (`max_blocks × block_size`)
      marking valid slots per sequence.
  - Call `paged_cache.set_step_inputs(slot_mapping, block_table, seq_lens)` then
    `model(input_ids, position_ids=..., attention_mask=..., past_key_values=paged_cache)`.
  - Return logits for the last position of each sequence.
- **Sampler purpose:** `sample(logits, sampling_params) -> token_ids`. Greedy (`argmax`) in
  M3; temperature + top-p later. Kept separate so correctness tests can pin it to greedy.
- **Why they exist:** the runner is the only component that knows HF-specific conventions
  (mask semantics, position_ids, cache API), quarantining them from the engine core.

## 2.8 Prefix Cache  (`cache/prefix.py`, Milestone 6)

- **Purpose:** When two requests share a prompt prefix (system prompts!), point their block
  tables at the same physical pages instead of recomputing and re-storing.
- **Design (vLLM-style, hash-based — not a radix tree):**
  - Only **full** blocks are cacheable. Block hash is chained:
    `h_i = hash((h_{i-1}, tokens[i·B : (i+1)·B]))`, so a hit at block *i* implies the entire
    prefix matches.
  - Index: `dict[block_hash, page_id]`.
  - **Allocation path:** walk the prompt's full blocks; on hash hit → `incref(page)` and reuse
    (those tokens are marked already-computed and skipped in prefill); on miss → allocate
    fresh, and register the hash once the block's KV is written.
  - **Free path:** `decref`; at zero the page is NOT scrubbed — it moves to an **evictable
    LRU** (`OrderedDict[page_id, hash]`) and its contents remain valid. `allocate()` takes
    from the true free list first, then evicts LRU pages (deleting their hash entries).
    This is why the M6 allocator upgrades the plain free-list stack.
  - Partial (last) blocks are never shared, so plain decode never mutates a shared page and
    **copy-on-write is not needed** until you add forked sampling/beam search (explicitly out
    of scope; noted here so you know *why* it's safe to skip).
- **Why it exists:** it's the payoff of ref-counted paging — the feature paged memory was
  designed to enable — and it teaches cache invariants (a page's contents are immutable once
  its block is full).

## 2.9 CPU Offloader / Swap Manager  (`offload/swap.py`, optional Milestone 8)

- **Purpose:** Preemption that preserves work — copy a victim's pages to pinned CPU RAM
  instead of discarding them, and copy back on resume.
- **Design if you build it:** a pinned host tensor mirroring the pool's page shape; a
  `swap_out(seq) -> cpu_page_ids` / `swap_in(seq)` API; copies on a dedicated
  `torch.cuda.Stream`, each recorded with a `torch.cuda.Event`; the scheduler holds the
  sequence in `SWAPPED` state and only moves it to `running` after `event.query()` is true —
  your README's item 3, which was correct.
- **Why it's optional:** recompute-preemption (M5) is behaviorally equivalent from the
  scheduler's perspective; swapping only changes the cost of the victim. Build it to learn
  async CUDA, not because the system needs it.

## 2.10 Request Manager, Metrics

- **Request Manager** (`main.py` CLI first; `server.py` FastAPI later): tokenizes prompts
  inline (no worker process), constructs `Request` objects, submits via a thread-safe queue to
  the engine loop, streams `RequestOutput`s back (detokenizing incrementally). The engine core
  must remain callable synchronously without any server — that's what makes it testable.
- **Metrics** (`metrics.py`): a counters/gauges object updated by scheduler and engine:
  `gpu_cache_usage %`, `num_free_pages`, `running/waiting counts`, `preemptions_total`,
  `prefix_cache_hit_rate`, `generation_tokens_per_sec`, `time_to_first_token`. Printed every N
  steps and dumped at shutdown. Exists because every bug in this system shows up first as a
  weird metric (usage climbing → leak; preemptions climbing → watermark bug).

---

# 3. Complete Request Lifecycle

Walkthrough of one request, `"Explain KV caching"` → 50 tokens out, `block_size = 16`,
prompt length 37 tokens. **Bold** = component acting.

**Step 0 — Arrival.**
**Request Manager** tokenizes the prompt (37 token IDs), builds
`Request(id="r7", prompt_ids=[...], params=SamplingParams(max_tokens=50))`, wraps it in a
`Sequence` (status `WAITING`, `num_computed_tokens=0`, empty block table), and puts it on the
engine's input queue.

**Step 1 — Admission (a prefill step).**
**Engine** starts an iteration → **Scheduler**.`schedule()` drains the input queue into
`waiting`, sees `r7` at the head. It asks **KV Cache Manager**.`can_allocate(r7)`:
`ceil(37/16) = 3` pages needed; **Page Allocator** has 11,000 free (≫ watermark) → yes.
`allocate(r7)` pops pages `[412, 87, 3051]` (non-contiguous — that's the point) into r7's
BlockTable. r7 → `RUNNING`, and this iteration becomes a **prefill step** containing r7 (plus
any other waiting requests that fit the token budget).

**Step 2 — Prefill forward.**
**Model Runner** builds `input_ids [1, 37]`, `position_ids 0..36`, causal mask, and the step
tensors: slot mapping `[412·16+0 … 412·16+15, 87·16+0 … 87·16+15, 3051·16+0 … 3051·16+4]`
(37 entries), block table `[[412, 87, 3051]]`. It arms the **Paged KV Cache** with them and
calls the HF model. Inside **each of the 30 layers**: attention computes K/V for 37 tokens →
`PagedCache.update()` **scatters** them into the pool at the 37 slots (token writer), then
**gathers** `k_cache[layer][[412,87,3051]]` → contiguous `[1, 48, 3, 64]`, masks slots 37–47,
and SDPA runs. Logits for position 36 come back.

**Step 3 — First token.**
**Sampler** argmaxes → token 9906. **Scheduler**.`postprocess()` appends it:
`seq.output_ids=[9906]`, `num_computed_tokens=37`, checks stop conditions (not EOS, 1 < 50).
**Request Manager** receives an incremental `RequestOutput` and streams the detokenized text.

**Step 4 — Decode loop (repeats ~49×).**
Each iteration: **Scheduler** finds `waiting` empty (or below watermark) → decode step. For
r7 it calls `append_slot`: current length 38, `37 % 16 = 5 ≠ 0` at the moment of the first
decode… concretely, a new page is needed only when `seq_len` crosses a multiple of 16 — at
length 48→49 the manager pops page `9` and the block table becomes `[412, 87, 3051, 9]`.
**Model Runner** builds `input_ids [batch, 1]` (r7's token 9906 and every other running
sequence's last token), positions `[37]`, slot mapping `[3051·16+5]`, the padded block-table
tensor for the whole batch. **Paged KV Cache** scatters 1 new K/V per layer, gathers full
history, SDPA over 38 valid slots. **Sampler** → next token → **Scheduler** appends, streams.

**Step 4b — Memory pressure (if it happens).**
Some decode step, `append_slot` for sequence r9 raises `NoFreePages`. **Scheduler** preempts:
victim = most recently admitted running sequence (say r12); **KV Cache Manager**.`free(r12)`
decrefs all its pages back to the **Page Allocator**; r12's `num_computed_tokens` resets to 0,
status → `WAITING`, pushed to the *front* of the waiting queue (it will re-prefill its prompt
+ already-generated tokens later — recompute preemption). Retry `append_slot(r9)` → succeeds.
**Metrics** increments `preemptions_total`.

**Step 5 — Completion.**
r7 emits its 50th token (or EOS). **Scheduler** marks `FINISHED_LENGTH_CAPPED`, calls
`kv_mgr.free(r7)` → pages `[412, 87, 3051, 9]` decref to 0 → back on the free list (in M6:
into the evictable LRU with their hashes intact). r7 leaves `running`. **Request Manager**
gets the final `RequestOutput(finished=True)` and completes the client call.

**Step 6 — Reclamation check.**
There is no garbage collector — freeing is deterministic at decref-0. The invariant to assert
in tests: after all requests finish, `allocator.num_free == num_pages` (M6: free + evictable).
A leak here is always a missed `free()` on some exit path (preemption, client disconnect,
stop-token).

---

# 4. Data Flow

## 4.1 Request flow (control plane)

```
client ──► Request Manager ──► input queue ──► Scheduler.waiting ──► RUNNING ──► FINISHED
 text        tokenize,           (thread          admission =           │            │
             Request obj          boundary)       pages granted    decode loop   free pages,
                                                                   (or preempt    final output
                                                                    → WAITING)
```

## 4.2 Token flow (one decode iteration)

```
seq.last_token ─► Model Runner ─► HF forward ─► logits[batch] ─► Sampler ─► token_ids
                  input_ids[b,1]                                              │
      ┌───────────────────────────────────────────────────────────────────────┘
      ▼
 Scheduler.postprocess: seq.output_ids.append ─► stop check ─► Detokenizer ─► client stream
```

## 4.3 KV flow (data plane, per layer — the two halves of paged attention)

```
                         WRITE PATH (token writer)
 attn K,V [b, kv_heads, q_len, hd]
        └─ flatten valid tokens ─► [num_tokens, kv_heads, hd]
                                        │ index_copy_ at slot_mapping
                                        ▼
             pool.k[layer].view(pages·block_size, kv_heads, hd)

                         READ PATH (attention lookup)
 block_table [b, max_blocks] ──► pool.k[layer][block_table]
                                        │  [b, max_blocks, block_size, kv_heads, hd]
                                        ▼  reshape + transpose
                          K [b, kv_heads, max_blocks·block_size, hd] ─► SDPA (+mask)
```

## 4.4 Page allocation flow

```
 Scheduler                KV Cache Manager              Page Allocator
    │ can_allocate(seq)        │                              │
    ├──────────────────────────►  need = ceil(len/B) − hits   │
    │                          ├───────── num_free ≥ need? ───►
    │ allocate(seq)            │                              │
    ├──────────────────────────►  pop N page IDs              │
    │                          ├───────── allocate(N) ────────►
    │                          │ ◄──── [412, 87, 3051] ───────┤
    │                          │  block_table += pages        │
    │ free(seq)  (finish/preempt)                             │
    ├──────────────────────────►─────── decref each ──────────►
    │                          │            ref==0 → free list (M6: evictable LRU)
```

## 4.5 Boot flow (strict order — your "orchestrator handshake," corrected)

```
1. loader: HF model + weights → GPU               (fp16, small model)
2. profiler: mem_get_info() → free_after_weights
3. profiler: dummy forward @ max batch shape → activation peak; empty_cache()
4. profiler: num_pages = (free − peak − buffer) × util ÷ page_bytes
5. KVPool: torch.zeros k_cache, v_cache            (the ONE big allocation)
6. PageAllocator(num_pages) → KVCacheManager → Scheduler → Engine
7. Request Manager starts accepting
```

---

# 5. Folder Structure

Replace the 6-package layout with one installable package. Fewer, deeper modules; import
direction is strictly downward (server → engine → scheduler → cache → memory), which
prevents the circular-import problem your previous project hit.

```
KV_Cache_And_GPU_Memory_Management/
├── PLAN.md                      # this document
├── README.md
├── main.py                      # CLI runner: feed prompts from a file, print outputs.
│                                #   The primary dev harness from M3 onward.
├── server.py                    # (M7) FastAPI streaming server; thin shell over the engine.
│
├── kvserve/
│   ├── __init__.py
│   ├── config.py                # ModelConfig (layers/kv_heads/head_dim, from model.config),
│   │                            # CacheConfig (block_size, num_pages, dtype),
│   │                            # EngineConfig (max_num_seqs, max_num_batched_tokens,
│   │                            #   watermark, gpu_memory_utilization). The layout
│   │                            #   "single source of truth" your README wanted — as data,
│   │                            #   not a module of logic.
│   ├── request.py               # Request, Sequence, SequenceStatus, SamplingParams,
│   │                            #   RequestOutput. Zero dependencies; imported by everyone.
│   ├── metrics.py               # Stats counters/gauges + periodic logger.
│   │
│   ├── memory/                  # PHYSICAL layer — token-blind
│   │   ├── __init__.py
│   │   ├── profiler.py          # §2.2 boot-time sizing
│   │   ├── pool.py              # §2.3 KVPool tensors
│   │   └── allocator.py         # §2.3 free list + ref counts (+ evictable LRU in M6)
│   │
│   ├── cache/                   # LOGICAL layer — token-aware, tensor-free until slot maps
│   │   ├── __init__.py
│   │   ├── block_table.py       # §6 BlockTable
│   │   ├── manager.py           # §2.5 KVCacheManager
│   │   └── prefix.py            # §2.8 hash-based prefix cache (M6)
│   │
│   ├── core/                    # CONTROL plane
│   │   ├── __init__.py
│   │   ├── scheduler.py         # §2.4
│   │   └── engine.py            # step loop; owns boot sequence §4.5
│   │
│   ├── model/                   # EXECUTION plane — the only HF-aware package
│   │   ├── __init__.py
│   │   ├── loader.py            # load model/tokenizer, derive ModelConfig
│   │   ├── paged_cache.py       # §2.6 — scatter/gather Cache subclass. The heart.
│   │   ├── runner.py            # §2.7 batch prep + forward
│   │   └── sampler.py           # §2.7
│   │
│   └── offload/                 # (M8, optional)
│       ├── __init__.py
│       └── swap.py              # pinned host mirror + async copies
│
└── tests/
    ├── test_allocator.py        # M1: alloc/free/refcount/exhaustion invariants
    ├── test_block_table.py      # M2: growth boundaries, slot-mapping math
    ├── test_paged_cache.py      # M2: scatter then gather == identity (synthetic tensors)
    ├── test_equivalence.py      # M3: THE golden test — greedy output ≡ hf model.generate()
    ├── test_scheduler.py        # M4/M5: admission, budgets, preemption, no-leak invariant
    └── test_prefix.py           # M6: hit reuse, refcounts, eviction order
```

What got deleted from your layout and where it went: `shared/layout_config.py` →
`config.py` (data, not code) · `shared/system_profiler.py` → `memory/profiler.py` ·
`frontend/tokenizer_worker.py` → deleted (inline tokenization) · `kv_cache/radix_tree.py` →
`cache/prefix.py` (hash-based) · `kv_cache/layer_coordinator.py` → deleted (§1.2) ·
`scheduler/batch_state.py` + `preemptor.py` → `core/scheduler.py` (one policy file) ·
`execution/custom_attention.py` → `model/paged_cache.py` · `gpu_memory/swap_manager.py` →
`offload/swap.py`, optional.

---

# 6. Internal Data Structures

### `SamplingParams`
| field | type | why |
|---|---|---|
| `max_tokens` | int | stop condition; also lets the scheduler bound worst-case page demand |
| `temperature` | float | 0.0 ⇒ greedy — the setting every correctness test pins |
| `top_p` | float | nucleus sampling (post-M3) |
| `stop_token_ids` | list[int] | EOS + extras |

### `Request` (immutable input) / `RequestOutput` (streamed result)
`Request`: `request_id: str`, `prompt_token_ids: list[int]`, `params: SamplingParams`,
`arrival_time: float`. `RequestOutput`: `request_id`, `new_text: str`,
`output_token_ids`, `finished: bool`, `finish_reason: str | None`.

### `Sequence` — the central mutable object
| field | type | why it exists |
|---|---|---|
| `seq_id` | int | key in the KV manager's table |
| `request` | Request | back-pointer for params/output routing |
| `status` | SequenceStatus | `WAITING / RUNNING / SWAPPED / FINISHED_*` — the scheduler's state machine; exactly your `batch_state.py`, now an enum on the object it describes |
| `prompt_token_ids` | list[int] | needed again on recompute-preemption |
| `output_token_ids` | list[int] | grows one per decode step |
| `num_computed_tokens` | int | how many tokens have KV in the pool. **The key field**: prefill computes `len(all_tokens) − num_computed` tokens (prefix-cache hits and preemption-resume both just set this counter). Reset to 0 on preemption. |
| `block_table` | BlockTable | owned logically by the KV manager, carried here for O(1) access |
| `len` (property) | int | `len(prompt) + len(output)` — drives slot math |

### `BlockTable`
| field | type | why |
|---|---|---|
| `page_ids` | list[int] | logical block *i* ↔ physical page `page_ids[i]`; the entire virtual-memory mapping |
| `num_slots_used` | int | `= seq.len`; tells `append_slot` whether the tail page is full (`used % block_size == 0`) |

Methods: `append_page(pid)`, `slot(pos) -> page_ids[pos // B] * B + pos % B`.

### `PageAllocator` (fields)
`free_list: deque[int]` · `ref_count: dict[int, int]` · (M6) `evictable: OrderedDict[int, int]`
(page → hash, LRU order) · (M6) `hash_to_page: dict[int, int]`. Invariant checked in tests:
`len(free) + len(evictable) + len(live) == num_pages`, always.

### `KVPool` (fields)
`k_cache`, `v_cache: Tensor[num_layers, num_pages, block_size, num_kv_heads, head_dim]` ·
`block_size` · `page_bytes`. No methods with logic — views only.

### `ScheduledBatch` — what `schedule()` hands the runner
| field | type | why |
|---|---|---|
| `seqs` | list[Sequence] | batch membership, stable order |
| `is_prefill` | bool | selects the runner's two input-prep paths (§2.0 consequence) |
| `slot_mapping` | LongTensor [num_tokens] | token-writer targets, precomputed on CPU |
| `block_table` | IntTensor [batch, max_blocks] | gather index, padded |
| `seq_lens` | IntTensor [batch] | mask construction + position_ids |

### `Scheduler` (fields)
`waiting: deque[Sequence]` · `running: list[Sequence]` · `kv_mgr` · `engine_config`
(budgets/watermark) · `metrics`.

### `PagedCache(transformers.Cache)` (fields)
`pool: KVPool` · per-step: `slot_mapping`, `block_table`, `seq_lens` (armed by the runner) ·
`get_seq_length(layer)` override returning per-batch computed lengths (HF needs it for mask
logic). One method that matters: `update(key, value, layer_idx) -> (K_full, V_full)` = §2.6.

---

# 7. Component Relationships

## 7.1 Layered view (imports point down; calls point down; data flows back up)

```
┌────────────────────────── main.py / server.py ─────────────────────────┐
│ owns: tokenizer calls, client I/O, the thread-safe input queue         │
└──────────────────────────────────┬─────────────────────────────────────┘
                     add_request() │            ▲ RequestOutput stream
┌──────────────────────────────────▼─────────────────────────────────────┐
│ core/engine.py  — step(): schedule → execute → sample → postprocess    │
│    boot: loader → profiler → pool → allocator → kv_mgr → scheduler     │
└───────────┬──────────────────────────────────────────────┬─────────────┘
   schedule │ postprocess                          execute │ logits ▲
┌───────────▼─────────────────┐             ┌──────────────▼─────────────┐
│ core/scheduler.py           │             │ model/runner.py + sampler  │
│ queues, budgets, preemption │             │ batch prep, HF forward     │
└───────────┬─────────────────┘             └──────────────┬─────────────┘
  (can_)allocate / append_slot / free          set_step_inputs / update()
┌───────────▼─────────────────┐             ┌──────────────▼─────────────┐
│ cache/manager + block_table │  slot maps  │ model/paged_cache.py       │
│ + prefix (M6)               │ ──────────► │ scatter (write) /          │
└───────────┬─────────────────┘  blk tables │ gather (read)              │
   allocate/│free/incref                    └──────────────┬─────────────┘
┌───────────▼─────────────────┐                tensor views│
│ memory/allocator.py         │             ┌──────────────▼─────────────┐
│ free list + refcounts       │   sizes     │ memory/pool.py  (k/v cache)│
└─────────────────────────────┘ ◄────────── │ sized once by profiler.py  │
                                            └────────────────────────────┘
```

Note the deliberate asymmetry: the **control plane** (left column) passes *page IDs and
Python lists*; the **data plane** (right column) passes *CUDA tensors*. They meet in exactly
two objects — `ScheduledBatch` (built left, consumed right) and the pool (owned right,
budgeted left). Keeping that boundary crisp is what makes each half testable without the
other: the allocator/scheduler tests never touch a GPU; the paged-cache test never needs a
scheduler.

## 7.2 One engine step, as a sequence

```
Engine          Scheduler        KVMgr/Alloc         Runner          PagedCache/Pool
  │ schedule()      │                 │                 │                  │
  ├────────────────►│ admit/append ──►│ pages           │                  │
  │                 │ build ScheduledBatch (slot map, block tbl)           │
  │ ◄───batch───────┤                 │                 │                  │
  │ execute(batch)  │                 │                 │                  │
  ├─────────────────┼─────────────────┼────────────────►│ set_step_inputs ►│
  │                 │                 │                 │ hf forward ─────►│ scatter+gather ×L
  │ ◄──────────────logits─────────────┼─────────────────┤                  │
  │ sample → tokens │                 │                 │                  │
  │ postprocess()   │                 │                 │                  │
  ├────────────────►│ append, stop-check, free finished─►                  │
  │ ◄─ outputs ─────┤                 │                 │                  │
```

---

# 8. Implementation Roadmap

Each milestone ends with something runnable and a test that locks it in. Do not start
milestone *N+1* with milestone *N*'s test failing.

**M0 — Skeleton & boot (½ day).**
`config.py`, `request.py`, `loader.py`, `profiler.py`. `main.py` loads SmolLM2-135M fp16,
runs the dummy-forward profiling pass, prints the memory report: free-after-weights,
activation peak, `page_bytes` (verify against the §2.6 formula by hand), computed
`num_pages`. *Done when:* the printed budget survives `nvidia-smi` scrutiny and a second run
is stable.

**M1 — Physical layer (½ day, no GPU needed for tests).**
`pool.py`, `allocator.py`. *Done when:* `test_allocator.py` passes — exhaustion raises,
free-then-realloc reuses IDs, refcount double-free is caught by assertion, and the
count invariant (§6) holds under a randomized alloc/free fuzz loop.

**M2 — Logical layer + data plane, no model (1 day).**
`block_table.py`, `manager.py`, `paged_cache.py`'s scatter/gather as free functions first.
*Done when:* `test_paged_cache.py` writes random tensors through a slot mapping, gathers via
a block table, and gets bit-identical values back — including a case with deliberately
shuffled, non-contiguous page IDs and a partial last block.

**M3 — Single-sequence end-to-end + golden test (1–2 days, the crux).**
`paged_cache.py` as a real `Cache` subclass, `runner.py`, `sampler.py` (greedy), minimal
engine loop, one request at a time. *Done when:* `test_equivalence.py` shows your engine's
greedy output is **token-for-token identical** to `model.generate(do_sample=False)` for ~10
varied prompts (lengths that do and don't align to block boundaries). Debug order when it
isn't: slot mapping off-by-one → mask over unwritten tail slots → position_ids. Expect to
spend your time here; this test is the project's foundation.

**M4 — Continuous batching (1–2 days).**
Full `scheduler.py`: waiting/running queues, prefill-vs-decode steps, token/seq budgets,
sequences joining and leaving mid-flight. *Done when:* 20 concurrent requests of wildly
different lengths all finish with outputs matching their single-sequence golden results, and
the no-leak invariant holds at the end.

**M5 — Preemption by recompute (½–1 day).**
Shrink `num_pages` artificially (e.g., 64) to force pressure. On-demand preemption, LIFO
victim, re-prefill on resume using `num_computed_tokens=0`. *Done when:* `test_scheduler.py`
proves outputs are *still identical* to golden despite forced preemptions, and
`preemptions_total > 0` confirms the path actually ran.

**M6 — Prefix caching (1–2 days).**
`prefix.py`, allocator's evictable LRU, chained block hashing, `num_computed_tokens` skip in
prefill. *Done when:* two requests sharing a 200-token system prompt physically share pages
(assert same page IDs), outputs stay golden-identical, hit-rate metric reports correctly, and
eviction under pressure doesn't corrupt a third request.

**M7 — Serving & metrics (1 day).**
`server.py` (FastAPI, streaming), `metrics.py` wired everywhere, the thread-safe queue
boundary. *Done when:* you can `curl` two concurrent streaming completions while the metrics
line shows both running and cache usage moving.

**M8 (optional) — Swap-based preemption.** §2.9. *Done when:* a swapped-and-resumed sequence
still matches golden output, with `Event.query()` gating the resume.

**M9 (optional) — Triton paged-attention kernel.** Replace decode-path gather+SDPA with a
fused kernel; benchmark against M3's gather implementation. This is where the project
graduates from "how vLLM is organized" to "how vLLM is fast."

---
*Reference points while building: vLLM v0's `BlockSpaceManager`/`Scheduler` (this design
mirrors them at reduced scale), the vLLM paper (Kwon et al., SOSP '23) for the paging and
sharing arguments, and HF `transformers`' `Cache` API docs for the `update()` contract.*

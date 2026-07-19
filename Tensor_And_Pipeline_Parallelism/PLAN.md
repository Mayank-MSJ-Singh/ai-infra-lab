# Tensor & Pipeline Parallelism Engine — Architectural Blueprint

**Project:** A production-inspired, educational distributed LLM inference engine
**Stack:** Python + PyTorch Distributed (NCCL / Gloo)
**Status:** Design document — the single source of truth for the implementation
**Scope:** Tensor Parallelism (TP) + Pipeline Parallelism (PP) for autoregressive decoder inference

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Review — Principles, Scope, Topology](#2-architecture-review--principles-scope-topology)
3. [Complete System Architecture — Component Catalog](#3-complete-system-architecture--component-catalog)
4. [Complete Request Lifecycle](#4-complete-request-lifecycle)
5. [Complete Data Flow](#5-complete-data-flow)
6. [Component Relationships](#6-component-relationships)
7. [Folder Structure](#7-folder-structure)
8. [Internal Data Structures](#8-internal-data-structures)
9. [Tensor Parallelism Design](#9-tensor-parallelism-design)
10. [Pipeline Parallelism Design](#10-pipeline-parallelism-design)
11. [Consolidated Architecture Diagrams](#11-consolidated-architecture-diagrams)
12. [Design Decisions](#12-design-decisions)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [Glossary](#14-glossary)

---

## Running Example Used Throughout This Document

Every diagram and walkthrough in this document uses one fixed topology so you can
cross-reference sections without re-translating rank numbers:

```
World size:        4 processes (1 per GPU)
Tensor parallel:   TP = 2
Pipeline parallel: PP = 2
Model:             GPT-2-small-like decoder
                   12 transformer layers, 12 attention heads, hidden = 768

Rank layout (rank = pp_rank * tp_size + tp_rank):

            tp_rank 0     tp_rank 1
  pp_rank 0  [rank 0]      [rank 1]     ← Stage 0: embedding + layers 0–5
  pp_rank 1  [rank 2]      [rank 3]     ← Stage 1: layers 6–11 + final norm + LM head

TP groups (AllReduce domains):  {0, 1}   {2, 3}
PP groups (send/recv chains):   {0, 2}   {1, 3}
Driver rank (frontend):          rank 0
```

Everything generalizes: `world_size = TP × PP`, TP ranks are contiguous, and the
same code runs on 2 GPUs (TP=2, PP=1), 4, 8, or more. The design also runs on CPU
with the Gloo backend so you can develop and unit-test without a multi-GPU machine.

---

# 1. System Overview

## 1.1 The Problem: One Model, Too Big for One Device

Modern LLMs no longer fit on a single accelerator. The problem shows up on three
axes simultaneously:

1. **Memory capacity.** A 70B-parameter model in fp16 needs ~140 GB just for
   weights — before KV cache, activations, and CUDA overhead. No single GPU holds
   that.
2. **Memory bandwidth.** Autoregressive decoding is memory-bandwidth-bound: every
   generated token requires streaming all weights through the compute units.
   Splitting weights across GPUs multiplies aggregate bandwidth.
3. **Latency.** Even if a model *fits* on one GPU, a single device may be too slow
   to meet a latency target. Parallelism converts "more hardware" into "lower
   latency per token" (TP) or "higher throughput" (PP).

Distributed inference is therefore not an optimization — it is an existential
requirement. The two dominant strategies for slicing a model across devices are
**tensor parallelism** and **pipeline parallelism**, and every serious inference
engine (Megatron-LM, vLLM, DeepSpeed-Inference, TensorRT-LLM) implements both.

## 1.2 What Tensor Parallelism Is

Tensor parallelism splits **individual layers** across GPUs. Every GPU in a TP
group holds a *slice* of every weight matrix in the layer (a column-block or a
row-block), computes a partial result on its slice, and the group runs a
collective communication operation (AllReduce or AllGather) to combine the
partial results into the mathematically identical full result.

Key properties from a distributed-systems perspective:

- **All GPUs in a TP group execute the same layer at the same time.** It is
  intra-layer, synchronous, lock-step parallelism (SPMD in its purest form).
- **Communication is frequent and latency-sensitive.** Every transformer layer
  triggers collectives (two AllReduces per layer in the classic Megatron
  scheme). This is why TP is almost always confined to GPUs connected by
  NVLink/NVSwitch inside a single node.
- **It reduces per-GPU memory AND per-token latency.** Each GPU stores 1/TP of
  the weights and performs 1/TP of the FLOPs for a token, so a single token's
  forward pass genuinely gets faster (until communication overhead dominates).
- **It is invisible to the scheduler.** A TP group behaves, from the outside,
  like one bigger logical GPU. This abstraction is central to our architecture.

## 1.3 What Pipeline Parallelism Is

Pipeline parallelism splits the model **by layers** across GPUs. Stage 0 owns
layers 0–5, stage 1 owns layers 6–11. A forward pass becomes a relay race: stage
0 computes its layers and sends the activation tensor to stage 1 via a
point-to-point (P2P) send/recv.

Key properties:

- **GPUs execute different layers at different times.** It is inter-layer,
  asynchronous, dataflow parallelism — much closer to a classic distributed
  pipeline (think: staged stream processing) than to lock-step SPMD.
- **Communication is infrequent and cheap.** Only one activation tensor crosses
  a stage boundary per microbatch — a single `[batch, seq, hidden]` tensor. This
  tolerates slow interconnects, which is why PP is the strategy of choice for
  crossing node boundaries (Ethernet/InfiniBand instead of NVLink).
- **It reduces memory but not single-request latency.** A token still visits
  every layer sequentially; PP adds transfer hops. Its payoff is *throughput*:
  with multiple in-flight microbatches, all stages compute simultaneously.
- **It introduces the pipeline bubble problem.** When the pipeline fills and
  drains, some stages sit idle. Managing bubbles via microbatch scheduling is
  the core intellectual content of PP.

## 1.4 Why They Are Combined

TP and PP have exactly complementary failure modes:

| Axis | Tensor Parallelism | Pipeline Parallelism |
|---|---|---|
| Communication frequency | Every layer (2+ collectives) | Once per stage boundary |
| Communication pattern | Collective (AllReduce) | Point-to-point (send/recv) |
| Interconnect requirement | NVLink-class (fast) | Tolerates network-class (slow) |
| Latency per token | Improves | Slightly worsens (hops) |
| Throughput | Improves | Improves (with microbatching) |
| Scaling limit | ~8 GPUs (head count, NVLink domain) | Number of layers |
| Idle-time pathology | None (lock-step) | Pipeline bubbles |

Because TP degrades sharply past the NVLink boundary and PP degrades gracefully
across it, production systems compose them into a **2D parallelism grid**: TP
*inside* a node, PP *across* nodes. A 70B model on 2 nodes × 8 GPUs runs as
TP=8, PP=2. The TP group is the unit of computation ("one logical GPU"); the
pipeline connects those logical GPUs in a chain.

This 2D composition — and the abstraction layering that makes each dimension
ignorant of the other — is the single most important architectural idea in this
project. Our entire component design flows from one rule:

> **Tensor-parallel layers never know pipeline parallelism exists, and the
> pipeline scheduler never knows tensor parallelism exists. Only the
> ParallelContext knows about both.**

## 1.5 Where This Is Used in Production

- **Megatron-LM** (NVIDIA) defined the canonical TP layer designs
  (ColumnParallelLinear / RowParallelLinear, vocab-parallel embeddings) and the
  process-group bookkeeping module (`parallel_state` / `mpu`) that our
  `ParallelContext` is modeled after.
- **vLLM** uses Megatron-style TP inside its workers, treats the TP group as one
  logical worker, and drives it from a central engine process — the
  driver/worker split our Engine mirrors.
- **DeepSpeed-Inference** popularized inference-optimized pipeline scheduling
  and kernel-injection into existing HF models; our WeightLoader's
  "map HF checkpoint names → sharded parameters" approach follows its spirit.
- **TensorRT-LLM** compiles TP/PP into static engines; its insistence on
  *statically known shapes* for communication inspired our static-shape P2P
  protocol (§10.6).

## 1.6 What This Project Is and Is Not

**It is:** a from-scratch, multi-process, NCCL-backed inference engine that
loads a real HF GPT-2 checkpoint, shards it 2D (TP × PP), serves generation
requests through a real request manager and scheduler, pipelines microbatches
through stages, and streams tokens back — with metrics, logging, and a test
strategy.

**It is not:** a kernel-optimization project (no custom CUDA, no fused kernels,
no CUDA graphs), not a paged-KV-cache project (that is your
`KV_Cache_And_GPU_Memory_Management` project), and not a continuous-batching
project (that is your `Static_Dynamic_Continues_Batching` project). Where those
concerns appear here, they appear in their simplest correct form, with explicit
extension points noted.

---

# 2. Architecture Review — Principles, Scope, Topology

## 2.1 Design Principles

Every component in §3 exists because it serves one of these principles. When in
doubt during implementation, return here.

**P1 — SPMD with a single driver.**
All ranks run the *same program* (launched by `torchrun`). Rank 0 additionally
plays the "driver" role: it owns the RequestManager and Scheduler and broadcasts
control decisions to all ranks. This mirrors how Megatron and vLLM's older MP
backend work, and it avoids the complexity of a separate coordinator process
(Ray actors, RPC servers). The asymmetry lives in exactly one place — the
Engine's run loop — everything else is symmetric.

**P2 — Control plane and data plane are separate.**
*Control plane:* small Python objects (scheduling decisions, request metadata,
sampled token ids as lists) broadcast over a CPU/Gloo process group.
*Data plane:* large tensors (activations, logits) moving over NCCL.
Mixing them is the #1 source of deadlocks and mystery hangs in hand-rolled
distributed code. Keeping them on separate process groups with separate
managers makes every message's path auditable.

**P3 — The ParallelContext is the only component that knows the full topology.**
Layers ask it "what is my TP group?"; the pipeline scheduler asks "am I the
last stage?". Nobody else computes rank arithmetic. One authority, zero
disagreement.

**P4 — Static shapes on the wire.**
Every P2P transfer's shape is derivable from config + the scheduling decision
that was broadcast on the control plane. Receivers pre-allocate exact buffers.
No dynamic shape handshakes, no pickled tensors on the data plane. This is what
TensorRT-LLM does by compilation and what we do by convention.

**P5 — Every abstraction must be independently testable without GPUs.**
TP layers run under Gloo on CPU with world_size=2 in pytest. The pipeline
scheduler is a pure function from (num_microbatches, num_stages, rank) to an
action sequence, testable with no distributed runtime at all. This is not just
a convenience — designing for single-process testability *forces* the clean
separation of policy (scheduling) from mechanism (communication).

**P6 — Educational fidelity over feature count.**
One good pipeline schedule (fill-drain / GPipe-style forward-only), fully
understood, beats three half-implemented ones. Every "production system X does
Y instead" note is in §12 so you know exactly where the training wheels are.

## 2.2 Scope Decisions (What's In, What's Out)

| Concern | In scope | Out of scope (and why) |
|---|---|---|
| Tensor parallelism | Full Megatron-style: column/row linear, vocab-parallel embedding, TP attention, TP MLP | Sequence parallelism (an optimization of the same idea; noted in §9.8) |
| Pipeline parallelism | Fill-drain forward-only schedule, microbatching, P2P activations, decode-loop token feedback | Interleaved/virtual stages, 1F1B (training schedules — no backward pass here) |
| Batching | Static per-round batching with FCFS queue | Continuous batching (your other project; extension point noted) |
| KV cache | Preallocated contiguous per-slot cache, TP-sharded heads, per-stage layers | Paged attention (your other project) |
| Model | GPT-2 small via HF checkpoint | Anything requiring RoPE/GQA/MoE (orthogonal to the systems content) |
| Sampling | Greedy + temperature/top-k on last stage | Beam search, guided decoding |
| Serving | In-process client API + simple CLI; optional thin HTTP front | Production API server, auth, multi-tenancy |
| Fault tolerance | Clean startup validation, graceful shutdown, watchdog timeouts | Elastic recovery, rank failure healing (research-grade even in production) |
| Precision | fp32 (CPU/Gloo) and fp16 (GPU/NCCL) | Quantization |

## 2.3 Physical and Logical Topology

```
PHYSICAL                          LOGICAL

┌─────────────────────────┐      ┌──────────────────────────────────────┐
│ Node (1 machine)        │      │ 2D Device Mesh  (pp × tp)            │
│                         │      │                                      │
│  GPU0   GPU1            │      │            tp=0      tp=1            │
│   │      │   NVLink/PCIe│      │  pp=0   [ rank 0 ][ rank 1 ]  Stage0 │
│  GPU2   GPU3            │      │  pp=1   [ rank 2 ][ rank 3 ]  Stage1 │
│                         │      │                                      │
│  4 processes, torchrun  │      │  TP groups → rows  (AllReduce)       │
│  1 process per GPU      │      │  PP groups → cols  (send/recv)       │
└─────────────────────────┘      └──────────────────────────────────────┘
```

**Why TP ranks are contiguous** (`rank = pp_rank * tp_size + tp_rank`): with
`torchrun`, contiguous global ranks land on the same node first. TP needs the
fast intra-node links because it communicates every layer; PP crosses the slow
boundary because it communicates once per stage. This rank-assignment formula
is a *policy encoding of hardware reality*, and it lives in exactly one
function inside the ProcessGroupManager.

## 2.4 The Five Planes of the System

It helps to see the architecture as five horizontal planes; each component in
§3 belongs to exactly one:

```
┌────────────────────────────────────────────────────────────────────┐
│ SERVING PLANE      RequestManager, Scheduler, (CLI / client API)   │  rank 0 only
├────────────────────────────────────────────────────────────────────┤
│ ORCHESTRATION      Engine, PipelineScheduler, MicrobatchManager,   │  all ranks
│ PLANE              StageExecutor                                   │
├────────────────────────────────────────────────────────────────────┤
│ MODEL PLANE        ModelRunner, PipelineStage, TP layers,          │  all ranks
│                    WeightLoader, TensorShardingManager, Sampler    │
├────────────────────────────────────────────────────────────────────┤
│ COMMUNICATION      CommunicationManager (collectives + P2P +       │  all ranks
│ PLANE              control-plane broadcast)                        │
├────────────────────────────────────────────────────────────────────┤
│ RUNTIME PLANE      DistributedRuntime, ProcessGroupManager,        │  all ranks
│                    ParallelContext, DeviceManager, Config,         │
│                    Logging, Metrics                                │
└────────────────────────────────────────────────────────────────────┘
```

Calls only flow **downward** (serving → orchestration → model → communication →
runtime). No layer ever calls upward; upward information flow happens through
return values and the metrics/logging sinks. This one constraint is what keeps
the system understandable.

---

# 3. Complete System Architecture — Component Catalog

Components are presented bottom-up (runtime plane first) because that is also
the build order: you cannot test a TP layer before process groups exist.

For each component: **Why it exists · Responsibilities · Data owned · Internal
state · Public API · Internal API · Relationships · Lifecycle · Design
rationale.** API signatures are given in interface notation (name, inputs →
outputs); they describe contracts, not code.

---

## 3.1 Configuration System

**Why it exists.** In a distributed SPMD program, *every rank must derive
identical decisions from identical inputs*. If rank 0 thinks there are 4
microbatches and rank 2 thinks there are 3, the system deadlocks inside NCCL
with no error message. A single, validated, immutable configuration object —
constructed identically on every rank before anything else initializes — is the
first line of defense against divergence. It is also where invariants are
enforced *once*, at startup, so no downstream component ever re-checks them.

**Responsibilities.**
- Parse a YAML/CLI config into typed, frozen dataclasses.
- Validate all cross-field invariants at startup, before `init_process_group`:
  - `world_size == tensor_parallel_size × pipeline_parallel_size`
  - `num_attention_heads % tensor_parallel_size == 0`
  - `num_layers % pipeline_parallel_size == 0` (uniform stages; see §12)
  - `vocab_size % tensor_parallel_size == 0` (pad vocab if not)
  - `max_batch_size % num_microbatches == 0`
- Derive secondary values (padded vocab size, per-stage layer ranges, dtype).

**Data owned.** The frozen config tree:
`EngineConfig { ModelConfig, ParallelConfig, SchedulerConfig, RuntimeConfig }`
(fields detailed in §8.1).

**Internal state.** None after construction — deliberately immutable. Frozen
dataclasses mean any component holding a config reference can never observe a
change, which eliminates a whole class of rank-divergence bugs.

**Public API.**
- `EngineConfig.from_yaml(path, overrides) → EngineConfig`
- `EngineConfig.validate() → None` (raises `ConfigError` with *all* violations
  listed, not just the first)
- `config.stage_layer_range(pp_rank) → (first_layer, last_layer)`
- `config.padded_vocab_size → int`

**Internal API.** Per-section validators (`_validate_parallel`,
`_validate_model_vs_parallel`) — split so unit tests can target each invariant.

**Relationships.** Constructed first, injected into everything. Depends on
nothing. Nothing may mutate it.

**Lifecycle.** Built from file + env before distributed init; lives for the
process lifetime.

**Design rationale.** Dataclasses over a dict: typos become `AttributeError`s
at startup instead of silent `dict.get` defaults diverging across ranks.
Validation *before* `init_process_group`: a config error should print one clean
message and exit 1, not produce four stack traces and a hung NCCL collective.
Deriving stage layer ranges here (not in the pipeline code) keeps the
"which layers belong to stage k" policy in one queryable place that tests,
WeightLoader, and PipelineStage all share.

---

## 3.2 DistributedRuntime

**Why it exists.** `torch.distributed` initialization, environment parsing,
teardown, and hang diagnostics are process-level concerns that belong to no
model component. Isolating them gives every other component a simple worldview:
"by the time I am constructed, the distributed world exists and is healthy."
It is also the component that makes the CPU/Gloo development mode possible —
everything above it is backend-agnostic.

**Responsibilities.**
- Read `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR/PORT` from the
  `torchrun` environment.
- Call `init_process_group` with the right backend (NCCL on GPU, Gloo on CPU),
  and initialize a *second* Gloo group for the control plane even on GPU runs
  (P2 — control/data separation).
- Provide `barrier()`, ordered shutdown (`destroy_process_group`), and a
  watchdog: a monitor thread that logs "rank R stuck in <op> for >T seconds"
  before NCCL's own opaque timeout kills the job.
- Install signal handlers so Ctrl-C on one rank triggers coordinated shutdown
  instead of a zombie NCCL communicator.

**Data owned.** Backend handles, the control-plane Gloo group, rank/world-size
identity, watchdog thread.

**Internal state.** `initialized: bool`, `last_op_description: str` (set by
CommunicationManager before each blocking op — this is what makes hang logs
useful).

**Public API.**
- `initialize(config) → RuntimeInfo` (rank, world_size, local_rank, backend)
- `control_group() → ProcessGroup` (the Gloo group)
- `barrier(reason: str)` — the reason string is logged; anonymous barriers are
  banned because they are undebuggable.
- `shutdown(exit_code)`

**Internal API.** `_parse_env()`, `_start_watchdog()`, `_install_signal_handlers()`.

**Relationships.** First component to run after Config. ProcessGroupManager
requires it. Engine calls `shutdown` last.

**Lifecycle.** `initialize` → alive for process → `shutdown`. Exactly one
instance per process.

**Design rationale.** The watchdog + named-barrier discipline is the single
highest-value debugging investment in the whole project: distributed bugs
manifest as silent hangs, and "rank 2 stuck in `recv activations ubatch=3
from rank 0` for 60s" turns a day of debugging into a minute. The separate
Gloo control group exists because NCCL collectives cannot carry arbitrary
Python objects and because issuing control messages on the data-plane
communicator can interleave with tensor traffic and deadlock.

---

## 3.3 DeviceManager

**Why it exists.** "Which physical device does this process own, what dtype do
we run in, and how much memory is left" is a per-process hardware concern.
Without a single owner, `torch.cuda.set_device` calls scatter across the
codebase and one missed call silently lands tensors on GPU 0 from every
process — the classic multi-process PyTorch bug.

**Responsibilities.**
- Bind the process to `cuda:{LOCAL_RANK}` exactly once, before any tensor is
  created; or select CPU in Gloo mode.
- Own the dtype policy (fp16 on GPU, fp32 on CPU) so layers never hardcode it.
- Expose memory telemetry (`allocated / reserved / total`) to Metrics.
- Provide the device/dtype to every tensor-allocating component.

**Data owned.** `device: torch.device`, `dtype: torch.dtype`, device properties.

**Internal state.** Peak-memory watermark (sampled after weight load and after
warmup — the two numbers that tell you if your sharding math is right).

**Public API.**
- `bind(local_rank, config) → None`
- `device → torch.device`, `dtype → torch.dtype`
- `memory_snapshot() → MemoryStats`
- `empty_cache_after_load()` — one deliberate cache purge after weight loading
  (HF checkpoint loading fragments the allocator).

**Relationships.** Initialized right after DistributedRuntime (NCCL group
creation itself needs the device bound). Read by ModelRunner, WeightLoader,
CommunicationManager (buffer allocation), Metrics.

**Lifecycle.** Bound once at startup; immutable afterward.

**Design rationale.** Making dtype a DeviceManager policy (not a per-layer
choice) is what lets the identical model code run under Gloo/CPU in tests and
NCCL/fp16 in production mode — P5 depends on this component being boringly
strict.

---

## 3.4 ProcessGroupManager

**Why it exists.** PyTorch gives you one flat world; 2D parallelism needs
*subgroups* — and `torch.distributed.new_group` must be called **by every rank
for every group, in the same order**, even for groups the rank doesn't belong
to. This collective-creation subtlety is exactly the kind of landmine that
deserves a dedicated, heavily-tested component. It converts the flat rank space
into the structured mesh everything else thinks in.

**Responsibilities.**
- Compute the rank grid from ParallelConfig:
  `grid[pp][tp] = pp * tp_size + tp`.
- Create, in deterministic order on all ranks:
  - one **TP group** per mesh row (NCCL) — the AllReduce/AllGather domain,
  - one **PP group** per mesh column (NCCL) — the send/recv chain,
  - one **control group** spanning the world (Gloo) — obtained from
    DistributedRuntime and registered here for uniform lookup,
  - one **embedding group** = {first-stage ranks, last-stage ranks} per column
    (reserved for tied-embedding extension; created but unused — see §12.7).
- Answer membership queries: this rank's tp_rank, pp_rank, group handles,
  previous/next rank in the pipeline chain.

**Data owned.** All ProcessGroup handles and the `DeviceMesh` (§8.4).

**Internal state.** The mesh, this rank's coordinates, a name→group registry.

**Public API.**
- `build(runtime, config) → DeviceMesh`
- `tp_group() / pp_group() / control_group() → ProcessGroup`
- `tp_rank() / tp_size() / pp_rank() / pp_size() → int`
- `prev_pipeline_rank() → int | None`, `next_pipeline_rank() → int | None`
- `is_first_stage() / is_last_stage() → bool`
- `rank_at(pp, tp) → int` (used by Engine to find "last stage, tp 0" for
  token gathering)

**Internal API.** `_create_groups_in_canonical_order()` — the one function
encoding the "every rank creates every group" rule, with a docstring explaining
why reordering it deadlocks.

**Relationships.** Requires DistributedRuntime. ParallelContext wraps it.
Nothing else touches it directly — everyone goes through ParallelContext.

**Lifecycle.** Built once at startup. Groups destroyed by DistributedRuntime at
shutdown.

**Design rationale.** Separating this from ParallelContext (rather than one
merged class) splits *mechanism* (the delicate collective group creation) from
*interface* (topology queries). The mechanism gets exhaustive tests with
mocked `new_group` calls asserting call order; the interface stays a trivial,
obviously-correct facade.

---

## 3.5 ParallelContext

**Why it exists.** This is the keystone (P3). Every distributed decision in the
system — "which ranks do I AllReduce with", "am I the stage that samples",
"who do I send activations to" — reduces to a topology query. Megatron calls
this `parallel_state`, and every engine has an equivalent. Centralizing it
means rank arithmetic exists in exactly one tested place; the alternative
(each layer computing `rank % tp_size`) is how subtle topology bugs are born.

**Responsibilities.**
- Provide the complete, read-only topology API to all planes.
- Carry the per-process identity: global rank, tp_rank, pp_rank, is-driver.
- Host the "role predicates" that make call-sites readable:
  `is_driver()`, `is_first_stage()`, `is_last_stage()`,
  `is_tp_source()` (tp_rank == 0).

**Data owned.** References to DeviceMesh, ProcessGroupManager, RuntimeInfo,
DeviceManager. It owns no tensors and no mutable state.

**Internal state.** None mutable — it is a read-only view. This is deliberate:
a context object that accumulates state becomes a god object; a read-only one
stays a coordinate system.

**Public API.** Union of ProcessGroupManager's query API plus:
- `is_driver() → bool` (global rank 0)
- `describe() → str` — one-line topology summary each rank logs at boot
  (`"rank 2 = pp 1 / tp 0, stage layers 6–11, next=None prev=0"`); reading four
  of these lines and checking they're consistent is the first smoke test of
  every run.

**Internal API.** None — it is pure delegation.

**Relationships.** Constructed from ProcessGroupManager + DeviceManager.
Injected into CommunicationManager, all TP layers, PipelineScheduler,
ModelRunner, Engine, Metrics. **The only component whose reference is allowed
to appear everywhere.**

**Lifecycle.** Built at startup, immutable, process lifetime.

**Design rationale.** Explicit injection (constructor argument) rather than
Megatron's module-level globals. Globals are why Megatron is hard to unit-test;
with injection, a test constructs a fake ParallelContext with
`tp_size=2, pp_size=1` and tests a ColumnParallelLinear on CPU without any
distributed init. This one choice buys most of P5.

---

## 3.6 CommunicationManager

**Why it exists.** Raw `torch.distributed` calls scattered through model code
are untestable, unmeasurable, and undebuggable. Wrapping them in one component
gives you: (a) a seam to mock in single-process tests, (b) the place where
every byte is counted for Metrics, (c) the place where `last_op_description`
is set for the watchdog, and (d) enforcement of the static-shape protocol
(P4). This is the mechanism layer for *all* bytes that leave a process.

**Responsibilities.**
- **TP collectives** (on the tp_group): `all_reduce(tensor)`,
  `all_gather(tensor) → tensor` (concatenated on last dim),
  `broadcast_within_tp(tensor, src_tp_rank)`.
  (`reduce_scatter` included for the sequence-parallel extension; see §9.8.)
- **PP point-to-point** (on the pp_group): `send_activations(tensor, ubatch_id)`,
  `recv_activations(shape, dtype, ubatch_id) → tensor` — receiver allocates
  from shape metadata it already knows (P4).
- **Control plane** (on Gloo): `broadcast_control(obj, src=0) → obj` — pickled
  Python objects: ScheduleDirectives, sampled token lists, shutdown sentinel.
- **Bookkeeping:** per-call byte counts and wall time into Metrics; op
  descriptions into the watchdog.

**Data owned.** Reusable pinned/pre-allocated receive buffers keyed by
(shape, dtype) — avoids per-microbatch allocation.

**Internal state.** Buffer cache; cumulative byte/time counters per op type.

**Public API.** The nine methods above — the complete vocabulary of the system.
If an implementation file needs a distributed op not on this list, that is a
design smell to stop and examine.

**Internal API.** `_get_buffer(shape, dtype)`, `_record(op, bytes, seconds)`.

**Relationships.** Uses ParallelContext for groups, DeviceManager for buffer
placement, DistributedRuntime for the watchdog hook, Metrics as a sink. Used
by every TP layer (collectives), StageExecutor (P2P), Engine (control plane).

**Lifecycle.** Constructed after ParallelContext; process lifetime; buffers
freed at shutdown.

**Design rationale.** One manager for all three traffic classes (instead of a
TP-comms class and a PP-comms class) was chosen because the *cross-cutting*
concerns — metering, watchdog labeling, buffer reuse — are identical, and a
single seam is easier to mock. The methods are grouped internally by plane so
the file still reads as three small modules. Synchronous (blocking) ops only,
except `isend/irecv` pairs inside the pipeline steady-state where overlap is
the whole point (§10.5) — async everywhere would teach concurrency bugs, not
parallelism.

---

## 3.7 TensorShardingManager

**Why it exists.** The mapping *"given full parameter P of layer type T, which
slice belongs to tp_rank r?"* is pure policy — a small set of slicing rules —
but it is consumed by two very different clients: the WeightLoader (slicing
checkpoint tensors at load time) and the tests (verifying that re-assembling
all shards reproduces the full tensor). Extracting it as a stateless policy
object means the slicing math is written once, tested exhaustively in
isolation, and *provably shared* between loader and verifier.

**Responsibilities.**
- Define the shard spec for every parameter class:
  - column-parallel weight → split output dim (dim 0 of `[out, in]`) into
    `tp_size` blocks; bias split the same way.
  - row-parallel weight → split input dim (dim 1); bias **not** split
    (replicated, added only after AllReduce — see §9.3).
  - vocab-parallel embedding → split vocab dim (rows); owns
    `[r·V/tp, (r+1)·V/tp)`.
  - QKV fused weights → split *per head group* so each rank gets whole heads
    (never split inside a head).
  - replicated params (LayerNorm weights/biases) → no split; every TP rank
    holds a full copy.
- Compute `ShardSpec` objects (§8.7): full shape → (slice per dim, local shape).
- Provide the inverse operation (`assemble(shards) → full tensor`) for tests.

**Data owned / Internal state.** None — pure functions over (param name/type,
full shape, tp_rank, tp_size). Statelessness is the point.

**Public API.**
- `spec_for(param_kind, full_shape, tp_rank, tp_size) → ShardSpec`
- `shard(tensor, spec) → tensor`
- `assemble(shards, param_kind) → tensor` (test-only)

**Relationships.** Used by WeightLoader; mirrored by the TP layers' own local
shape computations (a startup assertion checks that layer-declared local shapes
match loader-produced shard shapes — catching the classic off-by-one in
head-group slicing at load time instead of as NaNs at runtime).

**Lifecycle.** Stateless; effectively a module of functions grouped as a class
for injection.

**Design rationale.** The alternative — each layer knowing how to shard its own
checkpoint tensor — couples model code to checkpoint format. With this split,
layers only declare *their local shapes*, the sharding manager only knows
*slicing math*, and the WeightLoader only knows *checkpoint format*. Three
small, independently testable pieces.

---

## 3.8 WeightLoader

**Why it exists.** Between "HF checkpoint on disk" and "correctly sharded
fp16 tensors inside this rank's layers" lies a surprising amount of systems
work: name mapping, layout fixes, stage filtering, memory-safe loading. In
DeepSpeed and vLLM this is a substantial subsystem; here it is one focused
component, because getting real GPT-2 weights in (instead of random init) is
what turns the project from a simulation into an engine whose output you can
verify against HuggingFace token-for-token.

**Responsibilities.**
- Open the GPT-2 checkpoint (safetensors) **memory-mapped** — never load the
  full state dict to RAM per rank.
- Map HF parameter names → our module names, including the GPT-2 quirk that
  `Conv1D` weights are stored transposed relative to `nn.Linear`.
- For this rank: iterate only over parameters of layers in
  `config.stage_layer_range(pp_rank)` (pipeline filtering), get each param's
  `ShardSpec` from TensorShardingManager (tensor filtering), read *only that
  slice* from the mmap, cast to DeviceManager's dtype, copy into the layer.
- Pad the vocab-parallel embedding rows if `padded_vocab_size > vocab_size`.
- Verify completeness: every local parameter received exactly one write; error
  with the full list of misses otherwise.

**Data owned.** The name-mapping table; nothing at runtime (weights are owned
by layers).

**Internal state.** During load only: set of filled parameter names, byte
counters for the load report (`rank 3: loaded 41/148 tensors, 118 MiB`).

**Public API.** `load(model_shard: PipelineStage, parallel_ctx) → LoadReport`

**Internal API.** `_hf_name_to_local(name)`, `_read_slice(mmap_tensor, spec)`,
`_verify_complete(model_shard)`.

**Relationships.** Uses TensorShardingManager (specs), ParallelContext (which
stage/tp rank), DeviceManager (dtype/device). Called once by ModelRunner during
startup.

**Lifecycle.** Constructed, `load()` once, discarded. Holding no post-load
state is deliberate — checkpoint handling should not leak into serving.

**Design rationale — every-rank-reads vs. rank-0-broadcast.** Two standard
strategies: (a) every rank opens the checkpoint and reads its slice; (b) rank 0
reads and scatters shards over NCCL. We choose **(a)**: with mmap +
slice-reads, each rank touches only the bytes it keeps, there is zero startup
communication code to deadlock, and it works identically on 1 node or N nodes
with a shared filesystem. (b) matters when checkpoints live on one node only or
object storage is slow — noted as an extension in §12.6.

---

## 3.9 ColumnParallelLinear

**Why it exists.** One of the two atoms of tensor parallelism. A linear layer
`Y = X·Wᵀ + b` whose weight is split by **output columns** across the TP group:
each rank computes a distinct *slice of the output features*. It exists as its
own module (rather than TP logic inside attention/MLP) because Megatron's key
insight is that **all of TP is just these two linear variants composed
correctly** — making them first-class is the whole lesson.

**Responsibilities.**
- Hold the local weight shard `[out/tp, in]` and bias shard `[out/tp]`.
- Forward: local matmul producing `[.., out/tp]`.
- Depending on `gather_output` flag: return the partial (normal case — the
  consumer is a RowParallelLinear that *wants* the split input) or AllGather
  into the full `[.., out]` (used by the LM head).
- Declare its local shapes for the WeightLoader cross-check (§3.7).

**Data owned.** Its weight/bias shards — the layer is the owner of record for
these tensors' lifetime.

**Internal state.** None beyond parameters (inference-only, stateless forward).

**Public API.** `forward(x) → y_partial_or_full`, `local_out_features`,
`shard_spec()`.

**Internal API.** None.

**Relationships.** Input must be **replicated** across the TP group (an
invariant documented on the class; upstream layers guarantee it). Composed
inside TPAttention (QKV projection), TPMLP (up-projection), OutputLayer
(LM head). Uses CommunicationManager only when `gather_output=True`.

**Lifecycle.** Constructed empty at model build → filled by WeightLoader →
immutable during serving.

**Design rationale.** `gather_output=False` as the default is *the* Megatron
trick: Column→Row pairs need **zero communication at the seam** because the
column output is exactly the row input layout. Making the flag explicit (not
implicit in the pair) teaches where communication is and isn't needed.

---

## 3.10 RowParallelLinear

**Why it exists.** The second atom. Weight split by **input rows**: each rank
consumes a distinct slice of the input features and produces a **partial sum**
of the full output, which an **AllReduce** completes. Column and Row are dual:
column splits outputs and needs a gather to reconstruct; row splits inputs and
needs a reduce to reconstruct. Every TP pattern in the model is a sandwich of
the two.

**Responsibilities.**
- Hold weight shard `[out, in/tp]`; bias **full** but added only on tp_rank 0
  pre-reduce or by all ranks post-reduce (we choose: add after AllReduce, on
  all ranks, from a replicated bias — simplest correct form).
- Forward: local matmul on the sharded input `[.., in/tp]` → partial `[.., out]`,
  then `comm.all_reduce` → identical full output on every rank.
- `input_is_parallel=True` default: expect pre-split input from a preceding
  ColumnParallelLinear.

**Data owned.** Weight shard + replicated bias.

**Internal state.** None.

**Public API.** `forward(x_sharded) → y_full`, `shard_spec()`.

**Relationships.** Downstream half of both TP sandwiches (attention output
projection; MLP down-projection). The **only** module (besides embedding) that
calls `all_reduce` — a deliberate concentration: when you profile TP traffic,
you know exactly which two call sites in the model produce it.

**Lifecycle.** Same as ColumnParallelLinear.

**Design rationale.** The AllReduce happens *inside* the layer, not in the
caller. Alternative (caller-managed reduction) would let a future optimizer
fuse/defer reductions, but it spreads correctness responsibility across every
call site. For an educational engine, "RowParallelLinear returns a complete,
correct tensor, always" is the stronger property.

---

## 3.11 VocabParallelEmbedding (Distributed Embedding Layer)

**Why it exists.** The embedding table `[vocab, hidden]` is one of the largest
single tensors in a small model (GPT-2: 50257×768 ≈ 38M of 124M params) — not
sharding it wastes the most memory precisely where TP=2 hurts most. It also
teaches a communication pattern the linear layers don't: **shard by lookup key,
mask misses, AllReduce to merge** — a distributed hash-table join in miniature.

**Responsibilities.**
- Own vocab rows `[r·V/tp, (r+1)·V/tp)` of the (padded) embedding table.
- Forward: build a mask of "token ids in my range"; offset ids into local row
  indices; look up (out-of-range ids read row 0); zero out masked positions;
  `all_reduce` so every rank ends with the full `[batch, seq, hidden]`
  embeddings. Add (replicated) positional embeddings after the reduce.
- Exists only on the **first pipeline stage**.

**Data owned.** Vocab shard + full positional embedding table (small,
replicated — sharding it teaches nothing and complicates position handling).

**Public API.** `forward(token_ids) → hidden_states`.

**Relationships.** First module inside stage-0's PipelineStage. Uses
CommunicationManager. Its shard spec comes from TensorShardingManager.

**Design rationale.** Alternative: replicate the whole table. Simpler, and some
engines do it for small vocabs — but it forfeits both the memory win and the
mask-and-reduce lesson. We shard. (Weight *tying* with the LM head across
stages is explicitly dropped — §12.7.)

---

## 3.12 TensorParallelAttention

**Why it exists.** Attention is where TP stops being a math trick and becomes
an architectural constraint: heads are the natural sharding unit, and the KV
cache — the dominant *runtime* memory consumer — inherits the head sharding.
This module exists to compose the two linear atoms around head-local attention
**with zero communication in between**.

**Responsibilities.**
- QKV projection = ColumnParallelLinear sliced *by head groups*: this rank
  produces Q,K,V for its `heads/tp` heads only (GPT-2 example: 6 of 12 heads).
- Attention math (scores, softmax, weighted sum) runs **entirely locally** —
  heads are independent, so no cross-rank traffic.
- Write this step's K,V into **this rank's KV-cache slice** for its heads
  (cache tensors owned by ModelRunner, §3.15; the layer receives views).
- Output projection = RowParallelLinear: partial per-head outputs → AllReduce
  → full hidden states. **Net: one AllReduce per attention block.**

**Data owned.** Its two linear sub-modules' shards. *Not* the KV cache (views
only) — cache lifetime is tied to request scheduling, which layers must not
know about.

**Internal state.** None (position/cache-length arrive via ExecutionContext,
§8.9).

**Public API.** `forward(hidden, exec_ctx) → hidden`.

**Relationships.** Child of a TransformerLayer inside PipelineStage. Consumes
ExecutionContext for cache offsets and attention masks.

**Design rationale.** Head-aligned sharding is why
`heads % tp_size == 0` is a startup invariant, and why the fused QKV weight
must be sliced per-head-group rather than naively by rows (the most common
hand-rolled-TP bug; TensorShardingManager owns that subtlety). The
"zero communication between column and row halves" property is the central
exhibit of §9.

---

## 3.13 TensorParallelMLP

**Why it exists.** The second TP sandwich, and deliberately the *simplest*
one — implement it first, before attention. It also carries the "shard the
big intermediate dimension" lesson: the 4× expansion (`768 → 3072`) is exactly
what you want split, since the activation `[batch, seq, 3072]` is the largest
in the model.

**Responsibilities.**
- Up-projection: ColumnParallelLinear `[hidden → 4·hidden/tp]`, no gather.
- GELU on the local slice (elementwise ⇒ no communication — *why* elementwise
  ops commute with column sharding is a §9 exhibit).
- Down-projection: RowParallelLinear `[4·hidden/tp → hidden]` + AllReduce.
  **Net: one AllReduce per MLP block.**

**Data owned / state / API.** Two linear shards; stateless;
`forward(hidden) → hidden`.

**Design rationale.** With attention, gives the canonical "2 AllReduces per
transformer layer" cost model: total TP traffic per token per layer =
`2 × batch × seq × hidden × dtype_bytes`, the number every capacity-planning
discussion in industry starts from.

---

## 3.14 PipelineStage (and TransformerLayer, OutputLayer)

**Why it exists.** The unit of model ownership for one pp_rank: "the slice of
the network between my recv and my send." It gives the pipeline plane a single
handle — StageExecutor calls `stage.forward(...)` without knowing whether this
stage contains embeddings, plain layers, or the head. Position-dependent
structure is resolved *once at build time*, not branched on at runtime.

**Responsibilities.**
- At build (from ParallelContext + Config): first stage prepends
  VocabParallelEmbedding; every stage owns TransformerLayers
  `stage_layer_range(pp_rank)`; last stage appends final LayerNorm +
  OutputLayer.
- Forward: input is token ids (first stage) or received hidden states (other
  stages); output is hidden states (non-last) or logits fragment (last).
- Report its output shape per microbatch so CommunicationManager receives
  can be pre-sized (P4).

**OutputLayer (LM head)** — last stage only: a ColumnParallelLinear over the
vocab dimension: each rank produces logits for *its* vocab slice
`[batch, V/tp]`; the Sampler decides how to combine (§3.16). Kept as a
distinct module (not "just another linear") because its consumer is the
sampler, not another layer, and because vocab-parallel logits are the one
place where `gather_output` policy is a real decision.

**Data owned.** All its sub-modules' parameters — the stage is the root of the
parameter tree WeightLoader fills and verifies.

**Public API.** `forward(input, exec_ctx) → output`, `output_shape(ubatch) →
shape`, `named_local_parameters()`.

**Relationships.** Owned by ModelRunner; executed by StageExecutor; filled by
WeightLoader; queried by MicrobatchManager for shapes.

**Design rationale.** Uniform layer split (`num_layers % pp == 0` enforced) vs.
memory-balanced split: the first stage carries the embedding table, so uniform
split is slightly memory-imbalanced. Production engines rebalance
(vLLM: `--pipeline-partition`); we accept the imbalance and *measure it* with
DeviceManager's watermark — seeing the imbalance in your own metrics is worth
more than silently fixing it (§12.4).

---

## 3.15 ModelRunner

**Why it exists.** Something must own the assembled model shard and everything
needed to *execute a forward pass correctly on this rank*: KV cache allocation,
execution contexts, warmup. The pipeline plane (StageExecutor) should say
"run microbatch 2's decode step"; translating that into cache offsets, masks,
and a stage forward is ModelRunner's job. In vLLM this exact split
(Worker → ModelRunner → Model) exists for the same reason: it separates
*what to run* from *how to run it*.

**Responsibilities.**
- Build PipelineStage for this rank; invoke WeightLoader; run verification.
- **Own the KV cache**: preallocate, per local layer, tensors
  `[max_batch, local_heads, max_seq, head_dim]` — sharded across TP by heads,
  and existing only for this stage's layers (the cache is 2D-sharded exactly
  like the compute; this observation is a key learning outcome).
- Manage cache *slots*: a free-list of `max_batch` slot indices; the driver's
  ScheduleDirective assigns request→slot; ModelRunner just obeys.
- Build ExecutionContext per (microbatch, step): slot ids, positions,
  sequence lengths, prefill-vs-decode flag.
- `warmup()`: run one max-shape dummy prefill+decode before serving —
  materializes allocator pools and NCCL communicators so the first real
  request doesn't pay init cost (and so shape bugs fail at startup, not on
  request #1).

**Data owned.** PipelineStage (parameters), KV cache tensors, slot free-list.

**Internal state.** Slot occupancy, current cache lengths per slot.

**Public API.**
- `initialize()` (build → load → verify → warmup)
- `execute(ubatch, phase) → StageOutput` (hidden states or logits shard)
- `allocate_slots(directive) / release_slots(request_ids)`
- `cache_stats() → CacheStats`

**Internal API.** `_build_exec_context(ubatch, phase)`, `_alloc_kv_cache()`.

**Relationships.** Owned by Engine; called by StageExecutor; owns
PipelineStage; uses DeviceManager, WeightLoader, ParallelContext.

**Lifecycle.** `initialize()` at startup; serves for process lifetime; cache
freed at shutdown.

**Design rationale.** KV cache owned here — not by attention layers (their
lifetime is wrong: caches outlive a forward and are indexed by *request*), and
not by the Scheduler (wrong plane: the driver's scheduler has no tensors).
The slot abstraction is the deliberately-simplified stand-in for paged
attention: one contiguous region per request, allocated whole. The interface
(`allocate/release` slots driven by scheduler directives) is *shaped* so that
swapping in your paged-KV project later changes only ModelRunner internals.

---

## 3.16 Sampler

**Why it exists.** Sampling sits at an awkward topological point: logits are
vocab-sharded across the last stage's TP group, but a token id is one integer
that the *whole system* must agree on. The Sampler is the component that
resolves "distributed tensor → single consensus scalar" — a genuinely
distributed-systems problem (agreement), not a math problem.

**Responsibilities.**
- On the last stage: take this rank's logits shard `[batch, V/tp]` for each
  sequence's final position.
- Combine across TP: `all_gather` shards → full `[batch, V]` logits on every
  last-stage rank (chosen strategy; alternatives in rationale).
- Apply sampling policy per request (greedy; temperature; top-k) using the
  request's seeded generator so runs are reproducible.
- Produce `[batch]` token ids, identical on all last-stage TP ranks.

**Data owned.** Per-slot RNG generators (seeded from Request.seed at slot
allocation).

**Public API.** `sample(logits_shard, sampling_params, exec_ctx) → token_ids`.

**Relationships.** Called by StageExecutor on the last stage only. Uses
CommunicationManager (`all_gather`). Token ids then enter the pipeline's
feedback path (§10.7).

**Design rationale.** Three candidate strategies:
(a) **AllGather logits, all ranks sample with identical seeds** — chosen:
after the gather every rank has identical inputs and identical RNG state, so
they deterministically compute the same token with *no further communication*;
(b) gather-to-one, sample on tp_rank 0, broadcast token — less traffic for big
vocabs but adds a second communication step and an asymmetric code path;
(c) distributed top-k without full gather (production trick) — real engines do
this for 128k vocabs; noted in §12 as an optimization exercise. For GPT-2's
50k vocab at educational scale, (a)'s clarity wins. The subtle lesson —
*replicated RNG state is a form of consensus* — is called out in the docs.

---

## 3.17 MicrobatchManager

**Why it exists.** Pipeline parallelism's fuel is microbatches, and the split
must satisfy an unusual constraint: **every rank must derive the identical
split without communicating** (the directive broadcast tells them the batch
composition; the split must be a pure function of it). Centralizing the split
policy — and the shape bookkeeping that P4's static-shape protocol depends
on — in one stateless component makes that determinism auditable.

**Responsibilities.**
- Split a scheduled Batch of R requests into M microbatches (M from config;
  policy: contiguous chunks, sizes as equal as possible, deterministic order).
- For prefill: group by padded-length bucket so a short prompt isn't padded
  to the longest prompt in the whole batch (one bucket boundary, not a full
  length-sorted scheduler — enough to teach why padding policy matters).
- Compute, for every microbatch, the exact activation shapes each stage will
  send/recv at each phase — the shape oracle CommunicationManager reads.
- Assign stable `ubatch_id`s used in logs, metrics, and P2P op labels.

**Data owned / Internal state.** None — pure functions of
(ScheduleDirective, config). Statelessness ⇒ trivially identical across ranks.

**Public API.**
- `split(directive) → list[MicroBatch]`
- `activation_shape(ubatch, phase) → shape`

**Relationships.** Called identically on every rank by the Engine after the
directive broadcast. Output consumed by PipelineScheduler (counts),
StageExecutor (contents), CommunicationManager (shapes).

**Design rationale.** Alternative — rank 0 splits and broadcasts the split —
costs an extra control message per step and hides the determinism requirement
instead of teaching it. Making the split a pure function *is* the lesson:
in SPMD systems, shared pure functions over broadcast state replace
communication.

---

## 3.18 PipelineScheduler

**Why it exists.** The heart of the pipeline plane. Given (num_microbatches M,
num_stages P, my pp_rank), it emits the ordered sequence of primitive actions
this rank performs: `RECV(u) / COMPUTE(u) / SEND(u)`. Isolating this as a
**pure schedule generator** — no tensors, no communication, no torch — is the
project's showcase of P5: the entire fill-drain schedule, bubble structure and
all, is testable with plain asserts in milliseconds.

**Responsibilities.**
- Generate the **fill-drain (GPipe-style, forward-only)** schedule:
  stage s processes microbatches 1..M strictly in order; it starts at
  timestep s (fill), streams one per step (steady state), and finishes M+s
  (drain). Bubble fraction = (P−1)/(M+P−1) — the formula every design review
  of PP quotes, here made executable.
- Emit per-rank action lists with explicit peer ranks and ubatch ids attached.
- Provide `bubble_ratio(M, P)` for Metrics to report *theoretical* vs
  *measured* bubble.
- For the decode phase: generate the per-token schedule (same shape, M
  microbatches per token step) including the last→first token feedback action
  (§10.7).

**Data owned / state.** None — pure.

**Public API.**
- `actions(pp_rank, M, phase) → list[Action]`
- `bubble_ratio(M, P) → float`

**Relationships.** Consumed by StageExecutor, which is the interpreter of the
action list. Knows nothing of TP (P3's dividend: TP is invisible here).

**Design rationale.** Schedule-as-data (an action list interpreted by an
executor) vs schedule-as-control-flow (nested loops with send/recv inline):
the data form costs one extra abstraction but yields (a) unit-testable
schedules, (b) logs that print the *planned* schedule next to the *executed*
one, (c) drop-in room for an interleaved schedule later without touching the
executor. This mirrors how DeepSpeed structures its pipeline engine
(instruction lists), and it is the difference between an engine and a script.

---

## 3.19 StageExecutor

**Why it exists.** The interpreter that turns PipelineScheduler's abstract
actions into real work on this rank: `RECV` → CommunicationManager.recv into a
pre-sized buffer; `COMPUTE` → ModelRunner.execute; `SEND` → isend to the next
stage. It is the *only* place where pipeline communication and model execution
meet, which makes it the natural home for overlap (irecv posted before
compute) and for per-action timing.

**Responsibilities.**
- Execute the action list for one phase (prefill pass or one decode step),
  maintaining in-flight isend/irecv handles and waiting them at the right
  boundaries.
- First stage: `RECV` degenerates to "read token ids from the directive"
  (prefill) or "receive fed-back tokens" (decode). Last stage: `SEND`
  degenerates to "hand logits to Sampler".
- Record per-action wall time → Metrics (this is where measured bubble ratio
  comes from: gap between actions = idle).
- Enforce ordering invariants (a `COMPUTE(u)` must follow its `RECV(u)`;
  asserts catch schedule-generator bugs).

**Data owned.** In-flight communication handles; the small buffer pool of
activation tensors for outstanding microbatches (at most 2 in flight:
receiving u+1 while computing u).

**Internal state.** Current action index, handle table keyed by ubatch_id.

**Public API.** `run_phase(actions, microbatches, phase) → PhaseResult`
(last stage: sampled tokens per ubatch; others: empty).

**Relationships.** Called by Engine each phase. Uses CommunicationManager,
ModelRunner, Sampler (last stage), Metrics.

**Design rationale.** Keeping the executor dumb (interpret actions, own no
policy) means every pipeline bug is attributable: wrong *plan* →
PipelineScheduler (pure, unit-tested); wrong *execution of the plan* → here,
where verbose per-action logging (`[rank2] COMPUTE u3 done in 41ms; RECV u4
already complete`) makes the interleaving visible. Merging planner and
executor — the obvious shortcut — destroys exactly that attribution.

---

## 3.20 RequestManager

**Why it exists.** The boundary between the asynchronous outside world
(callers submitting prompts, waiting on futures, streaming tokens) and the
synchronous SPMD inside world (lock-step ranks). Every engine has this
airlock; without it, client concurrency leaks into rank-synchronous code,
which is lethal. Lives on the **driver (rank 0) only**.

**Responsibilities.**
- Accept submissions (prompt + SamplingParams), assign request ids, tokenize
  (the tokenizer lives here — CPU-only, driver-only), validate lengths against
  `max_seq_len`, reject oversized requests at the door.
- Own the waiting queue (FCFS) and the registry of in-flight requests
  (RequestState: tokens generated so far, per-request future/stream).
- Deliver results: complete futures, push stream deltas, detokenize.
- Track per-request timestamps (arrival, first-token, completion) → the
  latency metrics that matter (TTFT, TPOT).

**Data owned.** Waiting queue, in-flight registry, tokenizer, request futures.

**Internal state.** Queues/registry above; monotonically increasing request id
counter.

**Public API.** (called from client code / CLI thread)
- `submit(prompt, sampling_params) → RequestHandle` (future + token stream)
- `abort(request_id)`
- (called by Engine driver loop) `take_waiting(max_n) → list[Request]`,
  `deliver_tokens(request_id, token_ids)`, `finish(request_id, reason)`

**Relationships.** Feeds the Scheduler (via Engine). Downstream of nothing.
Thread-safety note: `submit` is called from a client thread; the engine loop
consumes on the driver thread — the queue is the one lock-protected structure
in the system, and this is documented as such.

**Lifecycle.** Driver-only; created at startup; drains on shutdown (rejects
new, finishes in-flight).

**Design rationale.** Tokenization on the driver, not on ranks: token ids are
small (broadcast cheaply on the control plane), tokenizers are CPU-bound and
stateful, and replicating them per rank buys nothing. The synchronous
`take_waiting` pull model (engine pulls when it wants a new round) keeps
*all* scheduling initiative in one loop — no callbacks firing mid-round.

---

## 3.21 Scheduler

**Why it exists.** Decides **what the whole machine does next**: which waiting
requests form the next batch round, with which slot assignments, prefill
lengths, and microbatch count. It is pure policy on the driver; its output —
the **ScheduleDirective** — is the single control-plane message that all ranks
obey for an entire round. Splitting policy (here) from mechanism (everything
downstream) is P2/P5 applied at the top of the system.

**Responsibilities.**
- **Round-based static batching** (deliberate scope choice): when the engine
  is idle and the queue is non-empty, take up to `max_batch_size` requests →
  one Batch that runs prefill + decode-to-completion before the next round.
- Build the ScheduleDirective (§8.3): request ids, token id lists, slot
  assignments, padded prefill length (bucketed), microbatch count, per-request
  max_new_tokens and stop conditions.
- During decode, decide per-step continuation: which sequences finished (EOS /
  length) — encoded in a tiny per-step StepDirective so ranks can skip
  finished slots' sampling but keep tensor shapes static (finished sequences
  are masked, not removed — shape stability again).
- Own SchedulerState (§8.11): what is running, what each slot holds.

**Data owned.** SchedulerState; scheduling policy parameters.

**Public API.** (driver-only, called by Engine)
- `schedule_round(waiting) → ScheduleDirective | None`
- `step(finished_mask, sampled) → StepDirective`
- `on_round_complete() → list[FinishedRequest]`

**Relationships.** Reads from RequestManager (via Engine), writes directives
that Engine broadcasts. Never touches tensors, communication, or torch —
fully unit-testable with fake requests.

**Design rationale.** Round-based batching wastes throughput when sequences
finish early (their slots idle until the round ends) — precisely the problem
continuous batching solves, and precisely your *other* project. The interface
is shaped for that upgrade: because ranks obey a per-step StepDirective
already, moving to continuous batching = changing driver-side policy + adding
join/leave actions to the directive, with zero changes to TP layers or the
pipeline executor. The boundary was drawn here *so that* the two projects
compose later.

---

## 3.22 Engine

**Why it exists.** The conductor. Owns the top-level lifecycle (§4.1) and the
serving loop, and is the **only component where driver/worker asymmetry is
visible**: rank 0 additionally runs RequestManager + Scheduler and broadcasts
directives; all ranks (including 0) then execute the identical SPMD path.
Concentrating the asymmetry in one file is what keeps every other component
symmetric and simple.

**Responsibilities.**
- Startup sequence: Config → DistributedRuntime → DeviceManager →
  ProcessGroupManager → ParallelContext → CommunicationManager →
  ModelRunner.initialize → Metrics/Logging → barrier("ready").
- Serving loop, per round:
  1. *(driver)* pull waiting requests, `scheduler.schedule_round` →
     directive; *(all)* `broadcast_control(directive)` — workers block here,
     which is how workers idle;
  2. *(all)* MicrobatchManager.split; ModelRunner.allocate_slots;
  3. *(all)* run prefill phase via StageExecutor;
  4. *(all)* decode loop: per token step — StageExecutor runs the step
     schedule; sampled tokens flow last→first stage (data plane) and
     last→driver (control plane); *(driver)* streams tokens out, computes
     finished mask; *(all)* broadcast StepDirective; repeat until round done;
  5. *(all)* release slots; *(driver)* finish requests; loop.
- Shutdown: broadcast shutdown sentinel; drain; teardown in reverse order.

**Data owned.** All top-level components (composition root — every object in
the process is reachable from the Engine, which makes ownership auditable).

**Internal state.** Engine phase (INIT/READY/SERVING/DRAINING/STOPPED);
current round context.

**Public API.** `Engine.launch(config)` (what `torchrun` invokes on every
rank); driver-side `submit` delegating to RequestManager (used by CLI/tests).

**Relationships.** Calls everything; called by nothing except the entrypoint.

**Design rationale.** One process playing driver+worker vs a separate frontend
process (vLLM's AsyncLLMEngine): the separate process buys client isolation at
the cost of an IPC layer — wrong trade for learning. The
"broadcast directive, then everyone including the driver runs the same code"
pattern is the cleanest known solution to SPMD serving and is exactly how
Megatron inference and vLLM's MP backend behave; internalizing it is a core
outcome of the project.

---

## 3.23 Metrics

**Why it exists.** In distributed systems, *the metrics are how you know it
works*. Correct output proves the math; only measurements prove the
architecture (is TP actually reducing latency? is the measured bubble ≈
(P−1)/(M+P−1)?). Metrics is also the designated **upward information channel**
(the one legal way data flows from lower planes to the operator) so components
never call upward.

**Responsibilities.**
- Per-rank in-memory registry: counters (tokens generated, bytes by comm-op
  type), histograms (per-action latency, TTFT, TPOT), gauges (memory
  watermark, slot occupancy).
- Timers used by StageExecutor/CommunicationManager with ~zero overhead when
  disabled.
- Derived pipeline stats: measured bubble ratio per phase (idle gaps ÷ phase
  wall time) vs theoretical.
- End-of-run (or periodic) aggregation: gather per-rank snapshots on the
  control plane → driver prints the consolidated report per round: TTFT,
  TPOT, tokens/s, per-stage utilization, TP AllReduce bytes, P2P bytes.

**Data owned.** The registry; nothing shared across ranks except at gather.

**Public API.** `counter(name).add(v)`, `histogram(name).observe(v)`,
`time_block(name)` context manager, `snapshot() → MetricsSnapshot`,
`(driver) aggregate(snapshots) → Report`.

**Relationships.** Sink for every plane; depends only on ParallelContext (rank
labels) and the control plane (gather).

**Design rationale.** In-memory + control-plane gather instead of
Prometheus/OTel: the learning content is *which* numbers matter and *where*
they're captured, not exporter plumbing. Metric names are a stable public
contract listed in one module — benchmarks (§13) are written against them.

---

## 3.24 Logging

**Why it exists.** Four processes interleaving on one terminal is chaos;
undisciplined logging is worse than none. The logging component makes the
*topology* visible in every line, which is the difference between logs you
can debug a deadlock with and logs you can't.

**Responsibilities.**
- Configure per-rank loggers with a mandatory prefix:
  `[t+12.031s rank2 pp1/tp0]` — wall-clock offset, global rank, mesh coords.
- Per-rank log *files* (`logs/rank_2.log`) plus console for rank 0 only by
  default (workers stay quiet on the console; `--log-all-ranks` flips it).
- Level policy: lifecycle events INFO; per-action pipeline events DEBUG;
  per-collective events TRACE (custom level) — so "turn on TRACE on rank 2"
  is the first debugging move for any hang.
- A `log_once` helper (many messages are identical across TP peers; printing
  them 4× teaches nothing).

**Data owned.** Logger configuration; file handles.

**Public API.** `get_logger(component) → Logger`, standard levels + `trace`.

**Relationships.** Used everywhere; configured by Engine at startup right
after ParallelContext exists (the prefix needs mesh coords).

**Design rationale.** Per-rank files are non-negotiable for deadlock
debugging: you diff the tails of rank files to see who is waiting on whom —
`tail logs/rank_*.log` after a hang, combined with the watchdog's
`last_op_description`, is the project's prescribed debugging ritual (§10.9).

---

# 4. Complete Request Lifecycle

## 4.1 Process Lifecycle (context for the request lifecycle)

Every rank passes through these phases; requests are only accepted in SERVING.

```
torchrun spawns 4 processes
        │
        ▼
  ┌───────────┐   Config.from_yaml + validate          (identical on all ranks)
  │   BOOT    │   DistributedRuntime.initialize        (NCCL + Gloo groups)
  │           │   DeviceManager.bind                   (cuda:LOCAL_RANK, dtype)
  └─────┬─────┘   ProcessGroupManager.build → ParallelContext → CommManager
        ▼
  ┌───────────┐   ModelRunner.initialize:
  │   LOAD    │     build PipelineStage (my layers only, my shards only)
  │           │     WeightLoader.load (mmap, slice, verify)
  └─────┬─────┘     allocate KV cache, warmup forward
        ▼
  ┌───────────┐   barrier("ready")
  │  SERVING  │   driver: start accepting; workers: block on directive bcast
  └─────┬─────┘   ← the request lifecycle below lives here
        ▼
  ┌───────────┐   driver broadcasts SHUTDOWN sentinel instead of a directive
  │ SHUTDOWN  │   drain in-flight round, reverse-order teardown, destroy groups
  └───────────┘
```

## 4.2 End-to-End Walkthrough

Scenario: client submits prompt "The history of Rome" (5 tokens),
`max_new_tokens=3`, greedy. Three other requests are already waiting, so the
round batches R=4 requests as M=2 microbatches (u0 = requests {A,B},
u1 = {C,D}). Topology as in the running example.

### Stage-by-stage narrative

**Step 1 — Submission (driver, client thread).**
`RequestManager.submit` tokenizes the prompt (5 token ids), validates length,
creates `Request{id=A, token_ids, sampling_params, seed, t_arrival}`, enqueues
it on the waiting queue, and returns a `RequestHandle` (future + token stream)
to the caller. Nothing distributed has happened; workers are still blocked in
`broadcast_control`, consuming no GPU cycles.

**Step 2 — Scheduling (driver, engine loop).**
The engine loop finds the previous round finished, pulls up to
`max_batch_size` waiting requests: `take_waiting(4) → [A,B,C,D]`.
`Scheduler.schedule_round` builds the **ScheduleDirective**: request ids,
their token id lists, KV slot assignments {A→0, B→1, C→2, D→3}, padded
prefill length (say bucket 8), microbatch count M=2, per-request stop
conditions. SchedulerState marks the round RUNNING.

**Step 3 — Directive broadcast (all ranks, control plane).**
`CommunicationManager.broadcast_control(directive, src=0)` over Gloo. This
single message is the *entire* control input for the round; everything each
rank does until the decode loop's first step is a pure function of
(directive, config, my mesh coordinates). Workers wake.

**Step 4 — Local planning (all ranks, no communication).**
Every rank independently and identically runs `MicrobatchManager.split` →
u0={A,B}, u1={C,D}, with the shape oracle: prefill activations will be
`[2, 8, 768]` per microbatch. `ModelRunner.allocate_slots` reserves KV slots.
`PipelineScheduler.actions(pp_rank, M=2, phase=PREFILL)` yields e.g. for
rank 2 (stage 1): `[RECV(u0), COMPUTE(u0), RECV(u1), COMPUTE(u1)]` — the
last stage sends nothing; its COMPUTE ends in the Sampler.

**Step 5 — Prefill, stage 0 (ranks 0 and 1, lock-step).**
StageExecutor interprets `COMPUTE(u0)`:
- VocabParallelEmbedding: each rank looks up its vocab half, masks misses,
  **AllReduce over TP group {0,1}** → both hold full embeddings `[2,8,768]`.
- Layers 0–5, each: TPAttention (QKV column-shard → local attention over 6
  heads → K,V written to slots 0–1 of this rank's cache for layers 0–5 → row
  output projection with **AllReduce**), then TPMLP (column up → GELU → row
  down with **AllReduce**). 12 AllReduces total across the 6 layers, all on
  the {0,1} group. Ranks 0 and 1 stay bit-identical after every reduce.
- `SEND(u0)`: rank 0 isends `[2,8,768]` to rank 2; rank 1 to rank 3 (each TP
  column has its own P2P chain — the tensors are identical; both send so that
  stage 1's ranks each have a local copy without cross-column traffic).
  Stage 0 immediately proceeds to `COMPUTE(u1)` — u1 computes while u0's
  send completes: **the pipeline is now full.**

**Step 6 — Prefill, stage 1 (ranks 2 and 3).**
`RECV(u0)` completes into a pre-sized `[2,8,768]` buffer (shape known from the
directive — no handshake). Layers 6–11 run identically (12 more AllReduces on
group {2,3}; K,V into stage-1's cache). Then final LayerNorm and
**OutputLayer**: column-parallel LM head → logits shard `[2, 25632]` per rank
(padded vocab 51264 / tp 2), *only for each sequence's last real token
position* (an inference-only optimization: prefill needs just the final
position's logits — noted in §12.9).

**Step 7 — First sampled token (ranks 2 and 3).**
`Sampler.sample`: **AllGather** shards → full `[2, 51264]` logits on both
ranks; temperature/greedy per request with per-slot RNG → token ids `[2]`,
identical on ranks 2 and 3. This is TTFT territory: the driver's
`t_first_token` for A is stamped when this token reaches it (step 8).
Meanwhile u1 flows through the same path one step behind.

**Step 8 — Token feedback (the decode loop's defining data motion).**
Sampled ids for u0 travel two paths simultaneously (§10.7):
- **Data plane:** last stage tp-column peers send token ids to the first
  stage (rank 2→0, rank 3→1) so stage 0 can embed them next step.
- **Control plane:** last-stage tp_rank 0 (rank 2) sends ids to the driver;
  `RequestManager.deliver_tokens` detokenizes and pushes stream deltas —
  the client sees " began" appear.

**Step 9 — Decode steps (all ranks, repeated).**
Per token step: driver broadcasts the tiny **StepDirective** (finished-mask —
initially all-running); every rank runs the decode schedule from
PipelineScheduler — same fill-drain shape but with seq_len=1 tensors
`[2,1,768]`: stage 0 embeds the fed-back token, attention runs against the
KV cache (positions from ExecutionContext), stage 1 produces the next token.
With M=2 microbatches, stage 0 works on u1's step while stage 1 works on
u0's — steady-state overlap. After 3 tokens, A's stop condition
(max_new_tokens) trips.

**Step 10 — Completion and round end (driver + all).**
Scheduler marks A finished in the step directive; A's slot output is masked
(shapes stay static) while other sequences continue. When all four finish,
the round ends: ranks release KV slots; the driver completes futures
(`finish(A, LENGTH)`), stamps `t_complete`, Metrics aggregates the round
(TTFT, TPOT, bubble ratio, bytes), and the engine loop returns to Step 2 for
the next waiting batch.

## 4.3 The Same Lifecycle as a Rank-Time Grid

Prefill with M=2, P=2 (columns = time →):

```
            t0            t1            t2
Stage 0   COMPUTE u0    COMPUTE u1      idle          ← drain bubble
(r0,r1)   └─isend u0─┐  └─isend u1─┐
Stage 1     idle      ▼ COMPUTE u0   ▼ COMPUTE u1
(r2,r3)   ← fill bubble  → sample u0    → sample u0/u1, feed back tokens
```

Bubble ratio = (P−1)/(M+P−1) = 1/3 here; raising M to 8 drops it to 1/9 —
run the benchmark in §13 and watch Metrics confirm the formula.

## 4.4 What Every Component Does per Lifecycle Phase (summary matrix)

| Phase | Driver-only components | All-rank components |
|---|---|---|
| Submission | RequestManager (tokenize, enqueue) | — (workers blocked in bcast) |
| Scheduling | Scheduler (directive), SchedulerState | — |
| Broadcast | Engine (src of bcast) | CommManager (control plane), Engine (rx) |
| Planning | — | MicrobatchManager, PipelineScheduler, ModelRunner (slots) |
| Prefill | — | StageExecutor, PipelineStage, TP layers, CommManager (AllReduce + P2P), Metrics |
| Sampling | — | Sampler + CommManager (AllGather) on last stage |
| Feedback | RequestManager (deliver/stream) | CommManager (P2P last→first; control last→driver) |
| Decode step | Scheduler (StepDirective) | same as prefill with seq=1 shapes |
| Completion | RequestManager (finish futures), Metrics (aggregate) | ModelRunner (release slots) |

---

# 5. Complete Data Flow

Five distinct kinds of data move through the system, on three transports.
Confusing them is the root of most distributed-engine bugs, so each gets its
own subsection and its own rules.

## 5.1 Transport Summary

| Data | Transport | Path | Size class | When |
|---|---|---|---|---|
| Requests / directives / sampled-token lists | Gloo control plane (pickled objects) | driver ↔ all | bytes–KB | round start, each decode step |
| Weights | Filesystem (mmap slice reads) | disk → each rank | MB–GB | startup only |
| Activations | NCCL P2P send/recv | stage s → s+1 (within tp column) | `b·s·h·2` bytes | every microbatch, every phase |
| TP partial results | NCCL collectives (AllReduce/AllGather) | within TP group | `b·s·h·2` bytes | 2× per layer + embed + sampler |
| Fed-back token ids | NCCL P2P (tiny int tensor) | last stage → first stage | bytes | every decode step |

## 5.2 Request Flow (control plane)

```
 client thread                     driver (rank 0)                    all ranks
──────────────                    ────────────────                   ──────────
 submit(prompt) ──► RequestManager
                    [tokenize]
                    waiting queue ──► Scheduler.schedule_round
                                      ScheduleDirective ═══ Gloo bcast ══► every rank
                                                                           (pure-function
                                                                            planning)
                    per decode step:  StepDirective     ═══ Gloo bcast ══► every rank
   token stream ◄── RequestManager ◄══ Gloo p2p ◄══ last-stage tp0 (sampled ids)
   future done  ◄── finish()
```

Rules: control messages are small, infrequent, and *never* carry tensors;
every control message is either driver→all (broadcast) or last-stage→driver
(point). There are no other control paths — auditable by grepping for
`broadcast_control`.

## 5.3 Weight Flow (startup only)

```
 model.safetensors (disk, shared fs)
        │  mmap — no full-file read
        ▼
 per rank: for each param of MY stage layers:
        HF name ──name map──► local param
        full shape ──TensorShardingManager──► ShardSpec (my slice)
        mmap[slice] ──read+cast──► layer tensor on cuda:{local_rank}
        ▼
 verify: every local param written exactly once; log LoadReport
```

Rank 2 (stage 1, tp 0) reads *only* layers 6–11's tensors, and of those only
the tp0 slices — ~¼ of checkpoint bytes per rank in the 2×2 example. There is
**no weight communication at any point**: the filesystem is the broadcast.

## 5.4 Activation Flow (data plane, one prefill microbatch)

```
 token ids [2,8]                 (from directive, all ranks of stage 0)
     │
     ▼  Stage 0 (ranks 0,1 in lock-step)
 VocabParallelEmbedding ──AllReduce{0,1}──► hidden [2,8,768] replicated
     │
     ▼  ×6 layers:
   TPAttention:  Column(QKV, per-head) → local attn (6 heads) ─┐
                 KV → cache slots                              │ Row(out proj)
                                                               ▼
                 ──AllReduce{0,1}──► hidden replicated
   TPMLP:        Column(768→1536 local) → GELU → Row(1536→768)
                 ──AllReduce{0,1}──► hidden replicated
     │
     ▼  isend [2,8,768]:  rank0 ──► rank2      rank1 ──► rank3
     ▼  Stage 1 (ranks 2,3): recv → ×6 layers (same pattern, group {2,3})
     │
     ▼  final LN → OutputLayer(Column, vocab-sharded, last position only)
 logits shard [2, 25632] per rank ──AllGather{2,3}──► [2, 51264] full
     │
     ▼  Sampler → token ids [2]   (identical on ranks 2,3)
        ├── NCCL p2p: rank2──►rank0, rank3──►rank1   (feedback for next step)
        └── Gloo p2p: rank2──►driver                 (streaming out)
```

The key invariant, worth internalizing: **hidden states are always fully
replicated within a TP group at layer boundaries** — sharding exists only
*inside* the column→row sandwiches. That is why a pipeline send is a plain
tensor send within each TP column, and why the two parallelism dimensions
never need to know about each other (P3).

## 5.5 Microbatch & Pipeline Flow (steady-state decode, M=2)

```
 time ──►        step k                    step k+1
 Stage 0:   embed u0 │ embed u1        embed u0 │ embed u1
                 ▲   send    send           ▲   send    send
                 │       ╲       ╲          │       ╲      ╲
 Stage 1:   compute u1 │ compute u0     compute u1 │ compute u0
            sample u1  │ sample u0      sample u1  │ sample u0
                 │                          │
                 └──── token ids u0/u1 ─────┘   (last→first feedback closes
                                                 the ring; §10.7)
```

Decode transfers are tiny (`[2,1,768]` ≈ 3 KB fp16), so decode PP overhead is
latency-dominated, not bandwidth-dominated — students should *see* this in
Metrics: P2P bytes collapse after prefill while P2P op counts stay high.

## 5.6 KV Cache Data (never moves)

The KV cache is written and read strictly locally: sharded by TP (head dim)
and by PP (layer dim). It never crosses a wire. Cache *coordinates* (slot ids,
lengths) move — inside directives on the control plane. This
"data stays, coordinates move" split is a recurring production pattern
(same idea as shard-local state + routed keys in any sharded store).

---

# 6. Component Relationships

## 6.1 Ownership & Call Graph (calls flow downward only)

```
                              Engine  (composition root — owns everything)
        ┌──────────────┬─────────┴────────┬──────────────────┐
        ▼ driver only  ▼ driver only      ▼ all ranks        ▼ all ranks
  RequestManager ──► Scheduler      MicrobatchManager   PipelineScheduler
   (queue, tokenizer)  (directives)  (pure split/shapes)  (pure action lists)
                                            │                  │
                                            ▼                  ▼
                                        StageExecutor  ◄── interprets actions
                                          │        │
                              ┌───────────┘        └───────────┐
                              ▼                                ▼
                         ModelRunner                  CommunicationManager
                    (KV cache, exec ctx)          (AllReduce/AllGather/P2P/bcast)
                              │                                │
                              ▼                                │
                        PipelineStage                          │
                   ┌────────┼──────────┐                       │
                   ▼        ▼          ▼                       │
             VocabParEmb  TransformerLayer  OutputLayer ── Sampler
                          (TPAttention,          all TP layers call ──┘
                           TPMLP = Column+Row)
                              ▲
              WeightLoader ───┘ (fills params once, uses TensorShardingManager)

  Foundation (injected everywhere, owns topology, never calls upward):
  Config → DistributedRuntime → DeviceManager → ProcessGroupManager → ParallelContext
  Sinks (receive from everyone): Metrics, Logging
```

## 6.2 Who Owns / Calls / Manages What (quick reference)

| Concern | Owner |
|---|---|
| Request queue, futures, tokenizer | RequestManager (driver) |
| What runs next (policy) | Scheduler (driver) |
| All scheduling *execution* | StageExecutor (every rank) |
| Model weights (shards) | TP layers, under PipelineStage |
| KV cache tensors + slots | ModelRunner |
| All bytes on the wire | CommunicationManager (sole user of torch.distributed) |
| Topology / rank math | ParallelContext (sole authority) |
| Process groups creation | ProcessGroupManager |
| Component lifecycles | Engine (startup/shutdown order) |
| Config invariants | Config (validated once, immutable) |

Communication responsibility: **only** CommunicationManager touches
`torch.distributed`. Scheduling responsibility: Scheduler decides *what*,
PipelineScheduler decides *in what order per rank*, StageExecutor *does it*.

---

# 7. Folder Structure

```
Tensor_And_Pipeline_Parallelism/
├── PLAN.md                      # this document
├── README.md                    # quickstart: torchrun commands, CPU mode
├── requirements.txt
├── configs/
│   └── gpt2_tp2_pp2.yaml        # one config per topology experiment
├── scripts/
│   ├── launch.sh                # torchrun --nproc-per-node=N entry
│   └── verify_vs_hf.py          # compare outputs token-for-token vs HuggingFace
├── engine/                      # the package
│   ├── config.py                # frozen dataclasses + validation (§3.1)
│   ├── runtime/
│   │   ├── distributed.py       # DistributedRuntime (§3.2)
│   │   ├── device.py            # DeviceManager (§3.3)
│   │   ├── process_groups.py    # ProcessGroupManager (§3.4)
│   │   └── parallel_context.py  # ParallelContext (§3.5)
│   ├── comms/
│   │   └── manager.py           # CommunicationManager — the ONLY torch.distributed user
│   ├── core/
│   │   ├── engine.py            # Engine: startup, serving loop, shutdown (§3.22)
│   │   ├── request_manager.py   # driver-only (§3.20)
│   │   ├── scheduler.py         # driver-only policy (§3.21)
│   │   ├── microbatch.py        # MicrobatchManager (§3.17)
│   │   └── datatypes.py         # Request, Batch, MicroBatch, directives (§8)
│   ├── parallel/
│   │   ├── tensor/
│   │   │   ├── column_linear.py
│   │   │   ├── row_linear.py
│   │   │   ├── embedding.py     # VocabParallelEmbedding
│   │   │   ├── attention.py     # TPAttention
│   │   │   └── mlp.py           # TPMLP
│   │   └── pipeline/
│   │       ├── schedule.py      # PipelineScheduler — pure, no torch (§3.18)
│   │       └── executor.py      # StageExecutor (§3.19)
│   ├── model/
│   │   ├── stage.py             # PipelineStage, TransformerLayer, OutputLayer
│   │   ├── runner.py            # ModelRunner + KV cache (§3.15)
│   │   ├── sharding.py          # TensorShardingManager — pure slicing math (§3.7)
│   │   ├── weight_loader.py     # HF checkpoint → shards (§3.8)
│   │   └── sampler.py           # (§3.16)
│   └── observability/
│       ├── logging.py           # rank-prefixed loggers, per-rank files
│       └── metrics.py           # counters/histograms + driver aggregation
├── tests/
│   ├── unit/                    # no distributed init: config, sharding math,
│   │                            #   schedule generation, microbatch split
│   └── distributed/             # gloo/CPU, world_size=2–4 via torchrun:
│                                #   TP layers vs dense reference, P2P, e2e tiny model
└── benchmarks/
    └── bubble_ratio.py          # measured vs theoretical (P−1)/(M+P−1)
```

Rules of thumb: `parallel/` never imports `core/` (mechanism below policy);
`comms/manager.py` is the only file importing `torch.distributed`;
`pipeline/schedule.py` and `model/sharding.py` import no torch at all —
they're pure and unit-tested first.

---

# 8. Internal Data Structures (compact reference)

| Structure | Key fields | Owner / lifetime |
|---|---|---|
| **Request** | id, token_ids, SamplingParams (max_new_tokens, temperature, top_k, seed), timestamps, state (WAITING/RUNNING/FINISHED) | RequestManager; submit → finish |
| **Batch** | round_id, list[Request], slot_map {req→kv_slot} | Scheduler; one round |
| **MicroBatch** | ubatch_id, request_ids, token tensor, padded_len, slot ids | MicrobatchManager (created per round, identical on all ranks) |
| **ScheduleDirective** | round_id, requests' token id lists, slot_map, padded_len, M, stop conditions | Built by Scheduler, broadcast to all; one round |
| **StepDirective** | step_idx, finished_mask | per decode step |
| **DeviceMesh** | grid[pp][tp]→rank, my (pp_rank, tp_rank) | ProcessGroupManager; process lifetime |
| **ParallelContext** | mesh + group handles + role predicates | process lifetime, immutable |
| **ShardSpec** | param_kind, full_shape, dim, slice, local_shape | TensorShardingManager; transient |
| **PipelineStage** | modules for layer range, output_shape() | ModelRunner |
| **ExecutionContext** | phase (PREFILL/DECODE), slot ids, positions, seq lens, attn mask info | ModelRunner; one forward |
| **SchedulerState** | waiting/running sets, slot occupancy | Scheduler (driver) |
| **Action** | kind (RECV/COMPUTE/SEND/FEEDBACK), ubatch_id, peer rank | PipelineScheduler; one phase |
| **MetricsSnapshot** | counters, histograms, per rank | Metrics; gathered per round |

---

# 9. Tensor Parallelism Design (essentials)

## 9.1 The Two Atoms and the Sandwich

```
ColumnParallel: W split by OUTPUT dim          RowParallel: W split by INPUT dim
  X (replicated) ──► [W_col0] → Y0 ┐             X0 ──► [W_row0] → partial ┐
  X (replicated) ──► [W_col1] → Y1 ┤ Y=[Y0|Y1]   X1 ──► [W_row1] → partial ┤ AllReduce → Y
  no comms needed (output stays split)           ONE AllReduce → full Y

THE TRICK: Column(out split) feeds Row(in split) with ZERO comms at the seam:

  replicated ─► Column ─► split ─► elementwise (GELU/attn) ─► split ─► Row ─► AllReduce ─► replicated
                                                                          ▲
                                                            only comm point in the sandwich
```

- **TPMLP** = Column(h→4h/tp) → GELU → Row(4h/tp→h). 1 AllReduce.
- **TPAttention** = Column(QKV, sliced per whole heads) → local attention on
  my heads (heads are independent ⇒ no comms) → Row(out proj). 1 AllReduce.
- **Total: 2 AllReduces per transformer layer**, each moving
  `batch·seq·hidden·2` bytes within the TP group.

## 9.2 Embedding and LM Head

```
VocabParallelEmbedding (first stage):        OutputLayer + Sampler (last stage):
  vocab rows split across ranks               Column over vocab → logits shard [b, V/tp]
  mask ids outside my range → lookup          AllGather → full logits [b, V] on all ranks
  → zero masked rows → AllReduce              identical seeded sampling on all ranks
  → full embeddings everywhere                → same token everywhere, no extra comm
```

## 9.3 Primitive Cheat Sheet

| Primitive | Used by | Why |
|---|---|---|
| AllReduce | RowParallel output, embedding merge | combine partial sums → replicated result |
| AllGather | Sampler (logits), Column with gather_output | concatenate shards → full tensor |
| Broadcast | control plane (directives), token→driver | one source, everyone follows |
| ReduceScatter | (unused; = AllReduce that leaves output sharded) | door to sequence parallelism |
| send/recv (P2P) | pipeline activations, token feedback | stage-to-stage relay |

Invariant to remember: **at every layer boundary, hidden states are fully
replicated within the TP group.** Sharding exists only inside sandwiches.

---

# 10. Pipeline Parallelism Design (essentials)

## 10.1 Stages and the Schedule

Stage k owns layers `[k·L/P, (k+1)·L/P)` plus embedding (first) / LM head
(last). Schedule: **fill-drain, forward-only** — each stage processes
microbatches strictly in order, offset by its stage index:

```
 M=4 microbatches, P=2 stages, time →
 Stage 0:  u0  u1  u2  u3  ·          bubble ratio = (P−1)/(M+P−1)
 Stage 1:  ·   u0  u1  u2  u3         here: 1/5.  More microbatches → smaller bubble,
           ▲               ▲          but smaller per-ubatch GPU efficiency. That
          fill            drain       tension IS pipeline parallelism.
```

Per-rank action lists come from the pure `PipelineScheduler`; `StageExecutor`
interprets them, posting `irecv(u+1)` before `compute(u)` so transfer overlaps
compute.

## 10.2 Stage Communication

- Each TP column is its own P2P chain (rank0→rank2, rank1→rank3): activations
  are already replicated within a TP group, so each stage-1 rank gets its copy
  from its column peer — no cross-column traffic ever.
- Shapes are never negotiated: both sides compute them from the broadcast
  directive (static-shape protocol, P4). Receiver pre-allocates; sender isends.

## 10.3 Closing the Ring: Decode Feedback

Autoregression makes inference PP a **ring**, not a line: the last stage's
sampled token must reach the first stage to be embedded next step.

```
  stage0 ──activations──► stage1 ──sample──┐
    ▲                                      │ token ids (tiny int tensor, P2P)
    └──────────────────────────────────────┘
                       └─── also Gloo → driver (streaming to client)
```

## 10.4 Debugging Ritual (when it hangs — and it will)

1. Watchdog prints each rank's `last_op_description` after T seconds.
2. `tail logs/rank_*.log` — diff the tails: whoever is waiting on a peer that
   already moved on has the schedule bug.
3. Reproduce on CPU/Gloo with world_size=2 and TRACE logging before touching
   GPUs.

---

# 11. Consolidated Architecture Diagrams

The key diagrams live inline where they're explained; index:

- Device mesh & rank layout — *Running Example* (top) and §2.3
- Five planes / layering — §2.4
- Process lifecycle — §4.1
- Rank-time pipeline grid — §4.3, §10.1
- Control/weight/activation/microbatch flows — §5.2–5.5
- Ownership & call graph — §6.1
- TP sandwich & primitives — §9.1–9.3
- Decode ring — §10.3

---

# 12. Design Decisions (summary table)

| # | Decision | Alternative | Why ours |
|---|---|---|---|
| 1 | SPMD + rank-0 driver, directive broadcast | Separate coordinator process (Ray/RPC, à la vLLM) | one process model, zero IPC layer; asymmetry confined to Engine |
| 2 | Separate Gloo control plane | control msgs over NCCL | NCCL can't carry objects; mixing planes causes deadlocks |
| 3 | Static shapes on the wire | shape handshakes per transfer | no dynamic protocol to deadlock; forces shape discipline (TensorRT-LLM's lesson) |
| 4 | TP contiguous ranks (TP inner, PP outer) | PP inner | TP is comm-heavy → needs NVLink; PP tolerates slow links |
| 5 | Fill-drain forward-only schedule | 1F1B, interleaved stages | those exist for training (backward); inference needs only this |
| 6 | Every rank mmap-reads its own weight slice | rank-0 load + scatter | zero startup comms, works with shared fs; scatter is an extension |
| 7 | AllGather logits + replicated-seed sampling | gather-to-one + bcast token | one comm step, symmetric code; "replicated RNG = consensus" lesson |
| 8 | Round-based static batching | continuous batching | that's your other project; interfaces (slot + StepDirective) designed so it can be dropped in |
| 9 | Contiguous KV slots | paged KV | ditto — swap lives entirely inside ModelRunner |
| 10 | AllReduce inside RowParallel layer | caller-managed reductions | layers always return complete tensors — locality of correctness |
| 11 | Pure schedule/split/sharding functions | inline logic | unit-testable without GPUs; SPMD determinism made auditable |
| 12 | Uniform layer split per stage | memory-balanced split | accept + *measure* the embedding imbalance instead of hiding it |

---

# 13. Implementation Roadmap

1. **Foundations (no GPU needed):** config + validation; ProcessGroupManager +
   ParallelContext on Gloo; CommunicationManager; unit tests.
2. **TP layers:** Column/Row linear vs dense reference on CPU world_size=2;
   then embedding, MLP, attention; sharding math + assemble-tests.
3. **Single-stage engine (PP=1):** PipelineStage, WeightLoader (real GPT-2),
   ModelRunner + KV cache, Sampler → verify token-for-token vs HuggingFace.
4. **Pipeline:** pure PipelineScheduler (test schedules as data), StageExecutor,
   P2P activations, decode feedback ring → TP=1,PP=2 matches HF too.
5. **2D + serving:** TP=2×PP=2; RequestManager, Scheduler, directives, engine
   loop, streaming.
6. **Observability & benchmarks:** metrics report, bubble-ratio benchmark,
   TP-scaling benchmark.

Each milestone ends with the same acceptance test: *output identical to
HuggingFace GPT-2 greedy decoding.*

---

# 14. Glossary

**TP group** — ranks sharing one layer's shards (AllReduce domain). ·
**PP group** — one TP column's stage chain (send/recv). ·
**Driver** — rank 0's extra role: queue + scheduler + broadcasts. ·
**Directive** — the broadcast control object making all ranks agree. ·
**Microbatch** — pipeline scheduling unit; more ⇒ smaller bubble. ·
**Bubble** — stage idle time during fill/drain, (P−1)/(M+P−1). ·
**Fill-drain** — forward-only GPipe-style schedule. ·
**Slot** — a request's preallocated KV-cache region. ·
**Control/data plane** — Gloo objects vs NCCL tensors, never mixed.






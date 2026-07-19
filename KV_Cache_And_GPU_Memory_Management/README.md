```
kv_engine_project/
│
├── frontend/                   # 1. THE TEXT & NETWORK ENGINE
│   ├── __init__.py
│   ├── api_server.py           # Accepts OpenAI-compatible JSON requests.
│   └── tokenizer_worker.py     # Converts strings to token IDs in a separate CPU process.
│
├── shared/                     # 2. GLOBAL CONFIGURATIONS & ARBITRATION
│   ├── __init__.py
│   ├── system_profiler.py      # THE SHIELD: Runs before the memory allocator. Loads the 4-bit model weights first, measures exact remaining VRAM, and calculates the maximum safe page count.
│   └── layout_config.py        # The Single Source of Truth: Defines the strict 5D memory layout to guarantee contiguous memory coalescing for the PagedAttention kernel.
│
├── gpu_memory/                 # 3. THE BARE METAL MEMORY MANAGER
│   ├── __init__.py
│   ├── physical_allocator.py   # Takes the green-lit number of pages from the profiler and executes a single, safe `torch.zeros()` allocation. Completely blind to text/tokens.
│   ├── free_list.py            # A low-level integer stack with a pop/push API for physical page IDs.
│   └── swap_manager.py         # Manages asynchronous memory copies over the PCIe lane to offload inactive tensors to CPU RAM without stalling the GPU.
│
├── kv_cache/                   # 4. THE LLM BRAIN (Virtual Mapping)
│   ├── __init__.py
│   ├── radix_tree.py           # Hashes token sequences to point identical system prompts to the same physical page ID, preventing duplicate storage.
│   ├── block_table.py          # The Virtual MMU: maps logical token indices to the non-contiguous physical page IDs.
│   └── layer_coordinator.py    # Applies precise structural coordinate logic needed to compute vector stride jumps based on the target execution layer.
│
├── scheduler/                  # 5. THE TRAFFIC COP
│   ├── __init__.py
│   ├── batch_state.py          # Manages request state transitions (WAITING, RUNNING, SWAPPED).
│   └── preemptor.py            # The Policy Engine: When the free list has < 5% pages left, decides WHICH sequence to evict and instructs the swap manager to issue async copies.
│
└── execution/                  # 6. THE RUNTIME MATH KERNELS
    ├── __init__.py
    ├── model_runner.py         # Handles loading the model weights into VRAM and stepping through the network layers.
    └── custom_attention.py     # Implements PagedAttention to read scattered historical page maps and calculate attention without requiring contiguous memory allocation.

```

### The Final Directory Structure (The 6 Pillars)

This structure enforces a textbook separation of concerns, perfectly splitting the responsibilities between state tracking, virtual-to-physical memory mapping, and hardware execution.

* **1. `frontend/` (The Gateway)**
* 
`api_server.py`: Accepts OpenAI-compatible JSON requests.


* 
`tokenizer_worker.py`: Runs in a completely separate CPU process to translate strings to token IDs, preventing the GPU from waiting on slow string manipulations.




* **2. `shared/` (Global Configs & Arbitration)**
* 
`system_profiler.py`: **The Shield.** This runs before the memory allocator to load the 4-bit model weights, precisely measure the exact remaining VRAM, and calculate the maximum safe page count.


* 
`layout_config.py`: **The Single Source of Truth.** Defines the strict 5D memory layout (e.g., `[Max_Pages, Num_Layers, 2, Num_KV_Heads, Block_Size, Head_Dim]`) to guarantee contiguous memory coalescing for the attention kernel.




* **3. `gpu_memory/` (The Bare Metal GPU Manager)**
* 
`physical_allocator.py`: Takes the green-lit page count from the profiler and executes a single, safe `torch.zeros()` allocation. It remains completely blind to LLM concepts like text or tokens.


* 
`free_list.py`: A low-level integer stack with a `pop_free_page_id()` and `push_free_page_id()` API for managing physical page IDs.


* 
`swap_manager.py`: Manages a pool of asynchronous `torch.cuda.Stream()` objects to offload inactive user tensors to CPU RAM over the PCIe lane without stalling the GPU.




* **4. `kv_cache/` (The Virtual LLM Brain)**
* 
`radix_tree.py`: Hashes token sequences (Prefix Caching) to point identical system prompts to the same physical page ID, preventing duplicate storage.


* `block_table.py`: The Virtual MMU. It maps logical token indices to the non-contiguous physical page IDs assigned by the bare-metal allocator.


* 
`layer_coordinator.py`: Applies the precise structural coordinate logic needed to compute vector stride jumps based on the target execution layer.




* **5. `scheduler/` (The Traffic Cop)**
* 
`batch_state.py`: Manages request state transitions such as `WAITING`, `RUNNING`, and `SWAPPED`.


* `preemptor.py`: The Policy Engine. When the free list drops below a critical threshold (e.g., < 5%), this applies a strict scheduling policy (like FCFS) to intelligently swap out lower-priority requests.




* **6. `execution/` (The Runtime Math Kernels)**
* 
`model_runner.py`: Handles loading the model weights into VRAM and stepping through the network layers sequentially.


* 
`custom_attention.py`: Implements a PagedAttention kernel that reads scattered historical page maps provided by the block table and calculates attention without requiring contiguous memory.





---

### The Critical "Orchestrator" Startup Sequence

A beautiful directory tree will still crash if it doesn't boot up in the exact right mathematical sequence. Your root file (`orchestrator.py` or `main.py`) **must** execute these specific operational handshakes:

1. **The Weight-First Boot (Preventing the OOM Crash):**
You cannot allocate the KV cache first. The orchestrator must load the 4-bit model weights into the `model_runner` first, force PyTorch to clear temporary loading spikes, and then query the hardware for its total physical limit.
*Crucial Math Fix:* You must use `total_vram - used_vram - safety_buffer` to calculate the true remaining memory, rather than using `allocated_bytes.all.current` (which would cause a 9.0 GB spike and crash the 8GB card).


2. **The Thread-Safe Queue:**
To safely cross the CPU-to-GPU boundary, the frontend must emit tokenized requests into a thread-safe `queue.Queue()`. Only the scheduler is allowed to pop from this queue and touch the KV cache logic.


3. **Asynchronous Swap Synchronization:**
When the scheduler swaps a batch back in, it must use `torch.cuda.Event().synchronize()`. The scheduler must wait for the CUDA events to fire before marking a batch as "ready," preventing the GPU from reading garbage memory before the PCIe swap finishes.
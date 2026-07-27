from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    """
    Holds the physical architectural specs of the LLM.
    
    These parameters reflect the shape of the model's internal attention weight
    matrices and directly dictate the shape and VRAM memory footprint of the
    Key-Value (KV) cache tensors allocated at runtime.
    """

    num_layers: int  
    """
    Total Transformer layers in the network (e.g., 30 for SmolLM2-135M).
    Because KV caching must preserve attention history at every single layer,
    this value determines how many layer slices exist in the physical page pool.
    """

    num_kv_heads: int  
    """
    Number of Key-Value attention heads per layer.
    Modern architectures often use Grouped-Query Attention (GQA) or Multi-Query
    Attention (MQA) where num_kv_heads < num_query_heads to drastically reduce
    the memory footprint of the KV cache without losing model accuracy.
    """

    head_dim: int  
    """
    Dimension/length of the feature vector stored per individual head (e.g., 64).
    Calculated as: hidden_size // num_attention_heads.
    """

    dtype: torch.dtype = torch.float16  
    """
    Numeric precision format for cache tensors. Defaults to 16-bit float (float16)
    to balance numeric precision with half-precision memory efficiency on modern GPUs.
    """

    @property
    def dtype_bytes(self) -> int:
        """
        Calculates the exact byte size of a single numeric scalar in VRAM.
        
        Example: torch.float16 uses 16 bits.
        torch.finfo(torch.float16).bits // 8 = 16 // 8 = 2 bytes per float.
        """
        return torch.finfo(self.dtype).bits // 8


@dataclass
class EngineConfig:
    """
    Holds operational safety limits and runtime execution limits.
    
    Acts as the engine's control boundary to prevent GPU Out-Of-Memory (OOM) 
    crashes and manage maximum throughput during parallel batching.
    """

    max_num_seqs: int = 8  
    """
    Maximum number of concurrent user requests/prompts the engine is allowed 
    to execute together in a single batched step.
    """

    max_num_batched_tokens: int = 1024  
    """
    Upper bound on total combined tokens (across all active prompts) allowed
    in a single prefill forward pass. Prevents massive activation memory spikes
    when multiple long prompts arrive simultaneously.
    """

    watermark_backoff: float = 0.01  
    """
    A 1% safety memory buffer. If remaining free GPU VRAM drops below this fraction,
    the Scheduler temporarily halts admitting new prompts into the RUNNING state
    to guarantee active sequences have enough headroom to finish decode passes without hitting OOM.
    """

    gpu_memory_utilization: float = 0.9  
    """
    Fraction of free GPU memory to reserve for the KV cache page pool (e.g., 90%).
    The remaining 10% unallocated headroom is left specifically for dynamic PyTorch
    workspace allocations, execution graphs, and intermediate activation tensors.
    """


@dataclass
class CacheConfig:
    """
    Holds the computed physical dimensions and memory capacity of the allocated
    VRAM page pool.
    
    Instantiated at boot time inside profiler.py after executing dummy forward passes.
    """

    num_pages: int  
    """
    Total count of physical 16-token memory blocks allocated inside the GPU page pool.
    """

    dtype: torch.dtype  
    """
    Data precision format used for tensors inside the pool (matches ModelConfig.dtype).
    """

    block_size: int  
    """
    Fixed number of sequence tokens contained within a single physical block/page (typically 16).
    A non-fragmented block allocation strategy eliminates memory fragmentation.
    """

    page_bytes: int  
    """
    The exact physical VRAM footprint (in bytes) required to store a single page across ALL model layers.
    
    Calculated via the explicit Section 2.6 formula:
    page_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_bytes
    (The multiplier '2' accounts for storing both Key and Value vectors).
    """
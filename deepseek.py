"""Deepseek File"""

class MLA(nn.Module):
    """
        Multi-Headed Latent Attention

        Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems
        q_lora_rank (int): Rank for low-rank query projection
        kv_lora_rank (int): Rank for low-rank key/value projection
        qk_nope_head_dim(int): Dimensionality of non-positional query/key projections
        qk_rope_head_dim(int): Dimensionality of rotary-positional query/key projections
        qk_head_dim (int): Total dimensionality of query/key projections
        v_head_dim (int): Dimensionality of value projections
        softmax_scale (float): Scaling factor
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)


















"""Deepseek File"""

class MLA(nn.Module):
    """
        Multi-Headed Latent Attention

        Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems
        q_lora_rank (int): Rank for low-rank query projection
    """

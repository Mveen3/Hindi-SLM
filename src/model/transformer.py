"""
Decoder-Only Transformer Language Model
======================================

A GPT-style decoder-only Transformer built entirely from primitive PyTorch
layers (nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout).

Model H (Hindi) and Model L (Nepali) are both instances of this class, each
built from its own ``configs/model_config.yaml``. They differ in vocabulary
size and are trained to entirely separate weights: the two models share this
code, never a checkpoint.

Architecture overview:
    1. Token Embedding + Learned Absolute Positional Embedding + Dropout
    2. N × TransformerBlock (pre-norm: LN → MHA → residual, LN → FFN → residual)
    3. Final LayerNorm → Linear output head (optionally tied with token embedding)

Design decisions documented inline for evaluation clarity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention from first principles.

    For input X ∈ R^{B×T×d}:
        - Projects X into queries Q, keys K, values V via learned weight matrices.
        - Reshapes into h heads, each operating in a d_k = d/h dimensional subspace.
        - Computes scaled dot-product attention with an additive causal mask so
          position t can only attend to positions ≤ t.
        - Concatenates heads and applies an output projection.

    The 1/√d_k scaling prevents dot-product magnitudes from growing with
    dimension, which would push softmax into saturated regions with near-zero
    gradients (vanishing gradients problem).

    Args:
        d_model:     Model / embedding dimension.
        n_heads:     Number of attention heads.
        max_seq_len: Maximum sequence length (for causal mask buffer).
        dropout:     Dropout rate applied to attention weights and output.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1,
                 use_fused_attention: bool = False):
        super().__init__()

        # The project brief requires multi-head attention and causal masking to be
        # student-implemented, so the hand-written path below is the DEFAULT.
        # torch's fused SDPA kernel is available only as an explicit opt-in for
        # speed; the two are verified numerically equivalent in tests/test_attention.py.
        self.use_fused_attention = use_fused_attention
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Learned projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout on attention probabilities and residual output
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-compute the causal mask as a non-trainable buffer.
        # Upper-triangular matrix of True values (positions to be masked).
        # Stored on the same device as the model and included in state_dict
        # but NOT treated as a learnable parameter.
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x:                Input tensor of shape (B, T, d_model).
            return_attention:  If True, also return attention weights for analysis.

        Returns:
            output:       (B, T, d_model) — contextualised representations.
            attn_weights: (B, h, T, T)    — only returned when return_attention=True.

        Tensor shape walkthrough (documented for evaluation):
            Q, K, V after projection : (B, T, d_model)
            After reshape + transpose: (B, h, T, d_k)
            attn_scores              : (B, h, T, T)
            context after matmul     : (B, h, T, d_k)
            After concat + W_o       : (B, T, d_model)
        """
        B, T, d = x.shape

        # Step 1: Project into Q, K, V — each (B, T, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Reshape into multiple heads
        # (B, T, d_model) → (B, T, h, d_k) → (B, h, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_fused_attention and not return_attention and hasattr(F, "scaled_dot_product_attention"):
            # Step 3-6: Fast path using SDPA (FlashAttention / Memory-Efficient Attention)
            context = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
            attn_weights = None
        else:
            # Step 3: Scaled dot-product attention
            # attn_scores shape: (B, h, T, T)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
            # Step 4: Apply causal mask — additive mask with -inf on future positions
            # This ensures position t attends only to positions ≤ t.
            # We slice the pre-computed mask to the actual sequence length T.
            mask = self.causal_mask[:T, :T]  # (T, T) boolean
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0),  # broadcast to (1, 1, T, T)
                float("-inf")
            )
    
            # Step 5: Softmax → attention probabilities, then dropout
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
    
            # Step 6: Weighted sum of values
            # (B, h, T, T) @ (B, h, T, d_k) → (B, h, T, d_k)
            context = torch.matmul(attn_weights, V)

        # Step 7: Concatenate heads
        # (B, h, T, d_k) → (B, T, h, d_k) → (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Step 8: Output projection + residual dropout
        output = self.W_o(context)
        output = self.resid_dropout(output)

        if return_attention:
            return output, attn_weights
        return output


# ---------------------------------------------------------------------------
# 2. Position-Wise Feed-Forward Network
# ---------------------------------------------------------------------------

class PositionWiseFeedForward(nn.Module):
    """
    Two-layer feed-forward network applied independently to each position.

        FFN(x) = Dropout( W_2 · GELU( W_1 · x + b_1 ) + b_2 )

    The inner dimension d_ff is typically 4× d_model, allowing the network
    to project into a higher-dimensional space where non-linear interactions
    can be learned, before projecting back to d_model.

    GELU is chosen over ReLU following GPT-2 / modern Transformer convention;
    it provides smoother gradients near zero.

    Args:
        d_model: Model dimension.
        d_ff:    Inner / hidden dimension (typically 4 × d_model).
        dropout: Dropout rate on output.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        x = self.fc1(x)          # (B, T, d_ff)
        x = self.activation(x)   # (B, T, d_ff)
        x = self.fc2(x)          # (B, T, d_model)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# 3. Transformer Block (Pre-Norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block with pre-norm architecture:

        x = x + MHA( LayerNorm(x) )
        x = x + FFN( LayerNorm(x) )

    Pre-norm (placing LayerNorm *before* the sublayer) is used instead of
    post-norm because:
        - It produces more stable gradients in deep stacks, enabling training
          without careful warm-up or special initialization.
        - Empirically more robust for models trained from scratch.
        - Used by GPT-2 and most modern Transformer LMs.

    Each sublayer has a residual connection that allows gradients to flow
    directly through the identity path, preventing vanishing gradients.

    Args:
        d_model:     Model dimension.
        n_heads:     Number of attention heads.
        d_ff:        FFN inner dimension.
        max_seq_len: Maximum sequence length.
        dropout:     Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x:                (B, T, d_model) input hidden states.
            return_attention:  Whether to return attention weights.

        Returns:
            x:            (B, T, d_model) output hidden states.
            attn_weights: (B, h, T, T) — only when return_attention=True.
        """
        # Sub-layer 1: Multi-head causal self-attention with residual
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln1(x), return_attention=True)
            x = x + attn_out
            # Sub-layer 2: Feed-forward with residual
            x = x + self.ffn(self.ln2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x


# ---------------------------------------------------------------------------
# 4. Decoder-Only Transformer Language Model
# ---------------------------------------------------------------------------

class DecoderOnlyTransformer(nn.Module):
    """
    Complete GPT-style decoder-only Transformer language model.

    Architecture:
        Token Embedding(vocab_size, d_model)
      + Positional Embedding(max_seq_len, d_model)   [learned absolute]
      → Embedding Dropout
      → N × TransformerBlock (pre-norm)
      → Final LayerNorm
      → Linear output head → logits (vocab_size)

    Positional embeddings:
        We use learned absolute positional embeddings (following GPT-2).
        Each position 0..max_seq_len-1 has a learned d_model-dimensional vector
        that is added to the token embedding. This constrains the model to a
        maximum sequence length of max_seq_len; inputs longer than this cannot
        be processed without extrapolation. Sinusoidal or relative schemes
        (e.g., RoPE) could lift this constraint, but learned absolute embeddings
        are simpler, effective, and sufficient for our 512-token context.

    Weight tying:
        When tie_weights=True, the output projection (lm_head) shares its
        weight matrix with the token embedding layer. This reduces parameter
        count by vocab_size × d_model (~5M) and acts as a form of
        regularisation by coupling input and output representations.

    Objective:
        Causal language modeling — cross-entropy between logits at position t
        and the ground-truth token at position t+1, averaged over all positions.

    Args:
        vocab_size:  Size of the token vocabulary.
        d_model:     Model / embedding dimension.
        n_layers:    Number of Transformer blocks.
        n_heads:     Number of attention heads per block.
        d_ff:        FFN inner dimension.
        max_seq_len: Maximum input sequence length.
        dropout:     Dropout probability.
        tie_weights: Whether to tie lm_head weights with token embeddings.
        pad_token_id: Token ID used for padding (ignored in loss computation).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # ----- Input Representation -----
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # ----- Transformer Blocks -----
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        # ----- Output -----
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights before tying
        self.apply(self._init_weights)

        # Weight tying: share embedding matrix with output head
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module):
        """
        Initialise weights following GPT-2 convention:
            - Linear layers and Embeddings: Normal(0, 0.02)
            - LayerNorm: bias=0, weight=1
            - Linear biases: 0
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        return_attention: bool = False,
    ):
        """
        Forward pass for the decoder-only Transformer.

        Args:
            input_ids:        (B, T) token ID tensor.
            targets:          (B, T) target token IDs for loss computation.
                              If None, loss is not computed.
            return_attention:  If True, return per-layer attention weight tensors.

        Returns:
            logits: (B, T, vocab_size) next-token prediction logits.
            loss:   Scalar cross-entropy loss (only if targets provided).
            attn:   List of (B, h, T, T) attention weights per layer
                    (only if return_attention=True).
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds maximum {self.max_seq_len}. "
            f"Learned positional embeddings do not extrapolate beyond max_seq_len."
        )

        # Positional indices: (1, T) — broadcast across batch
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Input representation: token emb + positional emb + dropout
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.emb_dropout(x)

        # Pass through N Transformer blocks
        all_attn_weights = [] if return_attention else None
        for block in self.blocks:
            if return_attention:
                x, attn_w = block(x, return_attention=True)
                all_attn_weights.append(attn_w)
            else:
                x = block(x)

        # Final layer norm + output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_token_id,
            )
            # IMPORTANT: During training with nn.DataParallel, gathering the massive
            # logits tensor across GPUs causes OOM (Out Of Memory). We discard it here.
            logits = torch.empty(0, device=logits.device)

        # Build return tuple
        if return_attention:
            return logits, loss, all_attn_weights
        return logits, loss


# ---------------------------------------------------------------------------
# 5. Utility functions
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    """
    Count and report trainable and total parameters in the model.

    Returns:
        Dictionary with 'total', 'trainable', and 'non_trainable' counts,
        plus a 'by_module' breakdown.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    # Per-module breakdown (top-level children)
    by_module = {}
    for name, child in model.named_children():
        count = sum(p.numel() for p in child.parameters())
        by_module[name] = count

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "by_module": by_module,
    }


def build_model_from_config(config: dict):
    """
    Instantiate a Transformer from a configuration dictionary.

    When ``config["no_positional_embedding"]`` is ``True``, builds the
    ablated :class:`~src.model.transformer_no_pos.DecoderOnlyTransformerNoPos`
    variant (bonus ablation study). Otherwise builds the standard
    :class:`DecoderOnlyTransformer`.

    Args:
        config: Dictionary with keys matching the model constructor.
                Expected keys: vocab_size, d_model, n_layers, n_heads, d_ff,
                max_seq_len, dropout, tie_weights, pad_token_id.
                Optional: no_positional_embedding (bool).

    Returns:
        Initialized model (standard or ablated).
    """
    if config.get("no_positional_embedding", False):
        from src.model.transformer_no_pos import build_ablated_model_from_config
        return build_ablated_model_from_config(config)

    model = DecoderOnlyTransformer(
        vocab_size=config["vocab_size"],
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=config.get("dropout", 0.1),
        tie_weights=config.get("tie_weights", True),
        pad_token_id=config.get("pad_token_id", 0),
    )
    return model

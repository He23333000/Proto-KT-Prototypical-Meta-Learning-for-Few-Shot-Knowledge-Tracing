"""
Self-Attentive Knowledge Tracing (SAKT) model.

Based on: "A Self-Attentive Model for Knowledge Tracing" (Pandey & Karypis, 2019)
Implementation follows the architecture described in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with causal masking for Knowledge Tracing.
    
    This implements the scaled dot-product attention mechanism from "Attention is All You Need".
    Multiple attention heads allow the model to attend to different aspects of the sequence simultaneously.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize multi-head attention module.
        
        Args:
            embed_dim (int): Dimensionality of input/output embeddings (e.g., 128)
            num_heads (int): Number of parallel attention heads (e.g., 8)
            dropout (float): Dropout probability for regularization
        """
        super().__init__()
        # Ensure embedding dimension can be evenly split across heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim  # Total embedding dimension
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = embed_dim // num_heads  # Dimension per head (e.g., 128/8 = 16)
        
        # Linear projections for Query, Key, Value (Q, K, V)
        # Input: (batch, seq_len, embed_dim) -> Output: (batch, seq_len, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)  # Query projection
        self.k_linear = nn.Linear(embed_dim, embed_dim)  # Key projection
        self.v_linear = nn.Linear(embed_dim, embed_dim)  # Value projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # Output projection after concatenating heads
        
        # Dropout for regularization (applied to attention weights)
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot products: 1/sqrt(d_k) prevents softmax saturation
        # This stabilizes gradients when dimensions are large
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None, causal_mask=True, position_bias=None):
        """
        Compute multi-head attention over input sequences.
        
        Args:
            query: (batch, seq_len, embed_dim) - Query vectors (what we're looking for)
            key: (batch, seq_len, embed_dim) - Key vectors (what to match against)
            value: (batch, seq_len, embed_dim) - Value vectors (what to retrieve)
            mask: (batch, seq_len) - Padding mask (1 for valid tokens, 0 for padding)
            causal_mask: bool - If True, prevent attending to future positions
            position_bias: (num_heads, seq_len, seq_len) - Relative position bias (optional)
            
        Returns:
            output: (batch, seq_len, embed_dim) - Attention output
            attention_weights: (batch, num_heads, seq_len, seq_len) - Attention weights
        """
        batch_size, seq_len, _ = query.size()
        
        # Step 1: Project inputs through linear layers and split into multiple heads
        # Shape transformations: (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 2: Compute attention scores using scaled dot-product
        # Q @ K^T gives similarity between each query and key position
        # Shape: (batch, num_heads, seq_len, seq_len) where [b,h,i,j] = similarity of query_i to key_j
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # Scale by sqrt(d_k)
        
        # Step 2.5: Add relative position bias if provided
        if position_bias is not None:
            # position_bias: (num_heads, seq_len, seq_len)
            scores = scores + position_bias.unsqueeze(0)  # Broadcast over batch
        
        # Step 3: Apply causal mask (prevent looking into the future)
        if causal_mask:
            # Create upper triangular matrix of -inf values
            # This ensures position i cannot attend to positions j > i
            causal_mask_matrix = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device) * float('-inf'),
                diagonal=1  # Start from diagonal+1 (don't mask current position)
            )
            scores = scores + causal_mask_matrix  # Adding -inf before softmax = attention weight of 0
        
        # Step 4: Apply padding mask (ignore padded positions)
        if mask is not None:
            # Reshape mask from (batch, seq_len) to (batch, 1, 1, seq_len) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)
            # Set scores to -inf where mask is 0 (padded positions)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 5: Convert scores to probabilities via softmax
        attn_weights = F.softmax(scores, dim=-1)  # Softmax over keys (last dimension)
        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights
        
        # Step 6: Apply attention weights to values
        # Weighted sum of values based on attention weights
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Step 7: Concatenate all heads and project through output layer
        # Transpose and reshape: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)  # Final linear projection
        
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (FFN).
    
    Applied independently to each position in the sequence.
    Standard architecture: expand to hidden_dim, apply ReLU, project back to embed_dim.
    """
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        """
        Args:
            embed_dim (int): Input/output dimension (e.g., 128)
            hidden_dim (int): Hidden layer dimension (typically 4 * embed_dim = 512)
            dropout (float): Dropout probability for regularization
        """
        super().__init__()
        # Two-layer MLP: embed_dim -> hidden_dim -> embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # Expansion layer
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # Projection layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Apply feed-forward network to each position independently.
        
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            x: (batch, seq_len, embed_dim)
        """
        x = F.relu(self.fc1(x))  # Expand and activate: (batch, seq_len, hidden_dim)
        x = self.dropout(x)       # Regularization
        x = self.fc2(x)           # Project back: (batch, seq_len, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with self-attention and position-wise FFN.
    
    Architecture: 
        x -> [MultiHeadAttention + Residual + LayerNorm] -> [FFN + Residual + LayerNorm] -> output
    
    Uses Pre-LN (Layer Normalization) variant for stable training.
    """
    
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        """
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            hidden_dim (int): FFN hidden dimension
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Sub-layer 1: Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Sub-layer 2: Position-wise feed-forward network
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        
        # Layer normalization after each sub-layer
        self.norm1 = nn.LayerNorm(embed_dim)  # After attention
        self.norm2 = nn.LayerNorm(embed_dim)  # After FFN
        
        # Dropout after each sub-layer (before adding residual)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, position_bias=None):
        """
        Apply transformer block: attention + FFN with residuals.
        
        Args:
            x: (batch, seq_len, embed_dim) - Input sequence
            mask: (batch, seq_len) - Padding mask
            position_bias: (num_heads, seq_len, seq_len) - Relative position bias (optional)
            
        Returns:
            output: (batch, seq_len, embed_dim) - Transformed sequence
        """
        # Sub-layer 1: Self-attention with residual connection
        # x is used as query, key, and value (self-attention)
        attn_out, attn_weights = self.attention(x, x, x, mask=mask, causal_mask=True, position_bias=position_bias)
        x = x + self.dropout1(attn_out)  # Residual: x_new = x_old + sublayer(x_old)
        x = self.norm1(x)                 # Normalize after residual
        
        # Sub-layer 2: Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)  # Residual connection
        x = self.norm2(x)                # Normalize after residual
        
        return x


class SAKT(nn.Module):
    """
    Self-Attentive Knowledge Tracing (SAKT) model.
    
    Architecture:
        1. Embed interactions (question + response + position)
        2. Pass through Transformer encoder blocks
        3. Use question-specific attention for prediction
        4. Output probability of correct response
    
    Predicts student performance on next question given interaction history.
    """
    
    def __init__(
        self,
        num_questions,      # Total number of unique questions in dataset
        embed_dim=128,      # Embedding dimension for all components
        num_heads=8,        # Number of attention heads in each Transformer block
        num_layers=2,       # Number of stacked Transformer blocks
        dropout=0.1,        # Dropout probability for regularization
        max_seq_len=200,    # Maximum sequence length for positional embeddings
        num_skills=None,    # Number of unique skills/topics (optional)
        use_relative_pos=True,  # Use relative position encodings
        use_time_embeddings=False,  # Use continuous time embeddings
        predict_uncertainty=False   # Predict uncertainty (mean + variance)
    ):
        """
        Initialize SAKT model.
        
        Args:
            num_questions (int): Number of unique questions/skills in the dataset
            embed_dim (int): Dimension of embedding vectors
            num_heads (int): Number of attention heads
            num_layers (int): Number of Transformer encoder layers
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length
            num_skills (int, optional): Number of unique skills/topics
            use_relative_pos (bool): Use relative position encodings instead of absolute
            use_time_embeddings (bool): Use continuous time gap embeddings
            predict_uncertainty (bool): Predict both mean and variance for uncertainty
        """
        super().__init__()
        
        self.num_questions = num_questions
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_skills = num_skills
        self.use_relative_pos = use_relative_pos
        self.use_time_embeddings = use_time_embeddings
        self.predict_uncertainty = predict_uncertainty
        
        # === Embedding Layers ===
        
        # Question embeddings: Map question IDs to dense vectors
        # num_questions + 1 to account for padding token at index 0
        self.question_embed = nn.Embedding(num_questions + 1, embed_dim, padding_idx=0)
        
        # Response embeddings: Encode whether answer was correct (1) or incorrect (0)
        # Index 0 = incorrect, Index 1 = correct
        self.response_embed = nn.Embedding(2, embed_dim)
        
        # Skill/topic embeddings (optional): Encode higher-level concepts
        if num_skills is not None:
            self.skill_embed = nn.Embedding(num_skills + 1, embed_dim, padding_idx=0)
        else:
            self.skill_embed = None
        
        # Positional embeddings: Encode position in sequence (like "time step")
        # Allows model to use ordering information
        if not use_relative_pos:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        else:
            # Relative position embeddings (T5-style)
            # Bucketized relative positions for efficiency
            self.num_rel_pos_buckets = 32
            self.max_rel_pos_distance = 128
            self.rel_pos_embed = nn.Embedding(self.num_rel_pos_buckets, num_heads)
        
        # Continuous time embeddings (optional)
        if use_time_embeddings:
            self.time_encoder = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.time_encoder = None
        
        # === Transformer Encoder ===
        
        # Stack of Transformer blocks to encode interaction sequence
        # Hidden dim is typically 4x embed_dim (standard Transformer architecture)
        hidden_dim = embed_dim * 4
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # === Prediction Head ===
        
        # Question-specific attention mechanism
        # Given a target question, attend to relevant past interactions
        self.pred_query = nn.Linear(embed_dim, embed_dim)  # Project target question
        self.pred_key = nn.Linear(embed_dim, embed_dim)    # Project past interactions (keys)
        self.pred_value = nn.Linear(embed_dim, embed_dim)  # Project past interactions (values)
        
        # MLP to produce final prediction from attended context
        if not predict_uncertainty:
            # Standard: embed_dim -> embed_dim/2 -> 1 (binary classification)
            self.pred_fc = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1)  # Output logit (sigmoid applied later)
            )
            self.pred_fc_var = None
        else:
            # Uncertainty estimation: predict both mean and variance
            self.pred_fc = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1)  # Mean prediction
            )
            self.pred_fc_var = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1)  # Log variance prediction
            )
        
        # Dropout for embedding layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize all weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights using Xavier initialization for linear layers
        and normal distribution for embeddings.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform: Good for layers with sigmoid/tanh activation
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # Biases initialized to 0
            elif isinstance(module, nn.Embedding):
                # Embeddings: Small random values
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def _relative_position_bucket(self, relative_position):
        """
        Bucketize relative positions for efficient relative position encoding.
        Based on T5's bucketing strategy.
        
        Args:
            relative_position: (seq_len, seq_len) tensor of relative positions
            
        Returns:
            bucket_ids: (seq_len, seq_len) tensor of bucket indices
        """
        num_buckets = self.num_rel_pos_buckets
        max_distance = self.max_rel_pos_distance
        
        # Half buckets for negative positions, half for positive
        num_buckets //= 2
        buckets = (relative_position > 0).long() * num_buckets
        relative_position = torch.abs(relative_position)
        
        # Small positions get their own bucket (0 to num_buckets//2)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # Large positions are logarithmically bucketed
        val_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        buckets = buckets + torch.where(is_small, relative_position, val_if_large)
        return buckets
    
    def _compute_relative_position_bias(self, seq_len, device):
        """
        Compute relative position bias for attention scores.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensors on
            
        Returns:
            bias: (num_heads, seq_len, seq_len) position bias
        """
        # Create relative position matrix
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions[None, :] - positions[:, None]  # (seq_len, seq_len)
        
        # Bucketize and embed
        buckets = self._relative_position_bucket(relative_positions)
        bias = self.rel_pos_embed(buckets)  # (seq_len, seq_len, num_heads)
        bias = bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)
        
        return bias
    
    def embed_interactions(self, question_ids, responses=None, skill_ids=None, time_deltas=None):
        """
        Convert interaction sequence to embedding vectors.
        
        Combines multiple types of embeddings:
            1. Question embeddings (which question was answered)
            2. Response embeddings (correct/incorrect) - optional
            3. Skill embeddings (topic/concept) - optional
            4. Positional embeddings (position in sequence) or relative positions
            5. Time embeddings (time gaps between interactions) - optional
        
        Args:
            question_ids: (batch, seq_len) - Question IDs [1, num_questions]
            responses: (batch, seq_len) - Binary responses {0, 1} (optional)
            skill_ids: (batch, seq_len) - Skill/topic IDs (optional)
            time_deltas: (batch, seq_len) - Time gaps in seconds (optional)
            
        Returns:
            embeddings: (batch, seq_len, embed_dim) - Combined embeddings
        """
        batch_size, seq_len = question_ids.size()
        
        # Step 1: Embed question IDs
        # Maps each question ID to a learned dense vector
        q_embed = self.question_embed(question_ids)  # (batch, seq_len, embed_dim)
        
        # Step 2: Add response embeddings if provided (encode correctness)
        if responses is not None:
            r_embed = self.response_embed(responses)  # (batch, seq_len, embed_dim)
            embed = q_embed + r_embed  # Element-wise addition
        else:
            embed = q_embed  # Question only (e.g., for target question prediction)
        
        # Step 3: Add skill embeddings if available
        if skill_ids is not None and self.skill_embed is not None:
            s_embed = self.skill_embed(skill_ids)  # (batch, seq_len, embed_dim)
            embed = embed + s_embed
        
        # Step 4: Add positional embeddings (if using absolute positions)
        if not self.use_relative_pos:
            # Create position indices: [0, 1, 2, ..., seq_len-1]
            positions = torch.arange(seq_len, device=question_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_embed = self.pos_embed(positions)  # (batch, seq_len, embed_dim)
            embed = embed + pos_embed
        # Note: Relative positions are added in attention, not here
        
        # Step 5: Add time embeddings if available
        if time_deltas is not None and self.time_encoder is not None:
            # Normalize time deltas (log scale for better range handling)
            time_deltas_norm = torch.log1p(time_deltas.float()).unsqueeze(-1)  # (batch, seq_len, 1)
            time_embed = self.time_encoder(time_deltas_norm)  # (batch, seq_len, embed_dim)
            embed = embed + time_embed
        
        embed = self.dropout(embed)  # Apply dropout for regularization
        
        return embed
    
    def encode(self, question_ids, responses, mask=None, skill_ids=None, time_deltas=None):
        """
        Encode interaction sequence through transformer.
        
        Args:
            question_ids: (batch, seq_len)
            responses: (batch, seq_len)
            mask: (batch, seq_len) - padding mask
            skill_ids: (batch, seq_len) - skill IDs (optional)
            time_deltas: (batch, seq_len) - time gaps (optional)
            
        Returns:
            encoded: (batch, seq_len, embed_dim)
        """
        # Embed interactions
        x = self.embed_interactions(question_ids, responses, skill_ids, time_deltas)
        
        # Compute relative position bias if using relative positions
        position_bias = None
        if self.use_relative_pos:
            _, seq_len, _ = x.size()
            position_bias = self._compute_relative_position_bias(seq_len, x.device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask, position_bias=position_bias)
        
        return x
    
    def forward(self, question_ids, responses, next_question_ids, mask=None, 
                skill_ids=None, time_deltas=None, return_uncertainty=False):
        """
        Forward pass for prediction.
        
        Args:
            question_ids: (batch, seq_len) - past question IDs
            responses: (batch, seq_len) - past responses
            next_question_ids: (batch, seq_len) or (batch, 1) - next question(s) to predict
            mask: (batch, seq_len) - padding mask for past interactions
            skill_ids: (batch, seq_len) - skill IDs (optional)
            time_deltas: (batch, seq_len) - time gaps (optional)
            return_uncertainty: bool - return uncertainty estimates if available
            
        Returns:
            predictions: (batch, seq_len) or (batch, 1) - predicted probabilities
            uncertainty: (batch, seq_len) or (batch, 1) - predicted uncertainty (if return_uncertainty=True)
        """
        # Encode past interactions
        encoded = self.encode(question_ids, responses, mask, skill_ids, time_deltas)
        
        # Get next question embeddings (query)
        next_q_embed = self.question_embed(next_question_ids)
        
        # Question-specific attention
        # Use next question as query, past interactions as key/value
        query = self.pred_query(next_q_embed)  # (batch, 1 or seq_len, embed_dim)
        key = self.pred_key(encoded)            # (batch, seq_len, embed_dim)
        value = self.pred_value(encoded)        # (batch, seq_len, embed_dim)
        
        # Attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, value)  # (batch, 1 or seq_len, embed_dim)
        
        # Final prediction
        logits = self.pred_fc(context).squeeze(-1)  # (batch, 1) or (batch, seq_len)
        predictions = torch.sigmoid(logits)
        
        # Uncertainty estimation if enabled
        if return_uncertainty and self.pred_fc_var is not None:
            log_var = self.pred_fc_var(context).squeeze(-1)
            uncertainty = torch.exp(0.5 * log_var)  # Convert log variance to std
            return predictions, uncertainty
        
        return predictions
    
    def predict_next(self, question_ids, responses, next_question_id):
        """
        Predict single next interaction.
        
        Args:
            question_ids: (batch, seq_len)
            responses: (batch, seq_len)
            next_question_id: (batch,) - single next question
            
        Returns:
            prediction: (batch,) - probability of correct response
        """
        next_q = next_question_id.unsqueeze(1)  # (batch, 1)
        pred = self.forward(question_ids, responses, next_q)  # (batch, 1)
        return pred.squeeze(1)  # (batch,)
    
    def get_parameters_as_vector(self):
        """Get all model parameters as a flat vector (for MAML/Proto-KT)."""
        return torch.cat([p.flatten() for p in self.parameters()])
    
    def set_parameters_from_vector(self, vector):
        """Set model parameters from a flat vector."""
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(vector[offset:offset + numel].view_as(param))
            offset += numel


if __name__ == "__main__":
    # Test SAKT
    batch_size = 4
    seq_len = 10
    num_questions = 100
    
    model = SAKT(num_questions, embed_dim=64, num_heads=4, num_layers=2)
    
    question_ids = torch.randint(1, num_questions, (batch_size, seq_len))
    responses = torch.randint(0, 2, (batch_size, seq_len))
    next_question_ids = torch.randint(1, num_questions, (batch_size, 1))
    
    # Forward pass
    predictions = model(question_ids, responses, next_question_ids)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Input shape: {question_ids.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Predictions: {predictions.squeeze()}")
    
    # Test encoding
    encoded = model.encode(question_ids, responses)
    print(f"Encoded shape: {encoded.shape}")


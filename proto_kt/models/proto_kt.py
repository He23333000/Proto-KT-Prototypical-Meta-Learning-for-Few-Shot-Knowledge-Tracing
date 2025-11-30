"""
Proto-KT: Meta-Learning Student Prototypes for Few-Shot Knowledge Tracing

This module implements the core Proto-KT framework, which learns a set of student prototypes
for conditional model initialization.

Key components:
    1. ContextEncoder: Encodes student's support set into a context vector c_i
    2. PrototypeMemory: Stores k prototype embeddings {p_j} and parameter sets {Θ_j}
    3. Attention mechanism: Computes weights a_ij = softmax(c_i · p_j / sqrt(d))
    4. Parameter generation: θ_i^(0) = Σ_j a_ij * Θ_j (weighted mixture)

Advantages over MAML:
    - MAML: Single initialization θ for all students
    - Proto-KT: Conditional initialization θ_i^(0) based on student's first interactions
    - Result: Better few-shot adaptation by starting from a more appropriate point
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sakt import SAKT
import math
from collections import OrderedDict


class ContextEncoder(nn.Module):
    """
    Context Encoder: Encodes a student's support set into a fixed-size context vector.
    
    The context vector c_i summarizes the student's initial behavior pattern, which is then
    used to query the prototype memory for a personalized initialization.
    
    Architecture:
        Support set -> SAKT encoder -> Mean pooling -> Linear projection -> Context vector c_i
    """
    
    def __init__(self, num_questions, embed_dim=128, context_dim=128, num_heads=4):
        """
        Initialize context encoder.
        
        Args:
            num_questions (int): Number of questions in dataset
            embed_dim (int): Embedding dimension for SAKT encoder
            context_dim (int): Output dimension for context vector
            num_heads (int): Number of attention heads in encoder
        """
        super().__init__()
        
        # Lightweight SAKT encoder to process support set
        # Uses only 1 layer for efficiency (support sets are small, e.g., 5 interactions)
        self.encoder = SAKT(
            num_questions=num_questions,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=1,  # Single layer - shallow encoder for speed
            dropout=0.1
        )
        
        # Linear projection to context dimension
        # Allows context_dim to differ from embed_dim if needed
        self.context_proj = nn.Linear(embed_dim, context_dim)
        
    def forward(self, question_ids, responses, mask=None):
        """
        Encode support set interactions into a context vector.
        
        Process:
            1. Pass support set through SAKT encoder
            2. Pool sequence representations (mean pooling)
            3. Project to context dimension
        
        Args:
            question_ids: (batch, support_size) - Question IDs in support set
            responses: (batch, support_size) - Responses in support set
            mask: (batch, support_size) - Mask for variable-length sequences
            
        Returns:
            context: (batch, context_dim) - Context vector for each student
        """
        # Step 1: Encode support set through SAKT
        # encoded captures the sequential pattern of the student's initial interactions
        encoded = self.encoder.encode(question_ids, responses, mask)  # (batch, support_size, embed_dim)
        
        # Step 2: Aggregate sequence into single vector via mean pooling
        if mask is not None:
            # Masked mean: only average over valid (non-padded) positions
            mask_expanded = mask.unsqueeze(-1)  # (batch, support_size, 1) for broadcasting
            encoded_masked = encoded * mask_expanded  # Zero out padded positions
            # Sum and divide by number of valid positions
            context = encoded_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Simple mean across all positions
            context = encoded.mean(dim=1)  # (batch, embed_dim)
        
        # Step 3: Project to context dimension
        context = self.context_proj(context)  # (batch, context_dim)
        
        return context


class ParameterAlignmentLayer(nn.Module):
    """
    Parameter Alignment Layer: Normalizes parameters before weighted combination.
    
    Problem this solves:
        Naively mixing neural network parameters (θ = Σ a_j * Θ_j) can be problematic because:
        - Different layers have different scales (e.g., embeddings vs. final layer)
        - Direct interpolation may produce parameters in "low-quality" regions
        - No guarantee that linear combination maintains network properties
    
    Solution:
        - Normalize all parameter sets {Θ_j} to zero mean, unit variance
        - Compute weighted combination in normalized space
        - Denormalize back to original scale
        
    This ensures parameters are combined in a "canonical" space where
    linear interpolation is more meaningful.
    """
    
    def __init__(self, parameter_shapes):
        """
        Args:
            parameter_shapes (OrderedDict): Dictionary mapping parameter names to shapes
        """
        super().__init__()
        self.parameter_shapes = parameter_shapes
        
        # Register buffers for running statistics (mean and std of parameter bank)
        # Buffers are not trainable but are part of model state
        total_params = sum(shape.numel() for shape in parameter_shapes.values())
        self.register_buffer('param_mean', torch.zeros(total_params))  # Mean of each parameter across prototypes
        self.register_buffer('param_std', torch.ones(total_params))    # Std of each parameter across prototypes
        self.register_buffer('initialized', torch.tensor(0))           # Flag: has statistics been initialized?
        
    def initialize_statistics(self, parameter_bank):
        """
        Initialize normalization statistics from parameter bank.
        
        Called once after model initialization to compute mean/std across all prototypes.
        
        Args:
            parameter_bank: (num_prototypes, total_params) - All prototype parameters
        """
        num_prototypes = parameter_bank.shape[0]
        
        if num_prototypes == 1:
            # With only 1 prototype, std is undefined. Use default values.
            # Mean = the single prototype's values, std = 1.0 (no normalization effect)
            self.param_mean.copy_(parameter_bank[0])
            self.param_std.fill_(1.0)
        else:
            # Compute mean and std across prototypes (dim=0)
            # Use unbiased=False to avoid issues with small num_prototypes
            self.param_mean.copy_(parameter_bank.mean(dim=0))
            self.param_std.copy_(parameter_bank.std(dim=0, unbiased=False).clamp(min=1e-6))
        
        self.initialized.fill_(1)  # Mark as initialized
    
    def normalize(self, params):
        """
        Normalize parameters to zero mean, unit variance.
        
        Args:
            params: (..., total_params) - Parameter vector(s)
        Returns:
            normalized_params: (..., total_params) - Normalized parameters
        """
        if self.initialized == 0:
            return params  # Skip if not initialized
        return (params - self.param_mean) / self.param_std
    
    def denormalize(self, params):
        """
        Denormalize parameters back to original scale.
        
        Args:
            params: (..., total_params) - Normalized parameter vector(s)
        Returns:
            denormalized_params: (..., total_params) - Original scale parameters
        """
        if self.initialized == 0:
            return params  # Skip if not initialized
        return params * self.param_std + self.param_mean


class ProtoKT(nn.Module):
    """
    Proto-KT: Prototypical Meta-Learning for Knowledge Tracing
    
    Main components:
        1. Context Encoder: Encodes support set → context vector c_i
        2. Prototype Memory: Stores k prototype embeddings P = {p_1, ..., p_k}
        3. Parameter Bank: Stores k parameter sets Θ = {Θ_1, ..., Θ_k}
        4. Attention: Computes similarity a_i = softmax(c_i · P^T / sqrt(d))
        5. Parameter Generator: θ_i^(0) = Σ_j a_ij * Θ_j
    
    Algorithm (from paper):
        For each student i:
            1. Encode their support set: c_i = ContextEncoder(support_set_i)
            2. Compute prototype attention: a_i = softmax(c_i · P^T / sqrt(d))
            3. Generate personalized init: θ_i^(0) = Σ_j a_ij * Θ_j
            4. Adapt: θ_i' = θ_i^(0) - α * ∇_θ L(θ_i^(0); support_set_i)
            5. Evaluate: L(θ_i'; query_set_i)
    
    Meta-learning objective:
        min_Φ Σ_i L(θ_i'; query_set_i)  where Φ = {P, Θ}
    """
    
    def __init__(
        self,
        num_questions,      # Number of questions in dataset
        num_prototypes=8,   # Number of student prototypes (k)
        embed_dim=128,      # SAKT embedding dimension
        context_dim=128,    # Context vector dimension
        num_heads=8,        # Number of attention heads in SAKT
        num_layers=2,       # Number of Transformer layers in SAKT
        dropout=0.1,        # Dropout rate
        use_alignment=True  # Whether to use parameter alignment layer
    ):
        """
        Initialize Proto-KT framework.
        
        Args:
            num_questions (int): Number of unique questions
            num_prototypes (int): Number of prototypes k (default: 8)
            embed_dim (int): Embedding dimension for SAKT
            context_dim (int): Dimension of context vectors
            num_heads (int): Number of attention heads
            num_layers (int): Number of Transformer layers
            dropout (float): Dropout rate
            use_alignment (bool): Use parameter alignment layer
        """
        super().__init__()
        
        self.num_prototypes = num_prototypes
        self.context_dim = context_dim
        self.use_alignment = use_alignment
        
        # === Component 1: Context Encoder ===
        # Encodes support set into context vector c_i
        # Shared across all students (not part of meta-parameters)
        self.context_encoder = ContextEncoder(
            num_questions=num_questions,
            embed_dim=embed_dim // 2,  # Smaller for computational efficiency
            context_dim=context_dim,
            num_heads=4
        )
        
        # === Component 2: Prototype Memory ===
        # Learnable prototype embeddings: P = {p_1, ..., p_k}
        # Each p_j ∈ R^{context_dim} represents a student "archetype"
        self.prototype_embeddings = nn.Parameter(
            torch.randn(num_prototypes, context_dim)  # Shape: (k, context_dim)
        )
        
        # === Component 3: Parameter Bank ===
        # Create template SAKT model to get parameter structure
        template_model = SAKT(
            num_questions=num_questions,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Store parameter shapes for reconstruction
        self.param_shapes = OrderedDict()
        self.param_names = []
        for name, param in template_model.named_parameters():
            self.param_shapes[name] = param.shape
            self.param_names.append(name)
        
        # Calculate total number of parameters in SAKT
        total_params = sum(p.numel() for p in template_model.parameters())
        print(f"SAKT has {total_params:,} parameters")
        
        # Create parameter bank: Θ = {Θ_1, ..., Θ_k}
        # Each Θ_j is a complete set of SAKT parameters (flattened)
        # Shape: (num_prototypes, total_params)
        self.parameter_bank = nn.Parameter(
            torch.randn(num_prototypes, total_params) * 0.01
        )
        
        # Store SAKT configuration for creating task-specific models
        # MUST be set before _initialize_parameter_bank
        self.template_config = {
            'num_questions': num_questions,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        # === Component 4: Parameter Alignment (Optional) ===
        # Normalizes parameters before mixing to handle scale differences
        # MUST be defined before _initialize_parameter_bank
        if use_alignment:
            self.alignment_layer = ParameterAlignmentLayer(self.param_shapes)
        else:
            self.alignment_layer = None
        
        # Initialize parameter bank with k different random SAKT initializations
        self._initialize_parameter_bank(template_model)
        
        # Scale for attention
        self.attention_scale = math.sqrt(context_dim)
        
    def _initialize_parameter_bank(self, template_model):
        """Initialize each prototype's parameters with different random seeds."""
        for k in range(self.num_prototypes):
            # Create a new model with different initialization
            model = SAKT(**self.template_config)
            
            # Flatten parameters
            params = torch.cat([p.flatten() for p in model.parameters()])
            
            # Store in bank
            self.parameter_bank.data[k].copy_(params)
        
        # Initialize alignment layer if used
        if self.alignment_layer is not None:
            self.alignment_layer.initialize_statistics(self.parameter_bank.data)
    
    def compute_prototype_attention(self, context):
        """
        Compute attention weights over prototypes.
        
        Args:
            context: (batch, context_dim) - encoded support set
            
        Returns:
            attention_weights: (batch, num_prototypes) - attention distribution
        """
        # Scaled dot-product attention: softmax(c * P^T / sqrt(d))
        scores = torch.matmul(context, self.prototype_embeddings.T) / self.attention_scale
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights
    
    def generate_parameters(self, attention_weights):
        """
        Generate personalized parameters as weighted combination of prototype parameters.
        
        Args:
            attention_weights: (batch, num_prototypes)
            
        Returns:
            theta_init: (batch, total_params) - initial parameters for each student
        """
        # Weighted combination: theta_i = sum_j a_ij * Theta_j
        # (batch, num_prototypes) @ (num_prototypes, total_params) -> (batch, total_params)
        
        if self.use_alignment:
            # Normalize parameters before mixing
            normalized_bank = self.alignment_layer.normalize(self.parameter_bank)
            theta_init = torch.matmul(attention_weights, normalized_bank)
            # Denormalize back
            theta_init = self.alignment_layer.denormalize(theta_init)
        else:
            theta_init = torch.matmul(attention_weights, self.parameter_bank)
        
        return theta_init
    
    def create_task_model(self, theta_init):
        """
        Create a SAKT model with given parameters.
        
        Args:
            theta_init: (total_params,) - flat parameter vector
            
        Returns:
            model: SAKT model with initialized parameters
        """
        model = SAKT(**self.template_config)
        
        # Set parameters from vector
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            param.data.copy_(theta_init[offset:offset + numel].view(param.shape))
            offset += numel
        
        return model
    
    def forward(self, support_q, support_r, support_mask=None):
        """
        Generate initial parameters for a batch of students.
        
        Args:
            support_q: (batch, support_size) - support question IDs
            support_r: (batch, support_size) - support responses
            support_mask: (batch, support_size) - support mask
            
        Returns:
            dict with:
                - theta_init: (batch, total_params) - initial parameters
                - attention_weights: (batch, num_prototypes) - attention distribution
                - context: (batch, context_dim) - context vectors
        """
        # 1. Encode support set to context
        context = self.context_encoder(support_q, support_r, support_mask)
        
        # 2. Compute prototype attention
        attention_weights = self.compute_prototype_attention(context)
        
        # 3. Generate initial parameters
        theta_init = self.generate_parameters(attention_weights)
        
        return {
            'theta_init': theta_init,
            'attention_weights': attention_weights,
            'context': context
        }
    
    def get_prototype_parameters(self, prototype_idx):
        """Get parameters for a specific prototype."""
        return self.parameter_bank[prototype_idx]
    
    def get_all_prototype_models(self):
        """Get SAKT models for all prototypes (for analysis)."""
        models = []
        for k in range(self.num_prototypes):
            theta_k = self.parameter_bank[k]
            model_k = self.create_task_model(theta_k)
            models.append(model_k)
        return models


if __name__ == "__main__":
    # Test Proto-KT
    num_questions = 100
    batch_size = 4
    support_size = 5
    num_prototypes = 8
    
    proto_kt = ProtoKT(
        num_questions=num_questions,
        num_prototypes=num_prototypes,
        embed_dim=64,
        context_dim=64,
        num_heads=4,
        num_layers=2
    )
    
    # Create dummy support data
    support_q = torch.randint(1, num_questions, (batch_size, support_size))
    support_r = torch.randint(0, 2, (batch_size, support_size))
    
    # Forward pass
    output = proto_kt(support_q, support_r)
    
    print(f"Proto-KT prototypes: {num_prototypes}")
    print(f"Context shape: {output['context'].shape}")
    print(f"Attention weights shape: {output['attention_weights'].shape}")
    print(f"Theta init shape: {output['theta_init'].shape}")
    print(f"Attention weights (student 0): {output['attention_weights'][0]}")
    
    # Create a task-specific model
    theta_0 = output['theta_init'][0]
    task_model = proto_kt.create_task_model(theta_0)
    print(f"\nCreated task model with {sum(p.numel() for p in task_model.parameters())} parameters")
    
    # Test prediction
    query_q = torch.randint(1, num_questions, (1, 10))
    query_r = torch.randint(0, 2, (1, 10))
    next_q = torch.randint(1, num_questions, (1, 1))
    
    pred = task_model(query_q, query_r, next_q)
    print(f"Prediction: {pred.item():.4f}")


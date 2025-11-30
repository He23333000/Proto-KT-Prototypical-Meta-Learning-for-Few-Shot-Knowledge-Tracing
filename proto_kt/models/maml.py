"""
MAML-SAKT: Model-Agnostic Meta-Learning applied to SAKT.

This implements the MAML algorithm (Finn et al., 2017) for Knowledge Tracing.
MAML learns a single universal initialization θ that can be quickly adapted
to new students via a few gradient steps.

Key difference from Proto-KT:
    - MAML: ONE initialization for ALL students (θ_0 is the same for everyone)
    - Proto-KT: CONDITIONAL initialization (θ_0 differs based on student's first interactions)

This serves as the primary baseline for Proto-KT.
Functionally equivalent to Proto-KT with k=1 (single prototype).
"""
import torch
import torch.nn as nn
from .sakt import SAKT
from collections import OrderedDict


class MAML_SAKT(nn.Module):
    """
    MAML (Model-Agnostic Meta-Learning) applied to SAKT for Knowledge Tracing.
    
    Core idea:
        - Meta-train: Learn θ that works well after adaptation across many students
        - Adaptation: For a new student, adapt θ → θ' via gradient descent on their data
        - Prediction: Use adapted θ' to predict future performance
    
    The meta-learned parameters θ serve as a "good starting point" that enables
    fast adaptation with limited data (few-shot learning).
    """
    
    def __init__(
        self,
        num_questions,  # Total number of questions in dataset
        embed_dim=128,  # Embedding dimension for SAKT
        num_heads=8,    # Number of attention heads
        num_layers=2,   # Number of Transformer layers
        dropout=0.1     # Dropout rate
    ):
        """
        Initialize MAML-SAKT wrapper around SAKT model.
        
        Args:
            num_questions (int): Number of unique questions
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of Transformer layers
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # The meta-learned SAKT model
        # These parameters (θ) will be meta-learned via bi-level optimization
        self.sakt = SAKT(
            num_questions=num_questions,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Store configuration for creating task-specific model copies
        # Used when we need to instantiate a fresh SAKT with specific parameters
        self.template_config = {
            'num_questions': num_questions,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        # Store parameter shapes and names for reconstruction
        # Needed to convert between flat vector and structured parameters
        self.param_shapes = OrderedDict()
        self.param_names = []
        for name, param in self.sakt.named_parameters():
            self.param_shapes[name] = param.shape
            self.param_names.append(name)
        
        # Print model size
        total_params = sum(p.numel() for p in self.sakt.parameters())
        print(f"MAML-SAKT has {total_params:,} parameters")
    
    def get_initial_parameters(self, batch_size=1):
        """
        Get initial parameters θ_0 for adaptation.
        
        For MAML, θ_0 is the SAME for ALL students (universal initialization).
        This is the key difference from Proto-KT, which generates different θ_0
        for different students based on their support set.
        
        Args:
            batch_size (int): Number of students/tasks in the batch
            
        Returns:
            theta_init: (batch_size, total_params) - Initial parameter vectors
                        Note: All rows are identical in MAML
        """
        # Step 1: Flatten all current model parameters into a single vector
        # This converts structured parameters (layers, weights, biases) into one flat vector
        params = torch.cat([p.flatten() for p in self.sakt.parameters()])  # Shape: (total_params,) this is theta 0
        
        # Step 2: Replicate for entire batch
        # All students get the exact same initialization (MAML's limitation)
        theta_init = params.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, total_params)
        
        return theta_init
    
    def create_task_model(self, theta_init):
        """
        Create a SAKT model instance with specific parameter values.
        
        Used during adaptation: given adapted parameters θ', create a model with those parameters.
        
        Args:
            theta_init: (total_params,) - Flat parameter vector
            
        Returns:
            model: SAKT model with parameters set to theta_init
        """
        # Create a fresh SAKT model with the same architecture
        model = SAKT(**self.template_config)
        
        # Unflatten parameter vector and assign to model
        # Reverse of get_initial_parameters: vector -> structured parameters
        offset = 0
        for param in model.parameters():
            numel = param.numel()  # Number of elements in this parameter tensor
            # Extract the relevant slice and reshape to match parameter shape
            param.data.copy_(theta_init[offset:offset + numel].view(param.shape))
            offset += numel
        
        return model
    
    def forward(self, support_q, support_r, support_mask=None):
        """
        Generate initial parameters for a batch of students.
        
        For MAML: Returns the SAME initialization for ALL students (unconditional).
        For Proto-KT: Returns DIFFERENT initializations based on support set (conditional).
        
        This method provides a consistent interface between MAML and Proto-KT.
        
        Args:
            support_q: (batch, support_size) - Support set question IDs
            support_r: (batch, support_size) - Support set responses
            support_mask: (batch, support_size) - Support set mask (unused in MAML)
            
        Returns:
            dict containing:
                - theta_init: (batch, total_params) - Initial parameters for each student
                - attention_weights: None (MAML has no attention mechanism)
                - context: None (MAML does not encode context)
        """
        batch_size = support_q.size(0)
        
        # Get initial parameters (same for all tasks in MAML)
        # Note: support_q and support_r are ignored by MAML (not used for initialization)
        theta_init = self.get_initial_parameters(batch_size)
        
        # Return in same format as Proto-KT for compatibility
        return {
            'theta_init': theta_init,           # Initial parameters
            'attention_weights': None,          # No prototypes in MAML
            'context': None                     # No context encoding in MAML
        }
    
    def parameters(self):
        """Return parameters to be meta-learned."""
        return self.sakt.parameters()
    
    def state_dict(self):
        """Return state dict."""
        return self.sakt.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.sakt.load_state_dict(state_dict)


if __name__ == "__main__":
    # Test MAML-SAKT
    num_questions = 100
    batch_size = 4
    support_size = 5
    
    maml = MAML_SAKT(
        num_questions=num_questions,
        embed_dim=64,
        num_heads=4,
        num_layers=2
    )
    
    # Create dummy support data
    support_q = torch.randint(1, num_questions, (batch_size, support_size))
    support_r = torch.randint(0, 2, (batch_size, support_size))
    
    # Forward pass (get initializations)
    output = maml(support_q, support_r)
    
    print(f"MAML-SAKT initialized")
    print(f"Theta init shape: {output['theta_init'].shape}")
    
    # Create a task-specific model
    theta_0 = output['theta_init'][0]
    task_model = maml.create_task_model(theta_0)
    print(f"\nCreated task model with {sum(p.numel() for p in task_model.parameters())} parameters")
    
    # Test prediction
    query_q = torch.randint(1, num_questions, (1, 10))
    query_r = torch.randint(0, 2, (1, 10))
    next_q = torch.randint(1, num_questions, (1, 1))
    
    pred = task_model(query_q, query_r, next_q)
    print(f"Prediction: {pred.item():.4f}")


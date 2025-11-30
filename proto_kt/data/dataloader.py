"""
Episodic DataLoader for meta-learning in Knowledge Tracing.

This module implements episodic sampling for meta-learning, where each "episode"
is a student (task) split into:
    - Support set: First K interactions (for adaptation)
    - Query set: Remaining interactions (for evaluation)

Meta-learning paradigm:
    - Inner loop: Adapt model on support set (few-shot adaptation)
    - Outer loop: Evaluate adapted model on query set (test performance)
    
Batch structure:
    A batch contains B students (tasks), each with their own support/query split.
    The meta-learner processes all B tasks in parallel, computing the meta-gradient
    by backpropagating through the inner-loop adaptation steps.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import pickle


class KTMetaDataset(Dataset):
    """
    Meta-learning dataset for Knowledge Tracing.
    
    Each data point is a student (task), which gets split into support/query sets.
    This follows the standard meta-learning / few-shot learning protocol.
    
    Terminology mapping:
        - Task = Student
        - Support set = First K interactions (for adaptation)
        - Query set = Remaining interactions (for evaluation)
        - Meta-batch = Batch of students
    """
    
    def __init__(self, sequences: List[Dict], support_size: int = 5):
        """
        Initialize meta-learning dataset.
        
        Args:
            sequences (List[Dict]): List of student sequences from preprocessing.
                                   Each dict contains 'question_ids', 'responses', 'user_id'
            support_size (int): Number of interactions to use as support set (default: 5)
                               The first support_size interactions are used for adaptation,
                               remaining interactions are used for evaluation
        """
        self.sequences = sequences
        self.support_size = support_size
        
    def __len__(self):
        """Return number of students (tasks) in dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single student (task) split into support and query sets.
        
        Args:
            idx (int): Student index
            
        Returns:
            dict containing:
                - support: Dict with support set data
                    * question_ids: (support_size,) - Questions in support set
                    * responses: (support_size,) - Responses in support set
                    * length: support_size
                - query: Dict with query set data
                    * question_ids: (query_len,) - Questions in query set
                    * responses: (query_len,) - Responses in query set
                    * length: query_len
                - user_id: Student identifier
        """
        seq = self.sequences[idx]
        
        # Convert to tensors
        question_ids = torch.tensor(seq['question_ids'], dtype=torch.long)
        responses = torch.tensor(seq['responses'], dtype=torch.long)
        total_len = len(question_ids)
        
        # Split chronologically: first K interactions → support, rest → query
        # This simulates the cold-start scenario: we see first K interactions,
        # then must predict future performance
        support = {
            'question_ids': question_ids[:self.support_size],
            'responses': responses[:self.support_size],
            'length': self.support_size
        }
        
        query = {
            'question_ids': question_ids[self.support_size:],
            'responses': responses[self.support_size:],
            'length': total_len - self.support_size
        }
        
        return {
            'support': support,
            'query': query,
            'user_id': seq['user_id']
        }


def collate_meta_batch(batch: List[Dict]) -> Dict:
    """
    Collate a batch of student tasks.
    
    Args:
        batch: List of dicts from KTMetaDataset
        
    Returns:
        dict with:
            - support: dict of batched support data
            - query: dict of batched query data (variable lengths)
            - user_ids: list of user IDs
    """
    support_questions = []
    support_responses = []
    
    query_questions = []
    query_responses = []
    query_lengths = []
    
    user_ids = []
    
    for item in batch:
        support_questions.append(item['support']['question_ids'])
        support_responses.append(item['support']['responses'])
        
        query_questions.append(item['query']['question_ids'])
        query_responses.append(item['query']['responses'])
        query_lengths.append(item['query']['length'])
        
        user_ids.append(item['user_id'])
    
    # Stack support (all same length)
    support_batch = {
        'question_ids': torch.stack(support_questions),  # (batch, support_size)
        'responses': torch.stack(support_responses),      # (batch, support_size)
    }
    
    # Pad query sequences to max length in batch
    max_query_len = max(query_lengths)
    padded_questions = []
    padded_responses = []
    query_masks = []
    
    for q_ids, resps, length in zip(query_questions, query_responses, query_lengths):
        # Pad to max length
        pad_len = max_query_len - length
        
        padded_q = torch.cat([q_ids, torch.zeros(pad_len, dtype=torch.long)])
        padded_r = torch.cat([resps, torch.zeros(pad_len, dtype=torch.long)])
        
        # Create mask (1 for real tokens, 0 for padding)
        mask = torch.cat([torch.ones(length), torch.zeros(pad_len)])
        
        padded_questions.append(padded_q)
        padded_responses.append(padded_r)
        query_masks.append(mask)
    
    query_batch = {
        'question_ids': torch.stack(padded_questions),  # (batch, max_query_len)
        'responses': torch.stack(padded_responses),      # (batch, max_query_len)
        'mask': torch.stack(query_masks),                # (batch, max_query_len)
        'lengths': torch.tensor(query_lengths)
    }
    
    return {
        'support': support_batch,
        'query': query_batch,
        'user_ids': user_ids
    }


class SequentialEvaluationDataset(Dataset):
    """
    Dataset for sequential evaluation of new students.
    Returns incrementally growing sequences for testing adaptation over time.
    """
    
    def __init__(self, sequences: List[Dict], support_size: int = 5, max_eval_len: int = 51):
        self.sequences = sequences
        self.support_size = support_size
        self.max_eval_len = max_eval_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns a student's full sequence for sequential evaluation.
        
        Returns:
            dict with:
                - question_ids: (seq_len,) full sequence
                - responses: (seq_len,) full sequence
                - support_end: index where support ends
                - user_id: student ID
        """
        seq = self.sequences[idx]
        
        # Take up to max_eval_len interactions
        eval_len = min(len(seq['question_ids']), self.max_eval_len)
        
        return {
            'question_ids': torch.tensor(seq['question_ids'][:eval_len], dtype=torch.long),
            'responses': torch.tensor(seq['responses'][:eval_len], dtype=torch.long),
            'support_end': self.support_size,
            'user_id': seq['user_id'],
            'length': eval_len
        }


def create_meta_dataloaders(
    data_path: str,
    support_size: int = 5,
    meta_batch_size: int = 16,
    num_workers: int = 0,
    train_fraction: float = 1.0,
    val_fraction: float = 1.0,
    test_fraction: float = 1.0,
    subset_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create meta-learning dataloaders from processed data.
    
    Args:
        data_path: Path to processed pickle file
        support_size: Number of interactions for support set
        meta_batch_size: Number of students (tasks) per meta-batch
        num_workers: Number of dataloader workers
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load processed data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loading data from {data_path}")
    print(f"  Train: {len(data['train'])} students")
    print(f"  Val: {len(data['val'])} students")
    print(f"  Test: {len(data['test'])} students")
    print(f"  Vocabulary: {data['num_questions']} questions")
    
    rng = np.random.default_rng(subset_seed)
    
    def _subset_sequences(sequences, fraction, split_name):
        if fraction >= 0.999 or len(sequences) <= 1:
            return sequences
        subset_size = max(1, int(len(sequences) * fraction))
        subset_size = min(subset_size, len(sequences))
        if subset_size == len(sequences):
            return sequences
        indices = rng.choice(len(sequences), subset_size, replace=False)
        indices.sort()
        print(f"  Using subset for {split_name}: {subset_size}/{len(sequences)} students ({fraction*100:.1f}%)")
        return [sequences[i] for i in indices]
    
    data['train'] = _subset_sequences(data['train'], train_fraction, 'train')
    data['val'] = _subset_sequences(data['val'], val_fraction, 'val')
    data['test'] = _subset_sequences(data['test'], test_fraction, 'test')
    
    # Create datasets
    train_dataset = KTMetaDataset(data['train'], support_size)
    val_dataset = KTMetaDataset(data['val'], support_size)
    test_dataset = SequentialEvaluationDataset(data['test'], support_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=meta_batch_size,
        shuffle=True,
        collate_fn=collate_meta_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=meta_batch_size,
        shuffle=False,
        collate_fn=collate_meta_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test loader: batch size 1 for sequential evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, data['num_questions']


if __name__ == "__main__":
    # Test dataloader
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataloader.py <processed_data.pkl>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    train_loader, val_loader, test_loader, num_q = create_meta_dataloaders(
        data_path,
        support_size=5,
        meta_batch_size=4
    )
    
    print("\nTesting dataloader...")
    batch = next(iter(train_loader))
    
    print(f"Support batch shape: {batch['support']['question_ids'].shape}")
    print(f"Query batch shape: {batch['query']['question_ids'].shape}")
    print(f"Query mask shape: {batch['query']['mask'].shape}")
    print(f"Number of students in batch: {len(batch['user_ids'])}")
    
    print("\nTest evaluation loader:")
    test_item = next(iter(test_loader))
    print(f"Test sequence shape: {test_item['question_ids'].shape}")
    print(f"Support ends at: {test_item['support_end']}")


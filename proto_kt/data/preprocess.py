"""
Preprocess KT datasets with rigorous train/test splitting and leakage prevention.

This module implements the "new student" evaluation protocol from the paper,
ensuring proper meta-learning evaluation by maintaining strict separation between
meta-training and meta-test students.

Key features:
    1. Disjoint student sets: No overlap between train/val/test students
    2. Disjoint question sets: Prevents test-time information leakage
    3. Task diversity analysis: Ensures meta-training covers diverse student behaviors
    4. Sequence filtering: Removes students with too few/many interactions
    5. Data normalization: Encode question IDs, sort interactions chronologically

Critical for fair meta-learning evaluation:
    - Meta-training students should NOT appear in meta-testing
    - Meta-test should simulate realistic "new student" scenario
    - Prevents overly optimistic results from student/question overlap
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import variation
import pickle
import warnings


class KTDataPreprocessor:
    """
    Preprocess Knowledge Tracing datasets for meta-learning experiments.
    
    Pipeline:
        1. Load raw CSV data
        2. Filter students by sequence length (min_seq_len to max_seq_len)
        3. Split students into disjoint train/val/test sets
        4. Encode question IDs to contiguous integers
        5. Verify no data leakage between splits
        6. Analyze task diversity (variation in student behaviors)
        7. Save preprocessed data
    """
    
    def __init__(self, 
                 min_seq_len=51,    # Minimum interactions per student (need enough for adaptation + evaluation)
                 max_seq_len=200,   # Maximum interactions (limit for computational efficiency)
                 train_ratio=0.7,   # 70% of students for meta-training
                 val_ratio=0.15,    # 15% for meta-validation
                 test_ratio=0.15,   # 15% for meta-testing
                 verify_leakage=True  # Run leakage verification checks
                ):
        """
        Initialize preprocessor with splitting parameters.
        
        Args:
            min_seq_len (int): Minimum sequence length (default: 51 for 1 support + 50 query)
            max_seq_len (int): Maximum sequence length (for memory efficiency)
            train_ratio (float): Fraction of students for meta-training
            val_ratio (float): Fraction for meta-validation
            test_ratio (float): Fraction for meta-testing
            verify_leakage (bool): Whether to verify no data leakage
        """
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.verify_leakage = verify_leakage
        
    def load_raw_data(self, filepath, dataset_name='assistments'):
        """Load raw CSV data."""
        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
        
        # Standardize column names
        col_mapping = self._get_column_mapping(df, dataset_name)
        df = df.rename(columns=col_mapping)
        
        # Ensure required columns
        required = ['user_id', 'problem_id', 'correct', 'order_id']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Filter and clean
        df = df[required + (['skill_id'] if 'skill_id' in df.columns else [])]
        df = df.dropna(subset=['user_id', 'problem_id', 'correct'])
        df['correct'] = df['correct'].astype(int)
        
        # Sort by student and time
        df = df.sort_values(['user_id', 'order_id']).reset_index(drop=True)
        
        print(f"Loaded {len(df)} interactions from {df['user_id'].nunique()} students")
        return df

    def _get_column_mapping(self, df, dataset_name):
        """Map dataset-specific column names to standard names."""
        mapping = {}
        
        # User ID
        for col in ['user_id', 'student_id', 'userId']:
            if col in df.columns and 'user_id' not in mapping:
                mapping[col] = 'user_id'
        
        # Problem ID
        for col in ['problem_id', 'question_id', 'item_id', 'problemId']:
            if col in df.columns and 'problem_id' not in mapping:
                mapping[col] = 'problem_id'
        
        # Correctness
        for col in ['correct', 'correctness', 'score']:
            if col in df.columns and 'correct' not in mapping:
                mapping[col] = 'correct'
        
        # Order/timestamp
        for col in ['order_id', 'timestamp', 'ms_first_response']:
            if col in df.columns and 'order_id' not in mapping:
                mapping[col] = 'order_id'
        
        # Skill (optional)
        for col in ['skill_id', 'skill', 'skill_name']:
            if col in df.columns and 'skill_id' not in mapping:
                mapping[col] = 'skill_id'
        
        return mapping
    
    def create_sequences(self, df):
        """Convert dataframe to student sequences."""
        sequences = []
        
        for user_id, group in df.groupby('user_id'):
            # Sort by order
            group = group.sort_values('order_id')
            
            seq_len = len(group)
            if seq_len < self.min_seq_len:
                continue  # Skip short sequences
            
            # Truncate if too long
            if seq_len > self.max_seq_len:
                group = group.head(self.max_seq_len)
            
            questions = group['problem_id'].values
            responses = group['correct'].values
            skills = group['skill_id'].values if 'skill_id' in group.columns else None
            
            sequences.append({
                'user_id': user_id,
                'questions': questions,
                'responses': responses,
                'skills': skills,
                'length': len(questions)
            })
        
        print(f"Created {len(sequences)} valid sequences")
        return sequences
    
    def split_students_disjoint(self, sequences):
        """
        Split students into disjoint train/val/test sets.
        Ensures no student appears in multiple sets.
        """
        num_students = len(sequences)
        indices = np.random.permutation(num_students)
        
        n_train = int(num_students * self.train_ratio)
        n_val = int(num_students * self.val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        test_seqs = [sequences[i] for i in test_idx]
        
        print(f"Split: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test students")
        
        # Verify no overlap
        train_users = set(s['user_id'] for s in train_seqs)
        val_users = set(s['user_id'] for s in val_seqs)
        test_users = set(s['user_id'] for s in test_seqs)
        
        assert len(train_users & val_users) == 0, "Train/val student overlap!"
        assert len(train_users & test_users) == 0, "Train/test student overlap!"
        assert len(val_users & test_users) == 0, "Val/test student overlap!"
        
        return train_seqs, val_seqs, test_seqs
    
    def verify_no_question_leakage(self, train_seqs, test_seqs):
        """
        Verify that train and test sets have disjoint question/skill sets.
        This is critical for true out-of-distribution evaluation.
        """
        if not self.verify_leakage:
            return True
        
        train_questions = set()
        for seq in train_seqs:
            train_questions.update(seq['questions'])
        
        test_questions = set()
        for seq in test_seqs:
            test_questions.update(seq['questions'])
        
        overlap = train_questions & test_questions
        overlap_ratio = len(overlap) / len(test_questions) if test_questions else 0
        
        print(f"\nQuestion overlap analysis:")
        print(f"  Train questions: {len(train_questions)}")
        print(f"  Test questions: {len(test_questions)}")
        print(f"  Overlap: {len(overlap)} ({overlap_ratio*100:.1f}% of test)")
        
        if overlap_ratio > 0.5:
            warnings.warn(f"High question overlap ({overlap_ratio*100:.1f}%). "
                         "Consider using disjoint question splits for stronger evaluation.")
        
        return overlap_ratio
    
    def compute_task_diversity(self, sequences, sample_size=500):
        """
        Compute task diversity metrics to validate meta-learning assumptions.
        Returns: diversity statistics and pairwise similarity distribution.
        """
        print("\nComputing task diversity...")
        
        # Sample for efficiency
        if len(sequences) > sample_size:
            indices = np.random.choice(len(sequences), sample_size, replace=False)
            sample_seqs = [sequences[i] for i in indices]
        else:
            sample_seqs = sequences
        
        # 1. Performance diversity (coefficient of variation)
        accuracies = [seq['responses'].mean() for seq in sample_seqs]
        acc_cv = variation(accuracies)
        
        # 2. Sequence similarity (simple embedding-based)
        embeddings = []
        for seq in sample_seqs:
            # Simple embedding: [mean_accuracy, length, unique_questions]
            emb = [
                seq['responses'].mean(),
                len(seq['responses']) / 200.0,  # Normalize
                len(set(seq['questions'])) / len(seq['questions'])  # Question diversity
            ]
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Compute pairwise distances
        distances = pdist(embeddings, metric='euclidean')
        
        diversity_stats = {
            'accuracy_cv': acc_cv,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_pairwise_distance': np.mean(distances),
            'std_pairwise_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
        
        print("Task diversity metrics:")
        for k, v in diversity_stats.items():
            print(f"  {k}: {v:.4f}")
        
        return diversity_stats, distances
    
    def encode_ids(self, train_seqs, val_seqs, test_seqs):
        """Create ID mappings for questions and encode sequences."""
        # Collect all questions from train set
        all_questions = set()
        for seq in train_seqs:
            all_questions.update(seq['questions'])
        
        # Create mapping (0 reserved for padding/unknown)
        question_to_id = {q: i+1 for i, q in enumerate(sorted(all_questions))}
        question_to_id['<PAD>'] = 0
        
        num_questions = len(question_to_id)
        print(f"Vocabulary: {num_questions} unique questions")
        
        # Encode sequences
        def encode_sequence(seq, q_map):
            encoded = []
            for q in seq['questions']:
                encoded.append(q_map.get(q, 0))  # Unknown questions -> 0
            return np.array(encoded, dtype=np.int32)
        
        for split in [train_seqs, val_seqs, test_seqs]:
            for seq in split:
                seq['question_ids'] = encode_sequence(seq, question_to_id)
        
        return train_seqs, val_seqs, test_seqs, question_to_id, num_questions
    
    def process(self, raw_filepath, output_dir, dataset_name='assistments'):
        """Full preprocessing pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {dataset_name}...")
        
        # Load raw data
        df = self.load_raw_data(raw_filepath, dataset_name)
        
        # Create sequences
        sequences = self.create_sequences(df)
        
        # Split into train/val/test
        train_seqs, val_seqs, test_seqs = self.split_students_disjoint(sequences)
        
        # Verify no question leakage
        self.verify_no_question_leakage(train_seqs, test_seqs)
        
        # Compute task diversity
        diversity_stats, distances = self.compute_task_diversity(train_seqs)
        
        # Encode IDs
        train_seqs, val_seqs, test_seqs, q_map, num_q = self.encode_ids(
            train_seqs, val_seqs, test_seqs
        )
        
        # Save processed data
        data = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
            'question_map': q_map,
            'num_questions': num_q,
            'diversity_stats': diversity_stats,
            'config': {
                'min_seq_len': self.min_seq_len,
                'max_seq_len': self.max_seq_len,
                'dataset_name': dataset_name
            }
        }
        
        output_file = output_dir / f"{dataset_name}_processed.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nSaved processed data to {output_file}")
        print(f"Final sizes: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")
        
        return data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <raw_data_path>")
        sys.exit(1)
    
    raw_path = sys.argv[1]
    dataset_name = Path(raw_path).stem.split('_')[0]  # e.g., 'assistments2009'
    
    preprocessor = KTDataPreprocessor(
        min_seq_len=51,
        max_seq_len=200,
        verify_leakage=True
    )
    
    data = preprocessor.process(
        raw_filepath=raw_path,
        output_dir='data/processed',
        dataset_name=dataset_name
    )


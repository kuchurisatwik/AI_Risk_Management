import numpy as np
from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
    """
    Purged K-Fold Cross Validation for financial time-series.
    
    Splits the data into n_splits contiguous blocks. For each fold, the test set 
    is one block, and the training set is all other blocks.
    
    To prevent data leakage from overlapping evaluation windows and market momentum:
      - Purge: Drops `purge_size` rows from the training set immediately BEFORE the test set.
      - Embargo: Drops `embargo_size` rows from the training set immediately AFTER the test set.
    """
    def __init__(self, n_splits=5, purge_size=12, embargo_size=24):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.purge_size = purge_size
        self.embargo_size = embargo_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate block sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + fold_size
            test_indices = indices[start:stop]
            
            # Train set BEFORE the test set (Purged)
            train_stop1 = max(0, start - self.purge_size)
            train_indices_before = indices[0:train_stop1]
            
            # Train set AFTER the test set (Embargoed)
            train_start2 = min(n_samples, stop + self.embargo_size)
            train_indices_after = indices[train_start2:n_samples]
            
            train_indices = np.concatenate([train_indices_before, train_indices_after])
            
            yield train_indices, test_indices
            
            current = stop
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

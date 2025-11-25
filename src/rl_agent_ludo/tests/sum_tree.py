"""
SumTree for efficient prioritized sampling.

A binary tree where parent nodes contain sum of children.
Enables O(log N) sampling and updates.
"""

import numpy as np


class SumTree:
    """
    Binary tree for storing priorities.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize SumTree.
        
        Args:
            capacity: Maximum number of elements
        """
        self.capacity = capacity
        # Full binary tree: 2*capacity - 1 nodes
        # Leaves are at indices [capacity-1, 2*capacity-2]
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Update parent nodes after priority change."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf node for given sample value."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def add(self, priority: float, data: any):
        """Add data with given priority."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> tuple:
        """Sample data by priority value."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])
    
    @property
    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Label = str
Features = Mapping[str, float]


def _gini_from_counts(counts: Mapping[Label, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    impurity = 1.0
    for c in counts.values():
        p = float(c) / float(total)
        impurity -= p * p
    return impurity


def _label_counts(labels: Sequence[Label]) -> Dict[Label, int]:
    counts: Dict[Label, int] = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return counts


@dataclass(frozen=True, slots=True)
class TreeNode:
    feature: Optional[str] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    counts: Optional[Dict[Label, int]] = None

    def is_leaf(self) -> bool:
        return self.counts is not None

    def predict_proba_one(self, x: Features) -> Dict[Label, float]:
        node: TreeNode = self
        while not node.is_leaf():
            if node.feature is None or node.threshold is None or node.left is None or node.right is None:
                break
            value = float(x.get(node.feature, 0.0))
            node = node.left if value <= node.threshold else node.right

        counts = node.counts or {}
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {label: c / float(total) for label, c in counts.items()}

    def predict_one(self, x: Features, *, default: Label = "") -> Label:
        proba = self.predict_proba_one(x)
        if not proba:
            return default
        return max(proba.items(), key=lambda item: item[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        if self.is_leaf():
            return {"type": "leaf", "counts": dict(self.counts or {})}
        return {
            "type": "node",
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "TreeNode":
        node_type = data.get("type")
        if node_type == "leaf":
            counts = data.get("counts") or {}
            return TreeNode(counts={str(k): int(v) for k, v in counts.items()})
        if node_type != "node":
            raise ValueError(f"Invalid TreeNode type: {node_type!r}")
        left = data.get("left")
        right = data.get("right")
        return TreeNode(
            feature=data.get("feature"),
            threshold=float(data.get("threshold")),
            left=TreeNode.from_dict(left) if left else None,
            right=TreeNode.from_dict(right) if right else None,
        )


class TinyDecisionTreeClassifier:
    def __init__(
        self,
        *,
        max_depth: int = 10,
        min_samples_split: int = 30,
        min_samples_leaf: int = 10,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.feature_names = list(feature_names or [])
        self.root: Optional[TreeNode] = None

    def fit(self, X: Sequence[Features], y: Sequence[Label], *, feature_names: Optional[Sequence[str]] = None) -> "TinyDecisionTreeClassifier":
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not X:
            raise ValueError("Empty dataset")

        if feature_names is not None:
            self.feature_names = list(feature_names)
        if not self.feature_names:
            names: set[str] = set()
            for row in X:
                names.update(row.keys())
            self.feature_names = sorted(names)

        self.root = self._build(list(X), list(y), depth=0)
        return self

    def predict_one(self, x: Features, *, default: Label = "") -> Label:
        if not self.root:
            return default
        return self.root.predict_one(x, default=default)

    def predict_proba_one(self, x: Features) -> Dict[Label, float]:
        if not self.root:
            return {}
        return self.root.predict_proba_one(x)

    def to_dict(self) -> Dict[str, Any]:
        if not self.root:
            raise ValueError("Tree not trained")
        return {
            "version": 1,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_names": list(self.feature_names),
            "root": self.root.to_dict(),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "TinyDecisionTreeClassifier":
        model = TinyDecisionTreeClassifier(
            max_depth=int(data.get("max_depth", 4)),
            min_samples_split=int(data.get("min_samples_split", 30)),
            min_samples_leaf=int(data.get("min_samples_leaf", 10)),
            feature_names=data.get("feature_names") or [],
        )
        model.root = TreeNode.from_dict(data["root"])
        return model

    def _build(self, X: List[Features], y: List[Label], *, depth: int) -> TreeNode:
        counts = _label_counts(y)
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or _gini_from_counts(counts) <= 1e-12
        ):
            return TreeNode(counts=counts)

        best = self._best_split(X, y)
        if best is None:
            return TreeNode(counts=counts)
        feature, threshold, left_idx, right_idx = best

        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return TreeNode(counts=counts)

        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]

        return TreeNode(
            feature=feature,
            threshold=threshold,
            left=self._build(left_X, left_y, depth=depth + 1),
            right=self._build(right_X, right_y, depth=depth + 1),
        )

    def _best_split(
        self, X: Sequence[Features], y: Sequence[Label]
    ) -> Optional[Tuple[str, float, List[int], List[int]]]:
        n = len(y)
        parent_counts = _label_counts(y)
        parent_gini = _gini_from_counts(parent_counts)
        if parent_gini <= 1e-12:
            return None

        best_feature = None
        best_threshold = None
        best_score = None
        best_left: List[int] = []
        best_right: List[int] = []

        for feature in self.feature_names:
            values = [float(row.get(feature, 0.0)) for row in X]
            order = sorted(range(n), key=lambda i: values[i])
            sorted_values = [values[i] for i in order]
            sorted_labels = [y[i] for i in order]

            # If all identical, can't split.
            if sorted_values[0] == sorted_values[-1]:
                continue

            left_counts: Dict[Label, int] = {}
            right_counts = _label_counts(sorted_labels)

            for i in range(n - 1):
                label = sorted_labels[i]
                left_counts[label] = left_counts.get(label, 0) + 1
                right_counts[label] -= 1
                if right_counts[label] <= 0:
                    right_counts.pop(label, None)

                v = sorted_values[i]
                nv = sorted_values[i + 1]
                if v == nv:
                    continue

                left_n = i + 1
                right_n = n - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                threshold = (v + nv) / 2.0
                score = (left_n / n) * _gini_from_counts(left_counts) + (right_n / n) * _gini_from_counts(right_counts)

                if best_score is None or score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                    best_left = [order[j] for j in range(left_n)]
                    best_right = [order[j] for j in range(left_n, n)]

        if best_feature is None or best_threshold is None or best_score is None:
            return None

        # No meaningful improvement.
        if best_score >= parent_gini - 1e-9:
            return None

        return best_feature, best_threshold, best_left, best_right

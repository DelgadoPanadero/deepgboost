"""Mixin for automatic categorical feature encoding."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted


class CategoricalEncoderMixin:
    """
    Mixin that transparently encodes categorical (string) features via
    one-hot encoding before passing data to the underlying model.

    Categorical columns are detected at ``fit`` time as any column whose
    values cannot be cast to ``float64``.  Numerical columns are preserved
    as-is and placed first; the OHE columns are appended after them.

    After ``fit``, the following attributes are set:

    ``categorical_columns_`` : list[int]
        Original column indices that were one-hot encoded.
    ``numerical_columns_`` : list[int]
        Original column indices kept as numeric.
    ``ohe_`` : OneHotEncoder or None
        Fitted encoder, or ``None`` when no categorical columns were found.
    """

    def _fit_transform_X(self, X) -> np.ndarray:
        """Fit the encoder on *X* and return the transformed float array."""
        X_arr = self._to_object_array(X)
        cat_cols = self._find_categorical_columns(X_arr)
        self.categorical_columns_: list[int] = cat_cols
        self.numerical_columns_: list[int] = [
            i for i in range(X_arr.shape[1]) if i not in cat_cols
        ]
        if cat_cols:
            self.ohe_ = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            X_cat = self.ohe_.fit_transform(X_arr[:, cat_cols])
            return self._assemble(X_arr, X_cat)
        self.ohe_ = None
        return check_array(X_arr, dtype=np.float64)

    def _transform_X(self, X) -> np.ndarray:
        """Apply the fitted encoder to *X* and return a float array."""
        check_is_fitted(self, "categorical_columns_")
        X_arr = self._to_object_array(X)
        if self.categorical_columns_:
            X_cat = self.ohe_.transform(X_arr[:, self.categorical_columns_])
            return self._assemble(X_arr, X_cat)
        return check_array(X_arr, dtype=np.float64)

    def _assemble(self, X_arr: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        if self.numerical_columns_:
            X_num = X_arr[:, self.numerical_columns_].astype(np.float64)
            return np.hstack([X_num, X_cat])
        return X_cat

    @staticmethod
    def _to_object_array(X) -> np.ndarray:
        """Convert any array-like (including DataFrames) to an object ndarray."""
        if hasattr(X, "to_numpy"):
            return X.to_numpy(dtype=object)
        return np.asarray(X)

    @staticmethod
    def _find_categorical_columns(X_arr: np.ndarray) -> list[int]:
        """Return indices of columns that cannot be cast to float64."""
        if np.issubdtype(X_arr.dtype, np.number):
            return []
        cat: list[int] = []
        for i in range(X_arr.shape[1]):
            try:
                X_arr[:, i].astype(np.float64)
            except (ValueError, TypeError):
                cat.append(i)
        return cat

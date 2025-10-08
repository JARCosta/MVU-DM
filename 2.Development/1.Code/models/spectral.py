from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse.linalg import eigsh

from utils import stamp
import utils
from datasets import datasets

class Spectral(ABC):
    def __init__(self, model_args:dict):
        self.model_args = model_args
        self.embedding_ = None
        self.kernel_ = None
        # self.plot = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': '3d'})

    @abstractmethod
    def _fit(self, X: np.ndarray):
        return self

    def fit(self, X: np.ndarray):
        """Computes the kernel matrix."""
        stamp.set()
        
        ret = self._fit(X)
        stamp.print(f"*\t {self.model_args['model']}\t fit")
        return ret

    def _restrict_components(self, eigenvalues:np.ndarray, eigenvectors:np.ndarray, use_smallest: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Restrict the number of components to the top `n_components`."""
        from datasets import datasets
        
        # Sort eigenvalues and eigenvectors in descending eigenvalue order (unless use_smallest=True)
        if use_smallest:
            idx = np.argsort(eigenvalues)  # ascending order for smallest eigenvalues
            representation_threshold = getattr(self, 'representation_threshold', 0.005)
        else:
            idx = np.argsort(eigenvalues)[::-1]  # descending order for largest eigenvalues
            representation_threshold = getattr(self, 'representation_threshold', 0.995)

        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        # if self.model_args['verbose']:
        #     print(f"Sorted Eigenvalues:", eigenvalues)

        for idx in range(len(eigenvalues)): # Possible because eigenvalues are sorted
            representation = np.sum(eigenvalues[:idx]) / np.sum(eigenvalues)
            if representation > representation_threshold:
                break
        smart_components, smart_representation = idx, representation

        utils.important(f"Smart selected {smart_components} components ({smart_representation*100:.2f}%).")

        intrinsic_components = datasets[self.model_args['dataname']].get('intrinsic', None)
        comparative_components = datasets[self.model_args['dataname']].get('comparative', None)
        MLE_components = datasets[self.model_args['dataname']].get('MLE', None)
        if intrinsic_components is not None:
            if smart_components != intrinsic_components:
                intrinsic_representation = np.sum(eigenvalues[:intrinsic_components]) / np.sum(eigenvalues)
                utils.warning(f"Forcing the selection to the intrinsic threshold ({intrinsic_components}; {intrinsic_representation*100:.2f}%).")
                self.model_args['restricted'] = smart_components
                smart_components = intrinsic_components
        elif comparative_components is not None:
            if smart_components != comparative_components:
                comparative_representation = np.sum(eigenvalues[:comparative_components]) / np.sum(eigenvalues)
                utils.warning(f"Forcing the selection to the comparative threshold ({comparative_components}; {comparative_representation*100:.2f}%).")
                self.model_args['restricted'] = smart_components
                smart_components = comparative_components
        elif MLE_components is not None:
            if smart_components != MLE_components:
                MLE_representation = np.sum(eigenvalues[:MLE_components]) / np.sum(eigenvalues)
                utils.warning(f"Forcing the selection to the MLE threshold ({MLE_components}; {MLE_representation*100:.2f}%).")
                self.model_args['restricted'] = smart_components
                smart_components = MLE_components
        eigenvalues = eigenvalues[:smart_components]
        eigenvectors = eigenvectors[:, :smart_components]

        return eigenvalues, eigenvectors

    def _eigenvalue_decomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition of the kernel matrix.
        Subclasses can override this method to implement different eigenvalue problems.
        
        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues in ascending order
        eigenvectors : np.ndarray
            Corresponding eigenvectors
        """
        if self.kernel_ is None:
            utils.warning("Kernel matrix is not initialized. Run fit(X) first.")
            return None, None
        if np.any(np.isnan(self.kernel_)):
            utils.hard_warning("Kernel matrix contains NaNs. Skipping.")
            return None, None

        if not np.allclose(self.kernel_, self.kernel_.T):
            raise ValueError(f"Kernel matrix is not symmetric. {len(np.where(~np.isclose(self.kernel_, self.kernel_.T, atol=1e-8))[0])} values differ")
        
        eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_)
        return eigenvalues, eigenvectors

    def _transform(self) -> np.ndarray | None:
        eigenvalues, eigenvectors = self._eigenvalue_decomposition()
        
        if eigenvalues is None or eigenvectors is None:
            return None
        
        eigenvalues, eigenvectors = self._restrict_components(eigenvalues, eigenvectors, getattr(self, 'use_smallest_eigenvalues', False))

        if getattr(self, 'apply_sqrt_scaling', True):
            # eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative eigenvalues
            self.embedding_ = eigenvectors @ np.diag(eigenvalues ** 0.5)  # Project data
        else:
            self.embedding_ = eigenvectors  # Use eigenvectors directly (for LLE-type methods)
        
        return self.embedding_
    
    def transform(self):
        """Performs spectral embedding using the top eigenvectors."""
        stamp.set()

        self._transform()
        stamp.print(f"*\t {self.model_args['model']}\t transform")
        
        return self.embedding_


    def fit_transform(self, X: np.ndarray):
        """Fits the model and computes embeddings."""
        return self.fit(X).transform()

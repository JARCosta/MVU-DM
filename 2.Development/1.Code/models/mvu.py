import numpy as np
from scipy.sparse import csgraph
import matlab.engine
import os

import plot
import utils
import models

def launch_matlab():
    eng = matlab.engine.start_matlab()#background=True)
    # Add ays the directory containing the MATLAB function to the MATLAB path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    eng.addpath(current_dir)#, nargout=0)
    utils.stamp.print(f"*\t MATLAB started")
    return eng

class MVU(models.Neighbourhood):

    def __init__(self, model_args:dict, n_neighbors:int):
        super().__init__(model_args, n_neighbors)
        self._mode = 0
        self.matlab_eng = None

    def _neigh_matrix(self, X:np.ndarray):
        return utils.neigh_matrix(X, self.n_neighbors, bidirectional=True)
    
    def _fit(self, X:np.ndarray):
        """Fit the MVU model and compute the low-dimensional embeddings using MATLAB."""
        
        cc, labels = csgraph.connected_components(self.NM, directed=False)
        if cc > 1:
            self.model_args['artificial_connected'] = True
            # self.NM = models.graph.connect_components_sklearn(X, self.NM, cc, labels)
            self.NM = models.graph.connect_components_default(X, self.NM, cc, labels)
            # self.NM = models.graph.connect_components_k2(X, self.NM, cc, labels)

            # temp_NM = utils.neigh_matrix(X, self.n_neighbors, bidirectional=True)
            # temp_NM = temp_NM - self.NM
            # plot.plot(X, temp_NM, title="MVU Input Graph")


        np.set_printoptions(threshold=200)
        
        
        if self.model_args['verbose']:
            print()
            print(f"Original distances:")
            print(f"\t min: {np.min(self.NM[self.NM > 0]):.4f}, max: {np.max(self.NM):.4f}")
        
        self.ratio = None
        if hasattr(self, 'ratio') and self.ratio is None:
            scaling_ratio = round(np.log10(np.max(self.NM))) - 2
            scaling_ratio = 10**(-scaling_ratio)

            self.NM = self.NM * scaling_ratio
            self.recovering_ratio = 1/scaling_ratio

            if self.model_args['verbose']:
                print(f"Scaling ratio: {scaling_ratio}")
                print(f"Scaled distances:")
                print(f"\t min: {np.min(self.NM[self.NM > 0]):.4f}, max: {np.max(self.NM):.4f}")
                print()
                
        self.NM[self.NM > 0] = self.NM[self.NM > 0]**2
        NM_matlab = matlab.double(self.NM.tolist())
        n_matlab = matlab.double(X.shape[0])

        if self.model_args['cached']:
            cache = utils.get_cached(self.model_args['model'], self.model_args['dataname'], self.n_neighbors, X, self.NM)
            print(f"loaded from cache: {cache is not None}")
            if cache is not None:
                self.kernel_ = cache['kernel']
                self.X = cache['X']
                self.NM = cache['NM']
                return self
        
        utils.stamp.print(f"*\t calling MATLAB")
        if self.matlab_eng is None:
            self.matlab_eng = launch_matlab()
        
        self.eps = self.model_args['precision']
        K_matlab, cvx_status = self.matlab_eng.solve_mvu_optimization(n_matlab, NM_matlab, self.eps, nargout=2)
        utils.stamp.print(f"*\t CVX\t {cvx_status}\t trace: {np.trace(K_matlab):.2f}\t precision:{self.eps}")
        self.eps = None

        self.model_args['status'] = cvx_status
        if cvx_status != 'Solved':
            utils.warning(f"MATLAB couldn't solve optimally: {cvx_status}")
        # Convert back to numpy array and ensure it's float64
        self.kernel_ = np.array(K_matlab, dtype=np.float64)
        self.X = X

        utils.save_cached(self.model_args['model'], self.model_args['dataname'], self.n_neighbors, self.X, self.NM, self.kernel_, self.id if hasattr(self, 'id') else None)

        return self

    def _transform(self):
        embedding = super()._transform()
        if embedding is None:
            return None

        if hasattr(self, 'recovering_ratio') and self.recovering_ratio is not None:
            
            old_embedding = embedding.copy()
            embedding = embedding * self.recovering_ratio

            if self.model_args['verbose']:
                if callable(getattr(self, '_neigh_matrix', None)):
                    NM_before = self._neigh_matrix(old_embedding)
                else:
                    NM_before = utils.neigh_matrix(old_embedding, self.n_neighbors, bidirectional=True)
                
                print()
                print(f"Inverse scaling ratio: {self.recovering_ratio}")
                print(f"Linearized scaled distances:")
                print(f"\t min: {np.min(NM_before[NM_before != 0]):.4f} , max: {np.max(NM_before):.4f}")
                print()

            self.recovering_ratio = None

        if self.model_args['verbose'] and callable(getattr(self, '_neigh_matrix', None)):
            NM = self._neigh_matrix(embedding)
            print(f"Output distances:")
            print(f"\t min: {np.min(NM[NM > 0]):.4f}, max: {np.max(NM):.4f}")



        if self.model_args.get('status', None) != 'Solved':
            utils.hard_warning(f"MVU embedding failed: {self.model_args['status']}. Scaling output to match the original scale.")
            Yi_NM = utils.neigh_matrix(embedding, self.n_neighbors)
            min_Yi, max_Yi = np.min(Yi_NM[Yi_NM != 0]), np.max(Yi_NM)


            Xi_NM = utils.neigh_matrix(self.X, self.n_neighbors)
            min_Xi, max_Xi = np.min(Xi_NM[Xi_NM != 0]), np.max(Xi_NM)
            # recovering_ratio = np.mean([min_Xi/min_Yi, max_Xi/max_Yi])
            recovering_ratio = 10**(round(np.log10(max_Xi/max_Yi)))
            
            embedding = embedding * recovering_ratio

            if self.model_args['verbose']:
                print(f"Recovery scalling ratio: {recovering_ratio:.4f}")
                Yi_NM = utils.neigh_matrix(embedding, self.n_neighbors)
                print(f"After recovery scalling:")
                print(f"\t min: {np.min(Yi_NM[Yi_NM != 0]):.4f}, max: {np.max(Yi_NM):.4f}")
                print()

            temp_NM = utils.neigh_matrix(self.X, self.n_neighbors, bidirectional=True) # NM of the original data (no inter-component connections)
            cc, labels = csgraph.connected_components(temp_NM, directed=False)
            Yl = [embedding[labels == c] for c in range(cc)] # components split by the NM of original data (no inter-component connections)
            if len(Yl) > 1 and embedding.shape[1] > 1 and self.model_args['overlap']:
                color = [[i]*len(Yl[i]) for i in range(len(Yl))]
                color = np.hstack(color)
                plot.plot(np.vstack(Yl), c=color, title=f"{self.model_args['dataname']} Y component overlap", block=False)
                plot.overlap(Yl, self.model_args, show=self.model_args['overlap'])

        self.embedding_ = embedding


class Ineq(MVU):
    def __init__(self, model_args:dict, n_neighbors:int):
        super().__init__(model_args, n_neighbors)
        self._mode = 1

class Nystrom(models.extensions.Nystrom, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, ratio:int=None, subset_indices:list=None):
        MVU.__init__(self, model_args, n_neighbors)
        super().__init__(ratio=ratio, subset_indices=subset_indices)

class ENG(models.extensions.ENG, MVU):
    pass

class Adaptative(models.extensions.Adaptative, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, k_max:int=10, eta:float=0.95):
        MVU.__init__(self, model_args, n_neighbors)
        super().__init__(k_max, eta)

class Adaptative2(models.extensions.Adaptative2, MVU):
    def __init__(self, model_args:dict, n_neighbors:int, k_max:int=10, eta:float=0.95):
        MVU.__init__(self, model_args, n_neighbors)
        super().__init__(k_max, eta)
    
class Our(models.extensions.Our, MVU):
    def __init__(self, model_args:dict, n_neighbors:int):
        MVU.__init__(self, model_args, n_neighbors)
        super().__init__(bidirectional=True)

class Based(models.extensions.Based, MVU):
    def __init__(self, model_args:dict, k1:int, k2:int):
        MVU.__init__(self, model_args, n_neighbors=k1)
        super().__init__(k1, k2)


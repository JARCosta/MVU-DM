# models/__init__.py

from .spectral import *
from .neighbourhood import *
from .extensions import *

from .graph import *

from .mvu import *
from .isomap import *
from .pca import *
from .kpca import *
from .le import *
from .lle import *
from .ltsa import *
from .hlle import *
from .tsne import *

def run(X, model_args, labels:np.ndarray=None):
    import sklearn.manifold
    from utils import stamp

    stamp.set()

    ########################################################
    # PCA ##################################################
    ########################################################
    
    if model_args['model'].lower() == "pca":

        model = models.pca.PCA(model_args, )
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "kpca.skl":
        from sklearn.decomposition import KernelPCA
        model = KernelPCA(n_components=model_args['#neighs'], kernel='rbf')
        Y = model.fit_transform(X)
    
    
    elif model_args['model'].lower() == "kpca":

        model = models.kpca.KPCA(model_args, )
        Y = model.fit_transform(X)


    ########################################################
    # Isomap ###############################################
    ########################################################

    elif model_args['model'].lower() == "isomap":

        model = models.isomap.Isomap(model_args, model_args['#neighs'], )
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "isomap.skl":

        model = sklearn.manifold.Isomap(n_neighbors=model_args['#neighs'],)
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "isomap.nystrom":

        model = models.isomap.Nystrom(model_args, ratio=0.2, n_neighbors=model_args['#neighs'])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "isomap.eng":

        model = models.isomap.ENG(model_args, n_neighbors=model_args['#neighs'])
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "isomap.adaptative":

        model = models.isomap.Adaptative(model_args, n_neighbors=model_args['#neighs'], k_max=20, eta=1e-3)
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "isomap.our":

        model = models.isomap.Our(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)
        

    ########################################################
    # MVU ##################################################
    ########################################################

    elif model_args['model'].lower() == "mvu":

        model = models.mvu.MVU(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "mvu.ineq":

        model = models.mvu.Ineq(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "mvu.nystrom":

        model = models.mvu.Nystrom(model_args, model_args['#neighs'], 0.1)
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "mvu.eng":

        model = models.mvu.ENG(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "mvu.adaptative":

        model = models.mvu.Adaptative(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "mvu.our":

        model = models.mvu.Our(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)
    
    elif model_args['model'].lower() == "mvu.based":

        model = models.mvu.Based(model_args, k1=model_args['#neighs'], k2=model_args['#neighs'])
        if labels is not None:
            model.model_args['labels'] = labels
        Y = model.fit_transform(X)


    ########################################################
    # LE ###################################################
    ########################################################

    elif model_args['model'].lower() == "le":
        
        model = models.le.LaplacianEigenmaps(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "le.skl":

        model = sklearn.manifold.SpectralEmbedding(n_neighbors=model_args['#neighs'])
        Y = model.fit_transform(X)

    elif model_args['model'].lower() == "le.eng":
            
        model = models.le.ENG(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)

    ########################################################
    # LLE ##################################################
    ########################################################

    elif model_args["model"].lower() == "lle":
        
        model = models.lle.LocallyLinearEmbedding(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)
        

    elif model_args["model"].lower() == "lle.skl":
        
        model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=model_args['#neighs'])
        Y = model.fit_transform(X)

    elif model_args["model"].lower() == "lle.eng":

        model = models.lle.ENG(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)

    ########################################################
    # HLLE #################################################
    ########################################################

    elif model_args["model"].lower() == "hlle":
        
        model = models.hlle.HessianLLE(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)

    elif model_args["model"].lower() == "hlle.skl":
    
        # if model_args['#neighs'] <= model_args['intrinsic'] * (model_args['intrinsic'] + 3) / 2:
        #     # raise ValueError("n_neighbors must be greater than [n_components * (n_components + 3) / 2]")
        #     utils.warning(f"n_neighbors must be greater than [n_components * (n_components + 3) / 2]")
        #     return None
        
        try:
            model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=model_args['#neighs'], method='hessian', eigen_solver='dense')
            Y = model.fit_transform(X)
        except ValueError:
            utils.hard_warning(f"n_neighbors must be greater than [n_components * (n_components + 3) / 2]")
            return None
    

    elif model_args["model"].lower() == "hlle.eng":
        
        model = models.hlle.ENG(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)

    ########################################################
    # T-SNE ################################################
    ########################################################
    
    
    elif model_args["model"].lower() == "tsne":
        # t-SNE typically uses 2D for visualization, but can use intrinsic if available
        target_dim = model_args.get('intrinsic', 2) if model_args.get('intrinsic') is not None else 2
        
        model = models.tsne.TSNE(model_args, target_dimension=target_dim)
        Y = model.fit_transform(X)
        

    elif model_args["model"].lower() == "tsne.skl":
        
        model = sklearn.manifold.TSNE()
        Y = model.fit_transform(X)
        
    
    ########################################################
    # LTSA #################################################
    ########################################################

    elif model_args["model"].lower() == "ltsa":
        
        model = models.ltsa.LTSA(model_args, model_args['#neighs'])
        Y = model.fit_transform(X)
        

    elif model_args["model"].lower() == "ltsa.skl":
        
        model = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=model_args['#neighs'], method='ltsa', eigen_solver='dense')
        Y = model.fit_transform(X)
    
    ########################################################
    # UMAP #################################################
    ########################################################

    elif model_args["model"].lower() == "umap.lib":
        import umap  # Ensure the umap-learn library is installed
        model = umap.UMAP(n_neighbors=model_args['#neighs'])
        Y = model.fit_transform(X)
        

    ########################################################
    # END ##################################################
    ########################################################
    else:
        raise ValueError(f"Unknown model name {model_args['model']}.")
    return Y

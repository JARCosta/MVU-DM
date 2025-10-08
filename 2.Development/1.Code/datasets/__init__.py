# Dataset modules
from . import swiss
from . import s_curve
from . import moons
from . import artificial
from . import natural

datasets = {
    "swiss": {
        "func": swiss.default,
        "intrinsic": 2,
        "comparative": 2,
        "MLE": 2,
    },
    "helix": {
        "func": artificial.helix,
        "intrinsic": 2,
        "comparative": 1,
        "MLE": 2,
    },
    "twinpeaks": {
        "func": artificial.twinpeaks,
        "intrinsic": 2,
        "comparative": 2,
        "MLE": 2,
    },
    "broken.swiss": {
        "func": swiss.broken,
        "intrinsic": 2,
        "comparative": 2,
        "MLE": 2,
    },
    "difficult": {
        "func": artificial.difficult,
        "intrinsic": 5,
        "comparative": 5,
        "MLE": 6,
    },

    "teapots": {
        "func": natural.teapots,
        "natural": True,
        "MLE": 2,
    },
    "mnist": {
        "func": natural.mnist,
        "natural": True,
        "comparative": 20,
        "MLE": 13,
    },
    "coil20": {
        "func": natural.coil20,
        "natural": True,
        "comparative": 5,
        "MLE": 4,
    },
    "orl": {
        "func": natural.orl,
        "natural": True,
        "comparative": 8,
        "MLE": 8,
    },
    "hiva": {
        "func": natural.hiva,
        "natural": True,
        "comparative": 15,
        "MLE": 15,
    },
    "mit-cbcl": {
        "func": natural.mit_cbcl,
        "natural": True,
        "MLE": 13,
    },
    "olivetti": {
        "func": natural.olivetti,
        "natural": True,
        "MLE": 8,
    },
    "cifar10": {
        "func": natural.cifar10,
        "natural": True,
        "MLE": 28,
    },
    "animalface": {
        "func": natural.animalface,
        "natural": True,
    },
    "flowers": {
        "func": natural.flowers,
        "natural": True,
    },
    "carnivores": {
        "func": natural.carnivores,
        "natural": True,
    },
    "cancer": {
        "func": natural.cancer,
        "natural": True,
    },
    "imagenet": {
        "func": natural.imagenet,
        "natural": True,
    },
    "inat": {
        "func": natural.inat,
        "natural": True,
    },


    "parallel.swiss": {
        "func": swiss.parallel,
        "intrinsic": 2,
        "MLE": 2,
    },
    "broken.s_curve": {
        "func": s_curve.broken,
        "intrinsic": 2,
        "MLE": 2,
    },
    "four.moons": {
        "func": moons.four,
        "intrinsic": 2,
        "MLE": 1,
    },
    "two.swiss": {
        "func": swiss.two,
        "intrinsic": 2, #3 # TODO: confirm
        "MLE": 2,
    },


    "s_curve": {
        "func": s_curve.default,
        "intrinsic": 2,
        "MLE": 2,
    },

    "moons": {
        "func": moons.default,
        "intrinsic": 2,
        "MLE": 1,
    },

}

def get_dataset(dataname:str, n_points:int, noise:float, random_state:int=None, cached:bool=True):
    import os
    import numpy as np

    import utils
    from utils import stamp


    datasets_dir = "cache/datasets"
    npz_path = f"{datasets_dir}/{dataname}.npz"

    # Prefer loading from NPZ (named arrays). If corrupted, delete and regenerate.
    if "natural" in datasets[dataname] and os.path.exists(npz_path) and cached:
        try:
            data = np.load(npz_path, allow_pickle=False)
            X = data["X"]
            labels = data["labels"] if "labels" in data.files else None
            t = data["t"] if "t" in data.files else None
            stamp.print_set(f"*\t {dataname} dataset loaded from {npz_path} {X.shape}.")
            if datasets[dataname].get("natural", False) or X.shape[0] == n_points:
                return X, labels, t
        except Exception as e:
            utils.warning(f"{dataname} cache at {npz_path} unreadable ({e}). Recomputing...")
            try:
                os.remove(npz_path)
            except OSError:
                pass

    np.random.seed(random_state) if random_state is not None else None
    
    X, labels, t = datasets[dataname]['func']() if "natural" in datasets[dataname] else datasets[dataname]['func'](n_points, noise)
    X = X - X.mean(0)
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    labels = labels[shuffle_indices] if labels is not None else None
    t = t[shuffle_indices] if t is not None else None

    stamp.print_set(f"*\t {dataname} loaded {X.shape}")

    if "natural" in datasets[dataname]:
        # Ensure cache directory exists and save in NPZ without pickles when possible
        os.makedirs(datasets_dir, exist_ok=True)
        save_payload = {"X": X}
        if labels is not None:
            save_payload["labels"] = labels
        if t is not None:
            save_payload["t"] = t
        np.savez_compressed(npz_path, **save_payload)
        stamp.print_set(f"*\t {dataname} dataset saved to {npz_path}.")
    
    return X, labels, t
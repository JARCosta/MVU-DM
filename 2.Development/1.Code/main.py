import multiprocessing
import os

import utils

os.environ['LOKY_MAX_CPU_COUNT'] = str(12)
# os.environ['OMP_NUM_THREADS'] = str(12)

# os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
# os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

def model_func(X, labels, model_args:dict):
    import models
    import plot

    
    utils.important(f"Running {model_args['model']} on {model_args['dataname']} with {model_args['#neighs']} neighbors")
    model_stamp = utils.Stamp()
    Y = models.run(X, model_args, labels)
    time_diff, _ = model_stamp.get()
    model_stamp.print(f"*\t {model_args['model']}\t {model_args['dataname']}(k={model_args['#neighs']})")

    if Y is None:
        utils.warning(f"could not compute Y for {model_args['model']} on {model_args['dataname']}(k={model_args['#neighs']})")
        utils.pop_measure(model_args['dataname'], model_args['model'], model_args['#neighs'])
        return None

    if 'labels' in model_args and model_args['labels'] is not None:
        labels = model_args['labels']
    if 'X' in model_args and model_args['X'] is not None:
        X = model_args['X']
    One_nn, T, C = utils.compute_measures(X, Y, labels, model_args["#neighs"])
    if model_args['measure']:
        utils.add_measure(model_args, Y.shape[1], One_nn, T, C, time_diff.total_seconds())
    
    if model_args['preview']:
        plot.plot_scales(X, c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']}({model_args['#neighs']}) Original data")
        plot.plot_scales(Y, c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']}({model_args['#neighs']}) Embedding", legend=model_args)
        # if Y.shape[1] > 3:
        #     plot.plot_scales(Y[:, 3:], c=labels, block=False, title=f"{model_args['model']} {model_args['dataname']}({model_args['#neighs']}) Embedding, 3-D", legend=model_args)
    
    model_args['X'] = None
    model_args['labels'] = None
    
    with open(f"2.Development/2.Data/results.csv", "a") as f:
        f.write(f"{time_diff.total_seconds()};{model_args['model']};{model_args['dataname']};{model_args['#neighs']};{One_nn};{T};{C}\n")

def main(paper:str, model_list:list, dataset_list:list, n_points:int, k_small:int, k_large:int, threaded:bool, preview:bool, overlap:bool, verbose:bool, measure:bool, pause:bool, seed:int, noise:float, precision:float, cached:bool, photos:bool) -> None:
    from datasets import get_dataset, datasets
    import matplotlib.pyplot as plt

    import plot

    for model in model_list:
        for dataname in dataset_list:
            X, labels, t = get_dataset(dataname, n_points, noise, random_state=seed, cached=cached)
            for n_neighbors in range(k_small, k_large + 1):
                
                model_args = {
                    "paper": paper,
                    "model": model,
                    "dataname": dataname,
                    
                    "intrinsic": datasets[dataname]['intrinsic'] if 'intrinsic' in datasets[dataname] else None,
                    "#points": X.shape[0],
                    "#neighs": n_neighbors,
                    "precision": precision,
                    
                    "cached": cached,
                    "threaded": threaded,
                    "measure": measure,
                    "preview": preview,
                    "overlap": overlap,
                    "verbose": verbose,
                    "photos": photos,
                    
                    "status": None,
                    "restricted": None,
                    "artificial_connected": None,
                }

                model_func(X, labels, model_args)
                if pause or preview:
                    input("Press Enter to continue...")
                    plt.close('all')

            if measure:
                utils.plot_measures(**model_args)
            
    print("finished")




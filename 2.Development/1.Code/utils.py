from datetime import datetime
import numpy as np


class bcolors:
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    REVERSE = '\033[7m'

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    ORANGE = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    LIGHTGRAY = '\033[37m'

    DARKGRAY = '\033[90m'
    LIGHTRED = '\033[91m'
    LIGHTGREEN = '\033[92m'
    YELLOW = '\033[93m'
    LIGHTBLUE = '\033[94m'
    PINK = '\033[95m'
    LIGHTCYAN = '\033[96m'


class Stamp:
    def __init__(self):
        self.timestamp = datetime.now()
        self.color = bcolors.GREEN

    def set(self):
        self.timestamp = datetime.now()
    
    def get(self):
        time_diff = datetime.now() - self.timestamp
        minutes = time_diff.seconds // 60
        seconds = time_diff.seconds % 60
        milliseconds = time_diff.microseconds // 1000
        time_str = '{:02}:{:02}.{:03}'.format(minutes, seconds, milliseconds)
        return time_diff, time_str
        

    def print(self, text:str="*", end:str="\n"):
        time_diff, time_str = self.get()
        self.color = bcolors.RED if time_diff.total_seconds() > 1 else bcolors.GREEN
        print(text.replace("*", self.color + time_str + bcolors.ENDC ), end=end)
        return self
    
    def print_set(self, text:str="*", end:str="\n"):
        return self.print(text, end=end).set()

stamp:Stamp = None

def warning(text:str):
    print(f"{bcolors.ORANGE}Warning: {text}{bcolors.ENDC}")

def hard_warning(text:str):
    print(f"{bcolors.RED}Warning: {text}{bcolors.ENDC}")

def important(text:str):
    print(f"{bcolors.BLUE}{text}{bcolors.ENDC}")

############################################################
# K-NEIGHBOURHOOD ##########################################
############################################################


def neigh_matrix(data, n_neighbors, bidirectional=False, common_neighbors=False):
    return neigh_graph(data, n_neighbors, bidirectional, common_neighbors).toarray()

def neigh_graph(data, n_neighbors, bidirectional=False, common_neighbors=False):
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    k = n_neighbors
    n_samples = data.shape[0]

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    neigh_matrix:sp.csr_matrix = neigh.kneighbors_graph(mode="distance") # TODO: this scales very badly with the number of points

    if bidirectional:
        neigh_bidirectional = neigh_matrix.maximum(neigh_matrix.T)
        neigh_matrix = sp.triu(neigh_bidirectional)


    if common_neighbors: # if i and j are my neighs, then i and j are neighs of each other
        raise NotImplementedError("Common neighbors not implemented for csr matrix")
        adj_bool = neigh_matrix > 0
        common_neigh = adj_bool.astype(int) @ adj_bool.astype(int).T
        common_neigh = (common_neigh > 0) & (~adj_bool) # selection of common neighbors that are not already connected
        for i,j in zip(*np.where(common_neigh)):
            shortest_path = np.inf
            # Find the k node that minimizes the distance between i and j
            for k in range(n_samples):
                if neigh_matrix[i,k] > 0 and neigh_matrix[j,k] > 0:
                    shortest_path = min(shortest_path, neigh_matrix[i,k] + neigh_matrix[j,k])
            neigh_matrix[i,j] = shortest_path
            neigh_matrix[j,i] = shortest_path
    return neigh_matrix

############################################################
# MEASURES #################################################
############################################################

def TC(X, Y, n_neighbors) -> tuple[float, float]:
    
    n_samples = X.shape[0]
    if n_samples == 8 and n_neighbors == 5:
        hard_warning("n_samples == 8 and n_neighbors == 5, can't compute measures")
        return 0.0, 0.0

    NM_X = neigh_matrix(X, n_neighbors) # need symetrix
    NM_Y = neigh_matrix(Y, n_neighbors) # need symetrix

    NM_X = np.where(NM_X != 0, 1, 0)
    NM_Y = np.where(NM_Y != 0, 1, 0)

    NM_T = NM_Y - NM_X

    NM_T[np.where(NM_T == -1)] = 0
    D_X = intra_dist_matrix(X)
    R_X = np.argsort(np.argsort(D_X, axis=1), axis=1)
    T = NM_T * R_X
    T[T != 0] -= n_neighbors
    T = 1 - (2 / (n_samples * n_neighbors * (2*n_samples - 3*n_neighbors - 1))) *  np.sum(T)

    NM_C = NM_X - NM_Y
    NM_C[np.where(NM_C == -1)] = 0
    D_Y = intra_dist_matrix(Y)
    R_Y = np.argsort(np.argsort(D_Y, axis=1), axis=1)
    C = NM_C * R_Y
    C[C != 0] -= n_neighbors
    C = 1 - (2 / (n_samples * n_neighbors * (2*n_samples - 3*n_neighbors - 1))) *  np.sum(C)

    return round(float(T), 4), round(float(C), 4)

def  one_NN(Y, labels) -> float:
    if labels is None:
        warning("labels is None")
        return None
    
    NG = neigh_graph(Y, 1)
    nearest_neigh_idx = [NG[i].indices[0] for i in range(Y.shape[0])]
    Y_labels = labels[nearest_neigh_idx]

    one_NN = np.count_nonzero(Y_labels - labels) / labels.shape[0]
    return round(float(one_NN), 5)

def compute_measures(X, Y, labels, n_neigh) -> tuple[float, float, float]:
    
    stamp.set()
    One_nn = one_NN(Y, labels)
    T, C = TC(X, Y, n_neigh)
    stamp.print_set(f"*\t 1_NN, T, C \t {One_nn}, {T}, {C}")

    return One_nn, T, C








def _get_json(best:bool=False):
    from json import loads, JSONDecodeError

    try:
        if best:
            with open("2.Development/2.Data/measures.best.json", "r") as f:
                measures = loads(f.read())
        else:
            with open("2.Development/2.Data/measures.all.json", "r") as f:
                measures = loads(f.read())
    except (FileNotFoundError, JSONDecodeError):
        measures = {}
    return measures

def get_measures(dataname:str, model:str, best:bool=False):
    measures = _get_json(best)
    measures[dataname] = measures.get(dataname, {})
    measures[dataname][model] = measures[dataname].get(model, {})
    return measures

def save_measures(measures:dict, best:bool=False):
    from json import dumps

    with open(f"2.Development/2.Data/measures.{'best' if best else 'all'}.json", "w") as f:
        f.write(dumps(measures, indent=4) + "\n")
    json_to_csv(best=best)

def pop_model(dataname:str, model:str):
    
    measures = get_measures(dataname, model)
    measures[dataname].pop(model)

    if len(measures[dataname]) == 0:
        measures.pop(dataname)

    save_measures(measures, best=False)

def pop_measure(dataname:str, model:str, n_neighs:int):

    measures = get_measures(dataname, model)
    
    if str(n_neighs) not in measures[dataname][model]:
        warning(f"{model} on {dataname}({str(n_neighs)}) had no previsous measures")
        return
    
    for k, v in measures[dataname][model].items():
        print(f"k: {k}, v: {v}")
    measures[dataname][model].pop(str(n_neighs))
    for k, v in measures[dataname][model].items():
        print(f"k: {k}, v: {v}")

    if len(measures[dataname][model]) == 0:
        warning(f"{model} on {dataname} no longer has any measures")
        pop_model(dataname, model)
    
    save_measures(measures, best=False)
    update_best_measures()


def update_best_measures():
    measures = _get_json(best=False)
    best_measures = _get_json(best=True)

    for dataname_it, model_data in measures.items():
        measures[dataname_it] = measures.get(dataname_it, {})
        best_measures[dataname_it] = best_measures.get(dataname_it, {})
        for model_it, _ in model_data.items():
            measures[dataname_it][model_it] = measures[dataname_it].get(model_it, {})
            best_measures[dataname_it][model_it] = best_measures[dataname_it].get(model_it, {})

            model_measures = measures[dataname_it][model_it]

            k_neighs = [int(n_neighs) for n_neighs in model_measures.keys()]
            One_NN = [model_measures[n_neighs]['1-NN'] for n_neighs in model_measures.keys()]
            T = [model_measures[n_neighs]['T'] for n_neighs in model_measures.keys()]
            C = [model_measures[n_neighs]['C'] for n_neighs in model_measures.keys()]

            if len(k_neighs) == 0:
                warning(f"No measures found for {dataname_it} {model_it}")
                pop_model(dataname_it, model_it)
                if len(best_measures[dataname_it]) == 0:
                    best_measures.pop(dataname_it)
                continue

            temp_One_NN = [i if i != None else np.inf for i in One_NN]
            temp_T = [i if i != None else -np.inf for i in T]
            temp_C = [i if i != None else -np.inf for i in C]

            best_measures_k = [np.argmin(temp_One_NN), np.argmax(temp_T), np.argmax(temp_C)]
            best_measures[dataname_it][model_it] = {
                '#neighs': '"' + str([k_neighs[best_measures_k[0]], k_neighs[best_measures_k[1]], k_neighs[best_measures_k[2]]]) + '"',
                '#points': model_measures[str(k_neighs[best_measures_k[0]])]['#points'],
                '1-NN': One_NN[best_measures_k[0]],
                'T': T[best_measures_k[1]],
                'C': C[best_measures_k[2]],
            }
    save_measures(best_measures, best=True)


def add_measure(model_args:dict, dimensions:int, One_nn:float=None, T:float=None, C:float=None, time:float=None):
    dataname = model_args['dataname']
    model = model_args['model']
    n_neighs = model_args['#neighs']
    points = model_args['#points']
    
    data = {
        '#points': points,
        '#dimensions': dimensions,
        '1-NN': One_nn,
        'T': T,
        'C': C,
        'time': time,
    }
    if 'status' in model_args:
        data['status'] = model_args['status']
    if 'restricted' in model_args:
        data['restricted'] = model_args['restricted']
    if 'artificial_connected' in model_args:
        data['artificial_connected'] = model_args['artificial_connected']
    
    measures = get_measures(dataname, model)
    measures[dataname][model][str(n_neighs)] = data
    measures[dataname][model] = dict(sorted([(int(k), v) for k, v in measures[dataname][model].items()]))
    save_measures(measures, best=False)
    update_best_measures()





def json_to_csv(best:bool):
    measures = _get_json(best)
    
    with open(f"2.Development/2.Data/measures.{'best' if best else 'all'}.csv", "w") as f:
        if best:
            f.write(f"dataname;model;n_neighs;points;1-NN;T;C\n")
            for dataname, model_data in measures.items():
                for model, data in model_data.items():
                    if len(data) != 0:
                        f.write(f"{dataname};{model};{data['#neighs']};{data['#points']};{data['1-NN']};{data['T']};{data['C']}\n")
        else:
            f.write(f"dataname;model;n_neighs;points;1-NN;T;C;dimensions;status;restricted;artificial_connected;time\n")
            for dataname, model_data in measures.items():
                for model, n_neighs_data in model_data.items():
                    for n_neighs, data in n_neighs_data.items():
                        f.write(f"{dataname};{model};{n_neighs};{data['#points']};{data['1-NN']};{data['T']};{data['C']};{data.get('#dimensions', None)};{data.get('status', None)};{data.get('restricted', None)};{data.get('artificial_connected', None)};{data.get('time', None)}\n")




def plot_measures(dataname:str, model:str, preview:bool=False, **kwargs):
    import matplotlib.pyplot as plt
    import os

    # measures = get_measures(dataname, model, best=True)
    # k_best = measures[dataname][model].get('k_best', None)
    measures = get_measures(dataname, model)
    measures = measures[dataname][model]

    results = np.array([
        [int(k) for k in measures.keys()],
        [measures[k]['1-NN'] for k in measures.keys()],
        [measures[k]['T'] for k in measures.keys()],
        [measures[k]['C'] for k in measures.keys()]
    ])

    fig, ax1 = plt.subplots()
    # plt.title(f"{model} applied on {dataname}, selected k={k_best}")
    plt.title(f"{model} applied on {dataname}")

    ax1.set_xlabel('size of k-neighbourhood')
    ax1.set_ylabel('log10 of T and C')
    ax1.tick_params(axis='y')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.set_ylabel('1-NN')  # we already handled the x-label with ax1
    ax2.tick_params(axis='y')

    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useOffset=False, useMathText=True)

    
    ax1.plot(results[0], results[2], color='tab:red')
    ax1.plot(results[0], results[3], color='tab:orange')
    ax2.plot(results[0], results[1], color='tab:blue')
    fig.legend(["T", "C", "1-NN"])
    
    ax1.scatter(results[0], results[2], color='tab:red')
    ax1.scatter(results[0], results[3], color='tab:orange')
    ax2.scatter(results[0], results[1], color='tab:blue')

    ticks = [5,] + list(results[0]) + [15]
    ax1.set_xticks(ticks)
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped     
    
    os.makedirs("cache") if not os.path.exists("cache") else None
    os.makedirs(f"cache/measures/{model}") if not os.path.exists(f"cache/measures/{model}") else None
    plt.savefig(f"cache/measures/{model}/{dataname}.png")
    if preview:
        plt.show(block=False)
    plt.close()




    
############################################################
# DISTANCE MATRIX ##########################################
############################################################

def intra_dist_matrix(data: np.ndarray, k_neighbors: int = None):
    """
    Compute distance matrix efficiently for k-nearest neighbors.
    If k_neighbors is None, computes all pairwise distances (memory intensive).
    If k_neighbors is specified, only computes distances for k-nearest neighbors.
    """
    from sklearn.neighbors import NearestNeighbors

    n_samples = data.shape[0]
    
    if k_neighbors is None or k_neighbors >= n_samples:
        # Original behavior - compute all pairwise distances
        dist, indices = NearestNeighbors(n_neighbors=n_samples).fit(data).kneighbors(data)
    else:
        # Memory-efficient: only compute k-nearest neighbors
        dist, indices = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(data).kneighbors(data)
    
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        dist_matrix[i][indices[i]] = dist[i]
        # print(i, list(indices[i][:10]))
        # if i == 3:
        #     breakpoint()
    return dist_matrix



############################################################
# CACHE ####################################################
############################################################

def get_cached(model:str, dataname:str, n_neighbors:int, confirmation_X:np.ndarray, confirmation_NM:np.ndarray, id:str=None):
    import os

    path = f"cache/{model}/{dataname}/{n_neighbors}/"
    if id is not None:
        path += f"{id}/"
    if os.path.exists(path):
        X_cached:np.ndarray = np.load(path + "X.npy")
        NM_cached:np.ndarray = np.load(path + "NM.npy")
        if confirmation_X.shape == X_cached.shape and np.allclose(confirmation_X, X_cached) and np.allclose(confirmation_NM, NM_cached):
            return {
                'X': X_cached,
                'NM': NM_cached,
                'kernel': np.load(path + "kernel.npy"),
            }

def save_cached(model:str, dataname:str, n_neighbors:int, X:np.ndarray, NM:np.ndarray, kernel:np.ndarray, id:str=None):
    import os
    path = f"cache/{model}/{dataname}/{n_neighbors}/"
    if id is not None:
        path += f"{id}/"
    os.makedirs(path) if not os.path.exists(path) else None
    np.save(path + "X.npy", X)
    np.save(path + "NM.npy", NM)
    np.save(path + "kernel.npy", kernel)

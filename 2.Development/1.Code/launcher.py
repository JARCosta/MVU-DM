import argparse

import utils
utils.stamp = utils.Stamp()


from main import main
from datasets import get_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the model launcher.")
    parser.add_argument("--paper", type=str, default='dev', help="Paper to use for the model launcher.")
    parser.add_argument("--n_points", type=int, default=2000, help="Number of points to use for the model launcher.")
    parser.add_argument("--k_small", type=int, default=5, help="Number of neighbors to start with.")
    parser.add_argument("--k_large", type=int, default=15, help="Number of neighbors to end with.")
    parser.add_argument("--threaded", action='store_true', default=False, help="Use threading for the model launcher.")
    parser.add_argument("--preview", action='store_true', default=False, help="Plot the results.")
    parser.add_argument("--overlap", action='store_true', default=False, help="Plot the overlap.")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose output.")
    parser.add_argument("--measure", action='store_true', default=False, help="Store the measure's results.")
    parser.add_argument("--pause", action='store_true', default=False, help="Pause for every model execution.")
    parser.add_argument("--forget", action='store_true', default=False, help="Forget any cached data.")
    parser.add_argument("--seed", type=int, default=27, help="Random seed.")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level for the dataset.")
    parser.add_argument("--precision", type=float, default=1e-30, help="Precision for the optimisation problems.")
    parser.add_argument("--photos", action='store_true', default=False, help="paper photos.")


    args = parser.parse_args()

    if args.paper == "comparative":
        models = [
            "pca",
            "isomap",
            "lle",
            "le",
            "hlle",
            "ltsa",
            "kpca",

            "mvu",

            # Sub
            # "isomap.skl",
            # "lle.skl",
            # "le.skl",
            # "hlle.skl",
            # "ltsa.skl",
            
        ]
        dataset_list = [
            'swiss',
            'helix',
            'twinpeaks',
            'broken.swiss',
            'difficult',

            # 'mnist', # WARNING: memory overload, Fully connected
            'coil20',
            'orl',
            # 'nisis', # WARNING: dataset not found
            # 'hiva',
        ]

    elif args.paper == "eng":
        models = [
            "isomap",
            "isomap.eng",
            "lle",
            "lle.eng",
            "le",
            "le.eng",
            "hlle",
            "hlle.eng",
            # "jme",
            
            "mvu",

            # # Sub
            # "lle.skl",
            # "le.skl",
            # "hlle.skl",
        ]
        dataset_list = [
            'broken.swiss',
            'parallel.swiss',
            'broken.s_curve',
            'four.moons',
            'two.swiss',
            'coil20', 
            'mit-cbcl',
        ]

    elif args.paper == "dev":
        models = [
            "isomap",
            "isomap.eng",
            "lle",
            "hlle",
            "le",
            "ltsa",
            "kpca",
            "tsne",
            "mvu",
            "mvu.eng",
            "mvu.based",            
        ]
        dataset_list = [



            # 'swiss', # Fully connected
            # 'helix', # Fully connected
            # 'twinpeaks', # Fully connected
            # 'broken.swiss', # broken.s_curve make this redundant
            # 'difficult', # Fully connected (n_points=1000, k >= 12) and (n_points=2000, k >= 6)
            
            'broken.s_curve',
            'parallel.swiss',
            'two.swiss',
            'four.moons', # too easy to get a good 1-NN, mvu is perfect in T and C

            # 'teapots', # Fully connected
            # 'mnist', # WARNING: memory overload, Fully connected
            'coil20', # n_points=2000, never connected
            'orl',
            # 'nisis', # Does not exist
            # 'hiva', # n_points=2000, Fully connected
            'mit-cbcl',
            'olivetti',
            
            # "cifar10", # Fully connected
            # "cancer",
            # "animalface",
            # "carnivores", # Fully connected
            # "flowers", # Fully connected
            # "imagenet",
            # "inat",

            ]

    elif args.paper == "none":
        dataset_list = [
            'broken.swiss',
            'parallel.swiss',
            'broken.s_curve',
            'four.moons',
            'two.swiss',
            'coil20', 
            'mit-cbcl',
        ]
        for dataname in dataset_list:
            X, labels, t = get_dataset(dataname, args.n_points, args.noise, random_state=args.seed, cached=not args.forget)
            None_1_NN = utils.one_NN(X, labels)
            print(f"loaded {dataname} {None_1_NN}")
            utils.add_measure({'dataname': dataname, 'model': 'none', '#neighs': 1, '#points': X.shape[0]}, None_1_NN)
        exit()

    else:
        raise ValueError("Paper preset not found.")
    
    try:
        main(
            paper=args.paper,
            model_list=models,
            dataset_list=dataset_list,
            n_points=args.n_points,
            k_small=args.k_small,
            k_large=args.k_large,
            threaded=args.threaded,
            preview=args.preview if not args.threaded else False,
            overlap=args.overlap if not args.threaded else False,
            verbose=args.verbose if not args.threaded else False,
            measure=args.measure,
            pause=args.pause if not args.threaded else False,
            seed=args.seed,
            noise=args.noise,
            precision=args.precision,
            cached=not args.forget,
            photos=args.photos,
        )
    finally:
        pass
        # utils.stamp.print("* Killing process")
        # from models.mvu import eng
        # if eng is not None:
        #     eng.quit()


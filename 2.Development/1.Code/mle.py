import skdim

from datasets import datasets, get_dataset
import utils

utils.stamp = utils.Stamp()


for dataname in datasets:
    if dataname not in ["mnist", "coil20", "hiva", "olivetti", "cifar10", "animalface", "flowers", "carnivores", "cancer", "imagenet", "inat"]:

        X, labels, _ = get_dataset(dataname, n_points=2000, noise=0.0, random_state=42, cached=False)
        mle_estimator = skdim.id.MLE()
        d_estimated = mle_estimator.fit(X).dimension_
        d_estimated = max(1, min(int(round(d_estimated)), X.shape[1]))

        print(f"{dataname}: {d_estimated:.2f}")

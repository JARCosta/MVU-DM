
# Maximum Variance Unfolding on Disjoint Manifolds

This is the development repository for the study on "Maximum Variance Unfolding on Disjoint Manifolds" by João Costa, advised by the professor Francisco Melo and Co-advised by João Caldeira.

### Abstract
This study investigates the problem of Maximum Variance Unfolding (MVU) on disjoint manifolds, aiming to improve the preservation of local structures in high-dimensional data when mapped to lower dimensions. We propose a novel approach that leverages the geometric properties of disjoint manifolds to enhance the performance of MVU. Our method is evaluated on various synthetic and real-world datasets, demonstrating its effectiveness in capturing the underlying data distribution while maintaining computational efficiency.

## Structure of the Repository
**Literature**
- [**Papers**](1.Literature/1.Papers/): Some of the research papers referenced in this work.
- [**PIC**](1.Literature/2.PIC/): The proposal for the thesis.
- [**ICLR2026**](1.Literature/3.ICLR2026/): The submission to ICLR 2026.
- [**Dissertation**](1.Literature/4.Dissertation/): The final dissertation documents.

**Development**
- [**Code**](2.Development/1.Code/): Implementation of the algorithm, datasets and utility functions.
- [**Data**](2.Development/2.Data/): Metric values collected in the experiments.
- [**Jobs**](2.Development/3.Jobs/): Scripts for running experiments and evaluations.

## Installation
This project was developed and tested using Python 3.10. Start by cloning the repository:
```bash
git clone https://github.com/JARCosta/52_Dissertation
cd 52_Dissertation
```
Then, create a virtual environment and install the required dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```
## Execution Instructions
The code is implemented in Python and uses libraries such as NumPy, SciPy, and Matplotlib. The [requirements file](requirements.txt) lists all the necessary dependencies.

To execute the code, run python on the [launcher file](2.Development/1.Code/launcher.py). It takes the following arguments:
- `--paper {str}`: Presets of datasets and models to run, according to different papers: ['comparative'](1.Literature/4.Dissertation/tex/Bibliography/comparison.pdf), ['eng'](1.Literature/4.Dissertation/tex/Bibliography/eng.pdf), ['dev'](1.Literature/4.Dissertation/tex/out/main.pdf).
- `--n_points {int}`: Number of points to be generated for synthetic datasets (default: 2000);
- `--k_small {int}`: Minimum number of neighbors to consider (default: 5);
- `--k_large {int}`: Maximum number of neighbors to consider (default: 15);
- `--noise {float}`: Standard deviation of Gaussian noise to be added to the datasets (default: 0.05);
- `--seed {int}`: Random seed for reproducibility (default: 42);
- `--forget`: If set, the cached datasets and model results will be ignored and recomputed.
- `--verbose`: If set, the program will print detailed logs of its execution.
- `--preview`: If set, the program will display figures during execution.

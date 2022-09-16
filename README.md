# Bayesian Optimization System for Chemical Synthesis

This repository contains a web application for performing  Bayesian optimization on a parameter space that includes both numerical and categorical variables.

In short, molecular SMILES used to represent categorical variables such as ligands, bases etc and are encoded into a vector according to their molecular properties. Then a Bayesian step is performed in this latent space and the vector of values is translated back into SMILES by calculating its minimum distance fromt he other SMILES of the library.

With this simple way of performing the Bayesian step you can see on the PowerPoint presentation that the reaction manages to find the maximum yield possible in a simple scenario with a Ligand (categorical), a Temperature (numeric) and a Consentration (numeric) within 30 experiments. The Ligand library consist of 12 ligands, temperature can range from 90 to 120 degrees and concentation ranges from 0.057 to 0.153.

The data for measuring the performance of the algorithm come from the [Shilds paper data](https://www.nature.com/articles/s41586-021-03213-y). This dataset also consideres bases and solvents which have been removed in order to create a minimal code example that is easy to extend.

## Requirements:
An [Anaconda python environment](https://www.anaconda.com/download) is recommend.
Check the environment.yml file, but primarily:
- Python >= 3.9
- Streamlit = 1.11.0
- Numpy = 1.21.5
- Pandas = 1.3.4
- Sklearn = 1.1.1
- Matplotlib = 3.4.3
- Scipy = 1.7.3

Jupyter notebook is required to run the ipynb examples.

 ## Suggested way to navigate the repository

 After installing the requirements, you should navigate to the [Notebooks](https://sourcecode.jnj.com/projects/ASX-AHWJ/repos/bo-app/browse/Notebooks) folder and see the ipynb in the given order. This will walk you through
 1. Gaussian Process regression, the working engine of Bayesian Optimization
 2. Bayesian optimization from scratch for the one-input one-output case
 3. Generalization of the above for multiple inputs
 4. Bayesian optimization using the latest framework used in research `botorch`
 5. Generalization of the above for a multivariate scenario
 6. How descriptors are used and calculated

After that, you can see a small demo in video format in [this](https://sourcecode.jnj.com/projects/ASX-AHWJ/repos/bo-app/browse/app_screenshots) folder. All the data for the demo are available on the [demo](https://sourcecode.jnj.com/projects/ASX-AHWJ/repos/bo-app/browse/app/demo) folder.

Lastly, open a command prompt and navigate to the `app_sklearn` folder [here](https://sourcecode.jnj.com/projects/ASX-AHWJ/repos/bo-app/browse/app/app_sklearn). By running the command `streamlit run app_sklearn.py`, you should be able to launch the application on your machine and use it according to the demos.

## Author

This software is written by Ted Ladas.

* [Ted Ladas JnJ](tladas@its.jnj.com)
* [Ted Ladas Academic](k20103629@kcl.ac.uk)

Feel free to reach out for with any questions!

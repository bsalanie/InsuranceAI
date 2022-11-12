# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_utils.ipynb.

# %% auto 0
__all__ = ['DATA_PATH', 'GENERAL_RESULTS_FOLDER', 'PDF_FOLDER', 'DASH', 'PARAMETERS', 'VARIABLE_RHO_NOTEBOOK_PATH_NO_EXTENSION',
           'CONSTANT_RHO_NOTEBOOK_PATH_NO_EXTENSION', 'ACCUMULATED_CONSTANT_RO_DATA_PATH',
           'PREDICTING_FHAT_ACCURACIES_DATA_PATH', 'PREDICTING_FHAT_NOTEBOOK_HTML_PATH',
           'PREDICTING_FHAT_SEE_RESULTS_NO_EXTENSION', 'ACCUMULATED_DATA_SEE_RESULTS_NO_EXTENSION', 'read_data',
           'select_variables']

# %% ../nbs/00_utils.ipynb 3
from sys import platform
from typing import Tuple
from pathlib import Path
import pandas as pd

# %% ../nbs/00_utils.ipynb 4
DATA_PATH = Path("..") / Path("Datasets")
if platform in ["linux", "linux2"]:
    DATA_PATH = Path("../Datasets")

GENERAL_RESULTS_FOLDER = Path("Results")
PDF_FOLDER = Path("pdfs")
DASH = "_"

PARAMETERS = "parameters"
VARIABLE_RHO_NOTEBOOK_PATH_NO_EXTENSION = "variable_rho"
CONSTANT_RHO_NOTEBOOK_PATH_NO_EXTENSION = "constant_rho"
ACCUMULATED_CONSTANT_RO_DATA_PATH = "accumulated_constant_ro_data.npy"
PREDICTING_FHAT_ACCURACIES_DATA_PATH = "predicting_fhat_accuracies_data.npy"
PREDICTING_FHAT_NOTEBOOK_HTML_PATH = "predicting_fhat_notebook.html"
PREDICTING_FHAT_SEE_RESULTS_NO_EXTENSION = "predicting_fhat_see_results"
ACCUMULATED_DATA_SEE_RESULTS_NO_EXTENSION = "accumulated_data_see_results"

# %% ../nbs/00_utils.ipynb 6
def read_data(dataset: str # "j88" or "jpropre"
             ) -> pd.DataFrame: # a clean data frame
    """
    Reads the dataset selected and returns a clean Pandas dataframe
    """
    data_file = f"{dataset}.txt"
    names_vars = [
        'Age below 25', 'Age', 'Age of driver', 'German car', 'American car', 'Age of License',
                  'Type of deductible', 'Theft deductible', 'Compulsory coverage', 'Comprehensive coverage',
                  'Responsibility', 'Citroën', 'Driver is insuree', 'No accident', 
                    'Responsibility code', 'Company',
                  'Date car', 'Department registered', 'Year birth insuree', 'Claim processing',
                  'Other french car', 'Group', 'Guarantee in claim', 'Male', 'Not at fault', 
                  'Italian car', 'Number cars in claim', 'Department claim', 'Deductible damages',
                  'Deductible theft', 'Reimbursement', 'Settlement', 'Total cost', 'Compulsory cost', 'Nature claim', 
                  'Number claims at fault', 'Identifier 1', 'Identifier 2', 'Basic premium',
                  'Peugeot', 'Fiscal HP', 'Total premium', 'Duration', 'Occupation', 'Number claims',
                  'Compulsory premium', 'Region', 'Renault', 'Class car', 'Gender', 'Real group',
                  'Bonus Malus', 'Age category car', 'Age category insuree', 'Basic premium category', 
                  'Shared responsibility', 'Car use', 'Zone'
                 ]
    data = pd.read_csv(DATA_PATH / data_file, delimiter = ' ', header=None)
    data.columns = names_vars
    # we change the types of the non-categorical variables
    for float_col in ['Deductible damages', 'Deductible theft', 'Reimbursement', 'Total cost', 'Compulsory cost',
                     'Total premium', 'Bonus Malus']:
        data[float_col]=data[float_col].astype(float)
    return data

# %% ../nbs/00_utils.ipynb 7
def select_variables(data: pd.DataFrame # the Pandas data frame
                    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]: # `y1, y2, w, X`
    """ returns the `y` variables, the weights, and the `X` covariates"""
    y1 = 1*(data['Comprehensive coverage'] > 0)
    y2 = 1*(data['Not at fault'] == 0)
    w = data['Duration']

    X = data[['Group', 'Male', 'Occupation', 'Region', 'Renault', 'Age category car',
           'Age category insuree', 'Car use', 'Zone']].astype('category')

    return y1, y2, w, X

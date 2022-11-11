from sys import platform

if platform in ["linux", "linux2", "darwin"]:
    SLASH = "/"
elif platform in ["win32"]:
    SLASH = r'"\"'
else:
    raise Exception("?")

DATA_PATH = "data/j88.txt"

GENERAL_RESULTS_FOLDER = "results" + SLASH
PDF_FOLDER = "pdfs" + SLASH
DASHES = "_"

PARAMETERS = "parameters"
VARIABLE_RHO_NOTEBOOK_PATH_NO_EXTENSION = "variable_rho"
CONSTANT_RHO_NOTEBOOK_PATH_NO_EXTENSION = "constant_rho"
ACCUMULATED_CONSTANT_RO_DATA_PATH = "accumulated_constant_ro_data.npy"
PREDICTING_FHAT_ACCURACIES_DATA_PATH = "predicting_fhat_accuracies_data.npy"
PREDICTING_FHAT_NOTEBOOK_HTML_PATH = "predicting_fhat_notebook.html"
PREDICTING_FHAT_SEE_RESULTS_NO_EXTENSION = "predicting_fhat_see_results"
ACCUMULATED_DATA_SEE_RESULTS_NO_EXTENSION = "accumulated_data_see_results"
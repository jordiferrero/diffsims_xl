#%%
from ml_difsims.predictors.evaluation_ops import *

root = r'G:\My Drive\PhD\projects\external_measurements\ml_difsims'

pred_folder = r'data\predictions\test'
data_folder = r'data\experimental\npz_files'
cif_files_folder = r'models\crystal_phases'
prediction_list_to_evaluate = glob.glob(os.path.join(root, pred_folder, '*_probability_pred.npz'))
prediction_list_to_evaluate.sort()

print(prediction_list_to_evaluate)


def evaluate_all(prediction_list_to_evaluate, data_folder, cif_files_folder, root_path):
    """
    Input files needed:
    - `*_probabilty_pred.npz` numpy array containing the predictions for each experimental file as a shape of (y, x, prob_n_phases) saved as 'pred' key AND the predicted labels as 'phases' key.
    - Link to the original `*_radial_crop.npz` files containing the 1D cropped diffraction patterns and the original q_axis range.
    - Link to the `.cif` files to use to simulate diffraction patterns from
    """

    evaluations = {}
    # Loop through all the files
    for f_pred in prediction_list_to_evaluate:
        evaluation = evaluate_data(f_pred, data_folder, cif_files_folder, root_path)
        evaluations[f_pred] = evaluation

    return evaluations


if __name__ == "__main__":
    evaluate_all(prediction_list_to_evaluate, data_folder, cif_files_folder, root)
    #result = evaluate_all.execute_in_process()

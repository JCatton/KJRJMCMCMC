from tqdm import tqdm
import os


def check_and_create_folder(save_loc):
    """
    Helper function to check if a folder exists. If not, create it.
    Parameters:
    - saveloc: The desired path for saving data.
    """
    # Check if the base folder "Outputs" exists
    if not os.path.exists("Outputs"):
        os.makedirs("Outputs")

    # Check if the specific saveloc exists
    full_saveloc = os.path.join("Outputs", save_loc)
    if not os.path.exists(full_saveloc):
        os.makedirs(full_saveloc)

    return full_saveloc

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from lightkurve.correctors import DesignMatrix, RegressionCorrector
from lightkurve import LightCurveCollection
from tqdm import tqdm


def download_data(
    target_name: str,
    exptime: int = 120,
    mission: str = "Tess",
    sector: int = None,
    author=None,
    cadence=None,
    indicies_requested=None,
    max_number_downloads: int = 20,
    apply_regressor_bool = False,
    pipeline_aper_bool = True,
    use_lightcurve_direct = False,
) -> tuple:
    """
    Downloads data from the target_name

    Parameters:
    - target_name: String representing the target name

    Returns:
    - times: Array of time values
    - flux: Array of flux values
    """
    # Create a dictionary of parameters
    search_params = {
        "mission": mission,
        "sector": sector,
        "exptime": exptime,
        "author": author,
        "cadence": cadence,
    }

    # Filter out parameters with None values
    search_params = {
        key: value for key, value in search_params.items() if value is not None
    }
    # search_results = lk.search_tesscut(target_name)

    #If want to use lightcurve directly
    if use_lightcurve_direct:
        # Search for the light curve
        search_results = lk.search_lightcurve(target_name, mission = mission, author = author, exptime=exptime)
        print(search_results)
        # Download the light curve
        if indicies_requested == None:
            corr = search_results.download_all()
        else:
            lower = indicies_requested[0]
            upper = indicies_requested[1]
            corr = search_results[lower:upper].download_all()
        # Remove outliers and nans
        corr = corr.stitch().remove_outliers().remove_nans()

    #If want to use targetpixelfile
    else:
        search_results = lk.search_targetpixelfile(target = target_name, mission = mission, author = author, exptime=exptime)   
        print(search_results)
        if indicies_requested == None:
            tpf_collection = search_results.download_all()
        else:
            lower = indicies_requested[0]
            upper = indicies_requested[1]
            tpf_collection = search_results[lower:upper].download_all()

        corr = tpfs_to_lightcurves(tpf_collection, apply_regressor_bool=apply_regressor_bool, pipeline_aper_bool = pipeline_aper_bool)

    # Filter the arrays
    combined_array_corr = np.array([corr.time.value, corr.flux])
    sorted_indices_corr = np.argsort(combined_array_corr[0])
    sorted_combined_array_corr = combined_array_corr[:, sorted_indices_corr]
    # Zero
    sorted_combined_array_corr[0] -= sorted_combined_array_corr[0, 0]

    # Filter the arrays
    combined_array_un_corr = np.array([corr.time.value, corr.flux])
    sorted_indices_un_corr = np.argsort(combined_array_un_corr[0])
    sorted_combined_array_un_corr = combined_array_un_corr[:, sorted_indices_un_corr]
    # Zero
    sorted_combined_array_un_corr[0] -= sorted_combined_array_un_corr[0, 0]

    plt.plot(
        sorted_combined_array_un_corr[0],
        sorted_combined_array_un_corr[1],
        label="uncorrected",
    )
    plt.plot(
        sorted_combined_array_corr[0], sorted_combined_array_corr[1], label="Corrected"
    )
    plt.plot()
    return sorted_combined_array_corr[0], sorted_combined_array_corr[1]


def apply_regressor(tpf, aper):
    """
    Apply the regressor to the data

    Parameters:
    - tpf: Target Pixel File

    Returns:
    - uncorrected_lc: Uncorrected light curve
    - corrected_ffi_lc: Corrected light
    """

    lc_raw = tpf.to_lightcurve(aperture_mask=aper)
    uncorrected_lc = lc_raw.remove_nans().remove_outliers()

    # Create a time mask: find which TPF timestamps are present in the cleaned light curve
    time_mask = np.in1d(tpf.time.value, uncorrected_lc.time.value)
    # Apply the time mask to tpf.flux to get the corresponding flux values
    clean_flux = tpf.flux[time_mask, :, :]

    # Build the design matrix using the pixels outside the aperture
    dm = DesignMatrix(clean_flux[:, ~aper], name='regressors').pca(5).append_constant()

    # Initialize the regression corrector and correct the light curve
    rc = RegressionCorrector(uncorrected_lc)
    corrected_ffi_lc = rc.correct(dm)

    corrected_ffi_lc = uncorrected_lc - rc.model_lc + np.percentile(rc.model_lc.flux, 5)

    return uncorrected_lc.normalize(), corrected_ffi_lc.normalize()


def tpfs_to_lightcurves(tpfs, apply_regressor_bool = False, pipeline_aper_bool = False):
    """
    takes lightkurve collection of tpfs and returns a lightkurve collection of light curves, use apply_regressor to apply the regressor to the data
    """
    un_corr = []
    corr = []
    for i, tpf in tqdm(enumerate(tpfs), desc="Processing Light Curves"):
        if pipeline_aper_bool:
            aperture_mask = tpf.pipeline_mask
        else:
            aperture_mask = tpf.create_threshold_mask()

        if apply_regressor_bool:
            uncorrected_lc, corrected_lc = apply_regressor(tpf, aperture_mask)
            un_corr.append(uncorrected_lc)
            corr.append(corrected_lc)
        else:
            uncorrected_lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
            un_corr.append(uncorrected_lc)
            # ax = uncorrected_lc.normalize().plot(label=f"Uncorrected lc for {i}")
            corrected_lc = uncorrected_lc.remove_outliers().remove_nans().normalize()
            # corrected_lc.plot(ax=ax, label=f"Corrected lc for {i}", ls = "--")
            corr.append(corrected_lc)
            # plt.show()

    corr_lc_collection = LightCurveCollection(corr)
    un_corr_lc_old_collection = LightCurveCollection(un_corr)

    stiched_corr_lc = corr_lc_collection.stitch()
    stiched_un_corr_lc = un_corr_lc_old_collection.stitch()

    return stiched_corr_lc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    name = "TIC 147977348"
    mission = None
    exptime = None
    author = "Kepler"
    cadence = "long"
    sector = None
    max_number_downloads = 10

    time, flux = download_data(
        name,
        exptime=exptime,
        mission=mission,
        sector=sector,
        author=author,
        cadence=cadence,
        max_number_downloads=max_number_downloads,
    )

    print(time, flux)
    plt.plot(time, flux)
    plt.show()

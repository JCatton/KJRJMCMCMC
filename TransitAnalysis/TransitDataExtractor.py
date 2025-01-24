import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

def download_data(target_name: str, exptime:int = 120, mission:str = "Tess", sector:int = None, author = None, cadence = None, max_number_downloads:int = 20) -> tuple:
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
    search_params = {key: value for key, value in search_params.items() if value is not None}
    search_results = lk.search_tesscut(target_name)
    # print(f"Searching for data with metadata \n{"\n".join([f"{k:=^9}: {v:<20}" for k,v in search_params.items()])}")
    # search_results = lk.search_lightcurve(target_name, **search_params)

    print(search_results)
    tpf_collection = search_results.download_all(cutout_size=(50, 50))

    un_corr, corr = tpfs_to_lightcurves(tpf_collection)

    times = corr.time - un_corr.time[0]
    flux = corr.flux

    ax = corr.plot(c='k', lw=2, label='Corrected')
    un_corr.plot(ax = ax,c='r', lw=2, label='Uncorrected')
    plt.show()
#
    return times, flux

def apply_regressor(tpf):
    """
    Apply the regressor to the data

    Parameters:
    - tpf: Target Pixel File

    Returns:
    - uncorrected_lc: Uncorrected light curve
    - corrected_ffi_lc: Corrected light
    """
    aper = tpf.create_threshold_mask()
    uncorrected_lc = tpf.to_lightcurve(aperture_mask=aper)

    uncorrected_lc = uncorrected_lc.remove_nans()

    clean_flux = tpf.flux[~np.isnan(tpf.to_lightcurve(aperture_mask=aper).flux), :, :]

    dm = DesignMatrix(clean_flux[:, ~aper], name='regressors').pca(5).append_constant()

    rc = RegressionCorrector(uncorrected_lc)

    corrected_ffi_lc = rc.correct(dm)

    corrected_ffi_lc = uncorrected_lc - rc.model_lc + np.percentile(rc.model_lc.flux, 5)

    return uncorrected_lc.normalize(), corrected_ffi_lc.normalize()

def tpfs_to_lightcurves(tpfs):
    """
    takes lightkurve collection of tpfs and returns a lightkurve collection of light curves, use apply_regressor to apply the regressor to the data
    """
    un_corr = []
    corr = []
    for tpf in tpfs:
        uncorrected_lc, corrected_lc = apply_regressor(tpf)
        un_corr.append(uncorrected_lc)
        corr.append(corrected_lc)

    corr_lc_collection = LightCurveCollection(corr)
    un_corr_lc_old_collection = LightCurveCollection(un_corr)

    stiched_corr_lc = corr_lc_collection.stitch()
    stiched_un_corr_lc = un_corr_lc_old_collection.stitch()

    return stiched_un_corr_lc, stiched_corr_lc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    name = 'Kepler-8'
    mission=None
    exptime=None
    author = 'Kepler'
    cadence = 'long'
    sector = None
    max_number_downloads = 10


    time, flux = download_data(name, exptime=exptime, mission=mission, sector=sector, author=author, cadence=cadence, max_number_downloads=max_number_downloads)

    print(time, flux)
    plt.plot(time, flux)
    plt.show()
import lightkurve as lk
import matplotlib.pyplot as plt


def download_data(target_name: str, exptime:int = 120, mission:str = "Tess", sector:int = None, author = None, cacence = None, max_number_downloads:int = 20) -> tuple:
    """
    Downloads data from the target_name

    Parameters:
    - target_name: String representing the target name

    Returns:
    - times: Array of time values
    - flux: Array of flux values
    """
    search_results = lk.search_lightcurve(target_name, mission=mission, sector=sector, exptime=exptime, author=author)

    light_curve_collection = search_results[:max_number_downloads].download_all()

    # Flattening ensures that the data is all normalised to the same level and applies savitzky-golay smoothing filter
    lc = light_curve_collection.stitch().remove_outliers().remove_nans().flatten()

    times = lc.time - lc.time[0]  # zero our times as this is in line with current simulations

    flux = lc.flux

    return times, flux


if __name__ == "__main__":
    name = 'HD 191939'
    mission='TESS'
    exptime=120
    author = 'SPOC'

    time, flux = download_data(name, exptime, mission, author = author, max_number_downloads=10)

    print(time, flux)
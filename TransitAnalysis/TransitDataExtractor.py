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

    print(f"Searching for data with metadata \n{"\n".join([f"{k:=^9}: {v:<20}" for k,v in search_params.items()])}")
    search_results = lk.search_lightcurve(target_name, **search_params)

    print(search_results)
    light_curve_collection = search_results[:max_number_downloads].download_all()

    # Flattening ensures that the data is all normalised to the same level and applies savitzky-golay smoothing filter
    lc = light_curve_collection.stitch().remove_nans().flatten()

    times = lc.time.value - np.min(lc.time.value)  # zero our times as this is in line with current simulations

    flux = lc.flux.value
#
    return times, flux


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
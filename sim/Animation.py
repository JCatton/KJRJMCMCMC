import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import FileCheck as fc
import os


def animate_orbits(x_pos, y_pos, x_orbit, y_orbit, times, planet_params, flux, saveloc):
    """
    Parameters:
    x_pos, y_pos: 2D arrays of x and y positions for the star and planets over time
    x_orbit, y_orbit: 2D arrays of unperturbed x and y positions for the planets over time
    times: 1D array of time values
    planet_params: List of planet parameters [radius, mass, orbital radius, eccentricity, omega (phase)]
    flux: 1D array of flux values
    """
    N = len(planet_params)

    def update(frame):
        star.set_offsets([x_pos[0, frame], y_pos[0, frame]])  # Update star position
        for i, planet in enumerate(planets):
            planet.set_offsets(
                [x_pos[i + 1, frame], y_pos[i + 1, frame]]
            )  # Update planet positions
        flux_line.set_data(times[:frame], flux[:frame])  # Update flux line
        return [star] + planets + [flux_line]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x displacement (AU)")
    ax1.set_ylabel("y displacement (AU)")

    for i in range(N):
        ax1.plot(x_orbit[i], y_orbit[i], "--", label=f"Planet {i+1} Orbit")

    star = ax1.scatter([], [], color="black", label="Star")
    planets = [
        ax1.scatter(
            [],
            [],
            label=f"Planet {i+1}, radius = {round(planet_params[i][0]/4.2635e-5, 2)} $R_{{\oplus}}$",
        )
        for i in range(N)
    ]

    ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(
        min(flux) * 0.99999, 1.00001
    )  # Extend y axis very slightly to avoid small plot
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Relative Brightness")
    ax2.set_title("Light Curve")

    (flux_line,) = ax2.plot([], [], "b-", label="Flux")

    total_frames = len(times)
    target_frames = 900  # 900 ensures mp4 file (if created with ffmpeg) is 15 seconds long at 60 fps
    frame_skip = max(total_frames // target_frames, 1)
    sampled_frames = range(0, total_frames, frame_skip)

    ani = animation.FuncAnimation(
        fig, update, frames=sampled_frames, interval=1000 / 60, blit=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the animation using 'pillow' to a GIF
    full_saveloc = fc.check_and_create_folder(saveloc)
    gif_path = os.path.join(full_saveloc, "N_body_animation.gif")
    ani.save(gif_path, writer="pillow", fps=60)

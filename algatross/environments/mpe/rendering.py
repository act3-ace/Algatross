"""Common rendering helpers for MPE."""

import io

from collections.abc import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image


def mplfig_to_npimage(figure: mpl.figure.Figure) -> np.ndarray:
    """
    Convert a matplotlib figure to an array image.

    Parameters
    ----------
    figure : mpl.figure.Figure
        The figure to convert.

    Returns
    -------
    np.ndarray
        The figure as an array.
    """
    buf = io.BytesIO()
    figure.savefig(buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.asarray(img)


def get_frame_functional(fps: int, height: int, width: int, **kwargs: dict) -> Callable:
    """Return a function that renderes a new mpl frame.

    Parameters
    ----------
    fps : int
        Frames per second
    height : int
        Height of the frame
    width : int
        Width of the frame
    `**kwargs`
        The locals() from a calling function. Check usage of kwargs below.

    Returns
    -------
    Callable
        Function definition for drawing a new frame.
    """
    frames = kwargs["frames"]
    # rewards = kwargs["rewards"]  # noqa: ERA001

    x = 1
    y = 1
    fig, axes = plt.subplots(y, x, figsize=(width * 2, height * 2))

    def render_frame(t: float) -> np.ndarray:
        """Generate the frame from moviepy definition of a timestep (t).

        Parameters
        ----------
        t : float
            The current time according to moviepy. Check calculate of frame_idx.

        Returns
        -------
        np.ndarray
            Rendered frame for moviepy.
        """
        frame_idx = int(t * fps)

        axes_obj = axes
        axes_obj.clear()
        axes_obj.imshow(frames[min(frame_idx, len(frames) - 1)])
        axes_obj.axis("off")
        axes_obj.set_title(f"Frame: {frame_idx + 1}", fontdict={"fontsize": 24})
        plt.tight_layout()
        # axes[1, 0].clear()  # noqa: ERA001

        # axes[1, 1].clear()  # noqa: ERA001

        # axes[0, 1].clear()  # noqa: ERA001
        # axes[0, 1].plot(rewards, c="orange")  # noqa: ERA001
        # axes[0, 1].set_title(f"Accumulative Reward: {rewards[frame_idx]:.3f}")  # noqa: ERA001
        # axes[0, 1].axvline(frame_idx, c="red")  # noqa: ERA001

        return mplfig_to_npimage(fig)

    return render_frame

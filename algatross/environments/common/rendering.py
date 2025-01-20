"""Environment rendering functions."""

import copy
import logging
import traceback

from collections.abc import Callable
from pathlib import Path

import numpy as np

from PIL.Image import Image as ImageType
from moviepy import VideoClip
from moviepy.config import FFMPEG_BINARY


def pix2inch(num_pixels: float) -> int:
    """Convert pixels to desired resolution.

    Parameters
    ----------
    num_pixels : float
        Number of pixels

    Returns
    -------
    int
        The output resolution in inches
    """
    return int(np.ceil(num_pixels / 100))


def make_movies(
    episodes: list[np.ndarray],
    save_dir: str | Path,
    get_frame_functional: Callable,
    fps: int = 20,
    duration_s: float = 0,
) -> list[str]:
    """Save animation to disk.

    Parameters
    ----------
    episodes : list[np.ndarray]
        List of episodes.
    save_dir : str | Path
        Directory to save movies at.
    get_frame_functional : Callable
        Function to call to retrieve frames
    fps : int, optional
        Frames per second of the animation. Defaults to 20.
    duration_s : float, optional
        Desired duration of the animation. Defaults to 0.

    Returns
    -------
    list[str]
        The list of output paths.

    Raises
    ------
    TypeError
        If there is a version mismatch between FFMPEG and MoviePy
    """
    logger = logging.getLogger("ray")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    files_written = []

    for episode_idx in range(len(episodes)):
        episode_duration: float = copy.deepcopy(duration_s)

        frames = episodes[episode_idx]
        if isinstance(frames[0], ImageType):
            frames = [np.asarray(frame) for frame in frames]  # type: ignore[assignment]

        if not episode_duration:
            episode_duration = len(frames) / fps

        if len(frames) == 0:
            logger.warning(f"Episode idx={episode_idx} frames was empty, make sure runner is saving frames!")
            continue

        height, width = pix2inch(frames[0].shape[0]), pix2inch(frames[0].shape[1])

        frame_functional = get_frame_functional(fps=fps, height=height, width=width, frames=frames)

        video_file = (save_dir / f"{episode_idx:0>5d}.mp4").as_posix()

        # creating animation
        with VideoClip(frame_functional, duration=episode_duration) as animation:
            try:
                animation.write_videofile(video_file, fps=fps, audio=False)
            except TypeError as err:
                # https://stackoverflow.com/questions/68032884/getting-typeerror-must-be-real-number-not-nonetype-whenever-trying-to-run-wr
                traceback.format_exc()
                msg = (
                    "Potential version mis-match with FFMPEG and MoviePy, try upgrading one or the other. "
                    "You may try installing FFMPEG system-wide and exporting FFMPEG_BINARY env variable with the `ffmpeg' path. "
                    "Then re-run this program. The current path is " + FFMPEG_BINARY
                )
                raise TypeError(msg) from err

        files_written.append(video_file)

    return files_written

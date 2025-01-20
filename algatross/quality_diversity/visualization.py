"""Provides grid_archive_heatmap."""

import io

from operator import sub
from typing import Literal

import matplotlib as mpl
import matplotlib.collections as mc
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable

import numpy as np

import torch

from torchvision.transforms import ToTensor

import PIL
import PIL.Image
import dask.array as da
import ribs

from ribs.visualize._utils import retrieve_cmap, set_cbar, validate_df  # noqa: PLC2701

from algatross.quality_diversity.archives.unstructured import UnstructuredArchive


def get_aspect(ax: mpl.axes.Axes) -> float:
    """
    Return the aspect ratio for the axes.

    Since `ax.get_aspect()` sometimes returns a string we have to caclulate the
    ratio by hand

    Parameters
    ----------
    ax : mpl.axes.Axes
        The axis object to analyze

    Returns
    -------
    float
        The aspect ratio of the axis
    """
    # Total figure size
    fig_w, fig_h = ax.get_figure().get_size_inches()  # type: ignore[union-attr]
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (fig_h * h) / (fig_w * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio


def nearest_neighbor_archive_heatmap(  # noqa: PLR0912, PLR0914, PLR0915
    archive: UnstructuredArchive,
    ax: mpl.axes.Axes | None = None,
    *,
    df: ribs.archives.ArchiveDataFrame = None,
    transpose_measures: bool = False,
    cmap: str | mpl.colors.Colormap = "magma",
    aspect: Literal["auto", "equal"] | float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: Literal["auto"] | mpl.axes.Axes | None = "auto",
    cbar_kwargs: dict | None = None,
    rasterized: bool = False,
    plot_strategy: Literal["patches", "scatter"] = "patches",
    plot_kwargs: dict | None = None,
) -> torch.Tensor:
    r"""Plot a heatmap of a :class:`~algatross.quality_diversity.archives.unstructured.UnstructuredArchive` with 1D or 2D measure space.

    This function creates a grid of cells and shades each cell with a color
    corresponding to the objective value of that cell's elite. This function
    uses :func:`~matplotlib.pyplot.pcolormesh` to generate the grid. For further
    customization, pass extra kwargs to :func:`~matplotlib.pyplot.pcolormesh`
    through the :python:`scatter_kwargs` parameter. For instance, to create black
    boundaries of width 0.1, pass in :python:`scatter_kwargs={"edgecolor": "black",
    "linewidth": 0.1}`.

    The result is an image converted from a Pillow image, so it has shape :python:`(channels, w, h)`,
    meaning reshaping of the dimensions is usually required to be able to show the image with
    matplotlib.

    Parameters
    ----------
    archive : algatross.quality_diversity.archives.unstructured.UnstructuredArchive
        A 1D or 2D :class:`~algatross.quality_diversity.archives.unstructured.UnstructuredArchive`.
    ax : mpl.axes.Axes | None
        Axes on which to plot the heatmap. If :data:`python:None`, the current axis will be used.
    df : ribs.archives.ArchiveDataFrame
        If provided, we will plot data from
        this argument instead of the data currently in the archive. This
        data can be obtained by, for instance, calling
        :meth:`~ribs.archives.ArchiveBase.data` with :python:`return_type="pandas"`
        and modifying the resulting
        :class:`~ribs.archives.ArchiveDataFrame`. Note that, at a minimum,
        the data must contain columns for index, objective, and measures. To
        display a custom metric, replace the "objective" column.
    transpose_measures : bool
        By default, the first measure in the archive
        will appear along the x-axis, and the second will be along the
        y-axis. To switch this behavior (i.e. to transpose the axes), set
        this to :data:`python:True`. Does not apply for 1D archives.
    cmap : str | mpl.colors.Colormap
        The colormap to use when
        plotting intensity. Either the name of a
        :class:`~matplotlib.colors.Colormap`, a list of RGB or RGBA colors
        (i.e. an :math:`N \times 3` or :math:`N \times 4` array), or a
        :class:`~matplotlib.colors.Colormap` object.
    aspect : Literal["auto", "equal"] | float | None
        The aspect ratio of the heatmap (i.e. height/width). Defaults to
        ``'auto'`` for 2D and ``0.5`` for 1D. ``'equal'`` is the same as
        :python:`aspect=1`. See :meth:`~matplotlib.axes.Axes.set_aspect` for more info.
    vmin : float | None
        Minimum objective value to use in the plot. If :data:`python:None`,
        the minimum objective value in the archive is used.
    vmax : float | None
        Maximum objective value to use in the plot. If :data:`python:None`,
        the maximum objective value in the archive is used.
    cbar : Literal["auto"] | mpl.axes.Axes | None
        By default, this is set to ``'auto'`` which displays the colorbar
        on the archive's current :class:`~matplotlib.axes.Axes`. If
        :data:`python:None`, then colorbar is not displayed. If this is an
        :class:`~matplotlib.axes.Axes`, displays the colorbar on the
        specified Axes.
    cbar_kwargs : dict | None
        Additional kwargs to pass to :func:`~matplotlib.pyplot.colorbar`.
    rasterized : bool
        Whether to rasterize the heatmap. This can be useful
        for saving to a vector format like PDF. Essentially, only the
        heatmap will be converted to a raster graphic so that the archive
        cells will not have to be individually rendered. Meanwhile, the
        surrounding axes, particularly text labels, will remain in vector
        format. This is implemented by passing ``rasterized`` to
        :func:`~matplotlib.pyplot.pcolormesh`, so passing ``"rasterized"``
        in the ``scatter_kwargs`` below will raise an error.
    plot_strategy : Literal["patches", "scatter"]
        The type of artist to use for plotting. Allowable
        values are:

        - "patches" (default): Use a :class:`~matplotlib.patches.Patch`
            object with a
            :class:`~matplotlib.collections.PatchCollection` to render
            The elites as rectangles (1D), or circles (2D)
        - "scatter": Use :func:`~matplotlib.pyplot.scatter` to render
            The elites as a scatterplot.

            .. note:: that due to the way markers are handled, the size
                and shape of the markers will not scale with the image
                or aspect ratio.

    plot_kwargs : dict | None
        Additional kwargs to pass to
        :class:`~matplotlib.collections.PatchCollection` or
        :func:`~matplotlib.pyplot.scatter` depending on the setting for
        ``plot_strategy``

    Returns
    -------
    torch.Tensor
        The heatmap image as a tensor

    Raises
    ------
    ValueError
        The archive's measure dimension must be 1D or 2D.

    Examples
    --------
        .. plot::
            :context: close-figs

            Heatmap of a 2D UnstructuredArchive

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from algatross.quality_diversity.archives.unstructured import UnstructuredArchive
            >>> from algatross.quality_diversity.visualization import nearest_neighbor_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = UnstructuredArchive(solution_dim=2, measure_dim=2, k_neighbors=5, novelty_threshold=1.0)
            >>> x = np.random.uniform(-1, 1, 10000)
            >>> y = np.random.uniform(-1, 1, 10000)
            >>> archive.add(solution=np.stack((x, y), axis=1), objective=-(x**2 + y**2), measures=np.stack((x, y), axis=1))
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> plt.imshow(nearest_neighbor_archive_heatmap(archive).moveaxis((0), (2)))
            >>> plt.title("Negative sphere function")
            >>> plt.axis("off")
            >>> plt.show()

        .. plot::
            :context: close-figs

            Heatmap of a 1D UnstructuredArchive

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from algatross.quality_diversity.archives.unstructured import UnstructuredArchive
            >>> from algatross.quality_diversity.visualization import nearest_neighbor_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = UnstructuredArchive(solution_dim=2, measure_dim=1, k_neighbors=5, novelty_threshold=1.0)
            >>> x = np.random.uniform(-1, 1, 1000)
            >>> archive.add(solution=np.stack((x, x), axis=1), objective=-(x**2), measures=x[:, None])
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> plt.imshow(nearest_neighbor_archive_heatmap(archive).moveaxis((0), (2)))
            >>> plt.title("Negative sphere function")
            >>> plt.axis("off")
            >>> plt.show()
    """
    if plot_strategy not in {"patches", "scatter"}:
        msg = f"Specified invalid plotting strategy ({plot_strategy}), expected `scatter` or `patches`."
        raise ValueError(msg)

    if aspect is None:
        # Handles default aspects for different dims.
        aspect = 0.5 if archive.measure_dim == 1 else "auto"

    # Try getting the colormap early in case it fails.
    cmap = retrieve_cmap(cmap)

    # Retrieve archive data.
    if df is None:
        objective_batch = archive.data("objective")
    else:
        df = validate_df(df)  # noqa: PD901
        objective_batch = df["objective"]

    # Retrieve data from archive.
    lower_bounds, upper_bounds, measures_batch, *_ = da.compute(archive.lower_bounds, archive.upper_bounds, archive.data("measures"))
    k_neighbors = min(archive.k_neighbors + 1, len(measures_batch))
    dist, indices = archive._cur_kd_tree.query(measures_batch, k=k_neighbors)  # noqa: SLF001
    mask = np.logical_and(indices != archive._cur_kd_tree.n, indices != np.arange(len(measures_batch))[:, None])  # noqa: SLF001
    mean_dist = dist.sum(axis=1, where=mask, initial=0.0) / mask.sum(axis=1)
    mean_dist = np.nan_to_num(mean_dist, nan=0, posinf=0, neginf=0)
    max_dist = mean_dist.max()

    # set the keyword arguments.
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    plot_kwargs.setdefault("linewidths", 0.0)
    vmin = np.min(objective_batch) if vmin is None else vmin
    vmax = np.max(objective_batch) if vmax is None else vmax

    if archive.measure_dim == 1:
        # Initialize the axis.
        fig, ax_ = plt.subplots()
        ax = ax_ if ax is None else ax
        ax.set_aspect(aspect)
        ax.set_xlim(lower_bounds[0] - max_dist, upper_bounds[0] + max_dist)
        ax.set_ylim(0.0, 1.0)

        # Turn off yticks; this is a 1D plot so only the x-axis matters.
        ax.set_yticks([])

        x_points = measures_batch[:, 0]

        t: mc.PatchCollection | mc.PathCollection
        if plot_strategy == "scatter":
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            angle = np.arctan(get_aspect(ax))
            mean_dist = np.pi * bbox.width * np.power(72 * mean_dist, 2.0) * np.sin(angle)

            t = ax.scatter(
                x_points,
                np.full_like(x_points, fill_value=0.5),
                c=objective_batch,
                s=mean_dist,
                marker="o",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=rasterized,
                **plot_kwargs,
            )
        elif plot_strategy == "patches":
            # move the points over since Rectangle coordinates start in the
            # lower left
            x_points = x_points - mean_dist  # noqa: PLR6104
            width = 2 * mean_dist
            height = np.ones_like(width)
            patches: list[mpl.patches.Circle | mpl.patches.Rectangle] = [
                mpl.patches.Rectangle((x, 0.0), width=w, height=h) for x, w, h in zip(x_points, width, height, strict=False)
            ]
            t = mc.PatchCollection(patches, array=objective_batch, cmap=cmap, clim=(vmin, vmax), rasterized=rasterized, **plot_kwargs)
            ax.add_collection(t)

    elif archive.measure_dim == 2:  # noqa: PLR2004
        # Initialize the axis.
        fig, ax_ = plt.subplots()
        ax = ax_ if ax is None else ax
        ax = plt.gca() if ax is None else ax
        ax.set_aspect(aspect)

        x_points, y_points = measures_batch[:, 0], measures_batch[:, 1]

        if transpose_measures:
            # Since the archive is 2D, transpose by swapping the x and y
            # boundaries and by flipping the bounds (the bounds are arrays of
            # length 2).
            x_points, y_points = y_points, x_points
            lower_bounds = np.flip(lower_bounds)
            upper_bounds = np.flip(upper_bounds)

        # Initialize the axis.
        ax.set_xlim(lower_bounds[0] - max_dist, upper_bounds[0] + max_dist)
        ax.set_ylim(lower_bounds[1] - max_dist, upper_bounds[1] + max_dist)

        if plot_strategy == "scatter":
            # do some scaling to the marker sizes so they are roughly the same
            # area if calculated from plot points
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            angle = np.arctan(get_aspect(ax))
            mean_dist = np.pi * np.linalg.norm([bbox.width, bbox.height]) * np.power(mean_dist * 72, 2.0) * np.sin(angle) * np.cos(angle)

            t = ax.scatter(
                x_points,
                y_points,
                c=objective_batch,
                s=mean_dist,
                marker="o",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=rasterized,
                **plot_kwargs,
            )
        elif plot_strategy == "patches":
            patches = [mpl.patches.Circle(xy, radius=radius) for (radius, xy) in zip(mean_dist, measures_batch, strict=False)]
            t = mc.PatchCollection(patches, array=objective_batch, cmap=cmap, clim=(vmin, vmax), rasterized=rasterized, **plot_kwargs)
            ax.add_collection(t)

    else:
        fig, ax_ = plt.subplots()

        axes = fig.subplot_mosaic(
            [
                [f"{d1},{d2}" if d2 < archive.measure_dim - 1 else f"{archive.measure_dim}" for d2 in range(archive.measure_dim)]
                for d1 in range(1, archive.measure_dim)
            ],
            gridspec_kw={"width_ratios": [1] * (archive.measure_dim - 1) + [archive.measure_dim / 19.0]},
        )

        for d2 in range(1, archive.measure_dim):
            for d1 in range(archive.measure_dim - 1):
                ax = axes[f"{d2},{d1}"]
                if d2 < d1 + 1:
                    ax.set(visible=False)
                    continue
                x_points, y_points = measures_batch[:, d1], measures_batch[:, d2]

                # Initialize the axis.
                if d2 == archive.measure_dim - 1:
                    ax.set_xlabel(f"measure {d1}")
                if d1 == 0:
                    ax.set_ylabel(f"measure {d2}")
                if d2 > d1 > 0:
                    ax.sharey(axes[f"{d2},{0}"])
                else:
                    ax.set_ylim(lower_bounds[d2] - max_dist, upper_bounds[d2] + max_dist)

                if d1 < d2 < archive.measure_dim - 1:
                    # last column of d2 shouldn't share `x` with anybody
                    ax.sharex(axes[f"{archive.measure_dim - 1},{d1}"])
                else:
                    ax.set_xlim(lower_bounds[d1] - max_dist, upper_bounds[d1] + max_dist)

                if plot_strategy == "scatter":
                    # do some scaling to the marker sizes so they are roughly the same
                    # area if calculated from plot points
                    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    angle = np.arctan(get_aspect(ax))
                    mean_dist = (
                        np.pi * np.linalg.norm([bbox.width, bbox.height]) * np.power(mean_dist * 72, 2.0) * np.sin(angle) * np.cos(angle)
                    )

                    t = ax.scatter(
                        x_points,
                        y_points,
                        c=objective_batch,
                        s=mean_dist,
                        marker="o",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        rasterized=rasterized,
                        **plot_kwargs,
                    )
                elif plot_strategy == "patches":
                    patches = [
                        mpl.patches.Circle(xy, radius=radius) for (radius, xy) in zip(mean_dist, measures_batch[:, [d1, d2]], strict=False)
                    ]
                    t = mc.PatchCollection(
                        patches,
                        array=objective_batch,
                        cmap=cmap,
                        clim=(vmin, vmax),
                        rasterized=rasterized,
                        **plot_kwargs,
                    )
                    ax.add_collection(t)
                # Create color bar - use a separate mappable so the colorbar is opaque even
                # with alpha > 0.
        ax = axes[f"{archive.measure_dim}"]
    # Create color bar - use a separate mappable so the colorbar is opaque even
    # with alpha > 0.
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_clim(vmin, vmax)
    if archive.measure_dim <= 2:  # noqa: PLR2004
        set_cbar(mappable, ax, cbar, cbar_kwargs)
    else:
        fig.colorbar(mappable, cax=ax)
    with io.BytesIO() as buff:
        fig.savefig(buff, format="jpeg")
        buff.seek(0)
        pil_img = PIL.Image.open(buff)
        img = ToTensor()(pil_img)
    if not buff.closed:
        buff.close()
    pil_img.close()
    fig.clear()
    plt.close(fig)
    return img

"""Inspection environments capable of rendering."""

import io

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch

import numpy as np

import PIL
import PIL.Image

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.spatial.transform import Rotation

from algatross.environments.pettingzoo_env import ParallelPettingZooEnv


class Arrow3D(FancyArrowPatch):
    """Helper class to draw an arrow in 3D projected plot.

    cf https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c

    Parameters
    ----------
    x : float | np.ndarray
        The ``x`` coordinate(s).
    y : float | np.ndarray
        The ``y`` coordinate(s).
    z : float | np.ndarray
        The ``z`` coordinate(s).
    dx : float | np.ndarray
        The x-component of the arrows direction.
    dy : float | np.ndarray
        The y-component of the arrows direction.
    dz : float | np.ndarray
        The z-component of the arrows direction.
    `*args`
        Additional positional arguments.
    `**kwargs`
        Additional keyword arguments.
    """

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):  # noqa: D102
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)  # noqa: F841
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):  # noqa: ARG002, D102
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):  # noqa: N802 # numpydoc ignore=PR01
    """Add an 3d arrow to an `Axes3D` instance."""
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


Axes3D.arrow3D = _arrow3D


class InspectionSimpleRenderer:
    """Simplified renderer for multi-agent inspection environment."""

    def __init__(self):
        self.color_map = {
            "deputy_0": "orange",
            "deputy_1": "blue",
            "deputy_2": "maroon",
            "deputy_3": "salmon",
            "deputy_4": "pink",
            "deputy_5": "cyan",
        }
        self.reset()

    def reset(self):
        """Reset the renderer states."""
        self.deputy_positions = defaultdict(list)

    def render(self, env: Any) -> PIL.Image.Image:  # noqa: ANN401
        """
        Render a single frame of the unwrapped environment.

        This is kind of expensive with matplotlib, but we won't be stuck with it for long?

        Parameters
        ----------
        env : Any
            The environment to render.

        Returns
        -------
        PIL.Image.Image
            The rendered frame.
        """
        figure = plt.figure(figsize=(6, 6))
        deputies_ax = figure.add_subplot(projection="3d")

        ax = deputies_ax

        # deputies
        for agent in env.possible_agents:
            self.deputy_positions[agent].append(env.deputies[agent].position.copy())
            pos_np = np.asarray(self.deputy_positions[agent])

            ax.scatter(  # type: ignore[misc]
                pos_np[:, 0],
                pos_np[:, 1],
                pos_np[:, 2],
                label=agent,
                marker="^",
                s=10,
                color=self.color_map[agent],
            )

        # points
        inspection_points = env.chief.inspection_points.points
        points_np = np.asarray(
            # n x 3
            [inspection_points[i].position for i in range(len(inspection_points))],
        )
        satisfied_l = []
        for i in range(len(inspection_points)):
            if inspection_points[i].inspected:
                satisfied_l.append("green")
            else:
                satisfied_l.append("red")

        ax.scatter(  # type: ignore[misc]
            points_np[:, 0],
            points_np[:, 1],
            points_np[:, 2],
            label="inspection points",
            marker="o",
            s=12,
            color=satisfied_l,
        )

        # chief
        ax.scatter(  # type: ignore[misc]
            *env.chief.position,
            label="chief",
            marker="o",
            s=16,
            color="purple",
        )

        ax.set_xlabel("R")
        ax.set_ylabel("I")
        ax.set_zlabel("C")  # type: ignore[attr-defined]

        ax.legend()
        plt.tight_layout()

        # write
        buf = io.BytesIO()
        figure.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        plt.close()

        del figure
        del ax

        return img


def camera_cone_factory(current_position: np.ndarray, orientation: np.ndarray, n_points: int = 15) -> np.ndarray:
    """
    Render a rotated camera cone for a deputy position.

    Given an offset from the chief in Hill frame (deputy's current position) and
    deputy camera orientation quaternion, render a 3D cone which illustates field of view.
    Note: visualization for the actual fov and focal length are not implemented yet.

    This function is based on examples from: https://likegeeks.com/3d-cones-python-matplotlib/

    Parameters
    ----------
    current_position : np.ndarray
        The current position of the deputy in Hill frame.
    orientation : np.ndarray
        Quaternion depicting the camera attitude.
    n_points : int, optional
        Controls the polygon count of rendered cone polygon (final being n_points**2). Default is 15

    Returns
    -------
    np.ndarray
        The rendered deputy view.
    """
    height = 10
    # TODO: change with focal length/fov. Actual fov may closer resemble an oblique cone.
    r = np.linspace(0, height, n_points)
    theta = np.linspace(0, 2 * np.pi, n_points)
    # create polygon mesh
    rp, thetap = np.meshgrid(r, theta)

    R = Rotation.from_quat(orientation)  # noqa: N806

    # to cartesian system with each axis as (p, p), then flatten to (P,)
    x = np.reshape(rp * np.cos(thetap), (n_points**2,))
    y = np.reshape(rp * np.sin(thetap), (n_points**2,))
    z = np.reshape(rp, (n_points**2,))
    # (P,) x 3 -> (3, P) -> (P, 3) to work with Rotation method
    xyz = np.stack([x, y, z]).T
    # (P, 3) -> R(P, 3) -> (3, P) -> (3, p, p)
    polygon_rot = R.apply(xyz).T.reshape((3, n_points, n_points))

    # then do, e.g., ax.plot_surface(polygon_rot[0], polygon_rot[1], polygon_rot[2], alpha=0.75)
    # apply Hill frame offset
    return np.stack([current_position[0] + polygon_rot[0], current_position[1] + polygon_rot[1], current_position[2] + polygon_rot[2]])


def append_flattened_dict(d1: dict[str, list], d2: dict[str, Any]):
    """
    Append a flattened version as list.

    Parameters
    ----------
    d1 : dict[str, list]
        The base dictionary
    d2 : dict[str, Any]
        The dictionary to use when updating the base

    Raises
    ------
    KeyError
        If ``d2`` has keys which are not in ``d1``
    """
    try:
        for k in d2:  # noqa: PLC0206
            d1[k].append(d2[k])
    except KeyError as e:
        msg = f"The key {k} was not found in d1! d1 keys: {d1.keys()}. d2 keys: {d2.keys()}"
        raise KeyError(msg) from e


class CompositedInspectionRenderer(InspectionSimpleRenderer):
    """Variant of simple renderer with more features.

    1) Three composited figures instead of one, including reward table and chief minimap.
    2) Camera cones to loosely visualize deputy camera fov.
    3) Sun vector on the XY plane.
    """

    def reset(self):
        """Reset render states."""
        self.deputy_positions = defaultdict(list)
        self.deputy_orientations = defaultdict(list)
        self.reward_components_l = None

    def render(self, env: Any, fixed_limits: bool = False) -> PIL.Image.Image:  # noqa: ANN401, PLR0915
        """
        Render the current timestep using matplotlib.

        Parameters
        ----------
        env : Any
            The environment to render.
        fixed_limits : bool
            Fix the deputy plot limits. Otherwise it will expand as agents move around. Default is :data:`python:True`

        Returns
        -------
        PIL.Image.Image
            The rendered frame.

        Raises
        ------
        KeyError
            If an attempt is made to retrieve an undefined reward from the environment.
        """
        if self.reward_components_l is None:
            # On the zeroth env step, reward names are not given by env, so we would only have agent names in the hud below.
            # Instead bake them in so we avoid an ugly jitter at the start, maybe fix this later?
            if "SixDof" in env.__class__.__name__:
                reward_names = ["observed_points", "delta_v", "success", "crash", "live_timestep", "facing_chief"]
            else:
                reward_names = ["observed_points", "delta_v", "success", "crash"]

            reward_dict = {name: [0] for name in reward_names}
            self.reward_components_l = {agent: reward_dict.copy() for agent in env.possible_agents}

        # single frame  figsize=(6, 8)
        figure = plt.figure(figsize=(9, 9))
        grid = gridspec.GridSpec(3, 3, figure=figure)
        grid.update(wspace=0.00, hspace=0.05, left=0.02, right=0.98, bottom=0.02, top=0.95)
        # index will be (row, col)
        deputies_ax: Axes3D = plt.subplot(grid[:2, :], projection="3d")

        # main axes
        ax: Axes3D = deputies_ax

        # deputies
        for agent in env.possible_agents:
            self.deputy_positions[agent].append(env.deputies[agent].position.copy())
            self.deputy_orientations[agent].append(env.deputies[agent].orientation.copy())
            append_flattened_dict(self.reward_components_l[agent], env.reward_components[agent])

            # position
            pos_np = np.asarray(self.deputy_positions[agent])
            ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], label=agent, marker="^", s=10, color=self.color_map[agent])

            camera_polygon = camera_cone_factory(env.deputies[agent].position, env.deputies[agent].camera.orientation)
            ax.plot_surface(camera_polygon[0], camera_polygon[1], camera_polygon[2], alpha=0.75, color=self.color_map[agent])

        # reward hud
        #    agent      delta_v     ...      reward_n
        # -    0          23.33     ...         42
        # -    1          32.44     ...         90
        # assumes same reward space for each agent!
        reward_ax = plt.subplot(grid[2, :2])

        reward_names = self.reward_components_l[env.possible_agents[0]].keys()
        df_data: dict[str, Any] = {"agent": []} | {reward: [] for reward in reward_names}

        for agent in self.reward_components_l:
            df_data["agent"].append(agent)

            for reward in reward_names:
                cell_val = np.sum(np.abs(self.reward_components_l[agent][reward]))  # sum up to this frame

                if isinstance(cell_val, float):
                    cell_val = np.around(cell_val, 3)
                df_data[reward].append(cell_val)

        rewards_df = pd.DataFrame.from_dict(df_data)
        try:
            rewards_df = rewards_df.astype({"observed_points": float, "crash": int, "success": int})
            reward_table = reward_ax.table(
                cellText=rewards_df.values,
                colLabels=rewards_df.columns,
                loc="center",
                rowLoc="center",  # header text placement
                cellLoc="center",  # cell text placement
                edges="horizontal",
            )
            reward_table.auto_set_font_size(False)
            reward_table.set_fontsize(9)
        except KeyError as e:
            msg = "Tried to access a reward that wasn't in the env!"
            raise KeyError(msg) from e

        reward_ax.set_frame_on(False)
        reward_ax.set_xticklabels([])
        reward_ax.set_yticklabels([])
        reward_ax.set_xticks([])
        reward_ax.set_yticks([])

        # points
        inspection_points = env.chief.inspection_points.points
        points_np = np.asarray(
            # n x 3
            [inspection_points[i].position for i in range(len(inspection_points))],
        )
        satisfied_l = []
        score = 0
        for i in range(len(inspection_points)):
            if inspection_points[i].inspected:
                score += 1
                satisfied_l.append("green")
            else:
                satisfied_l.append("red")

        # inspection points hud (and inside deputy view)
        chief_ax: Axes3D = plt.subplot(grid[2, 2], projection="3d")

        inspected_text = f"Inspected: {score}/{len(inspection_points)}"
        units = env.deputies[env.possible_agents[0]].position_with_units.units

        chief_ax.text2D(0.5, 1, inspected_text, horizontalalignment="center", verticalalignment="center", transform=chief_ax.transAxes)

        for _ax in [deputies_ax, chief_ax]:  # plot in hud and deputy view
            _ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], label="inspection points", marker="o", s=12, color=satisfied_l)

            # chief
            _ax.scatter(*env.chief.position, label="chief", marker="o", s=16, color="purple")

            # sun vector
            # big magnitude for deputy plot
            if _ax is deputies_ax:
                magnitude = 20
                mutation_scale = 10
            else:
                magnitude = 2
                mutation_scale = 20

            # XY plane only
            sun_vec = (magnitude * np.cos(env.sun.theta), magnitude * np.sin(env.sun.theta), 0)
            _ax.arrow3D(
                *sun_vec,  # x,  y,  z
                *np.negative(sun_vec),  # dx, dy, dz  (point back to chief)
                mutation_scale=mutation_scale,
                arrowstyle="-|>",
                linestyle="solid",
                ec="goldenrod",
                fc="yellow",
                label="sun vector",
            )

        ax.set_xlabel(f"R\n({units}s)")
        ax.set_ylabel(f"I\n({units}s)")
        ax.set_zlabel(f"C ({units}s)")

        chief_ax.set_xticklabels([])
        chief_ax.set_yticklabels([])
        chief_ax.set_zticklabels([])
        chief_ax.set_xticks([])
        chief_ax.set_yticks([])
        chief_ax.set_zticks([])
        chief_ax.set_xlim([-1, 1])
        chief_ax.set_ylim([-1, 1])
        chief_ax.set_zlim([-1, 1])

        ax.legend()
        if fixed_limits:
            ax.set_xlim([-env.max_distance // 2, env.max_distance // 2])
            ax.set_ylim([-env.max_distance // 2, env.max_distance // 2])
            ax.set_zlim([-env.max_distance // 2, env.max_distance // 2])

        # write
        buf = io.BytesIO()
        figure.savefig(buf)
        buf.seek(0)
        img = PIL.Image.open(buf)
        plt.close()

        del figure
        del ax

        return img


class RenderedParallelPettingZooInspection(ParallelPettingZooEnv):
    """A rendered version of sa-sims inspection."""

    renderer: InspectionSimpleRenderer = CompositedInspectionRenderer()

    def reset(self, *args, **kwargs):
        """
        Reset the environment's renderer.

        Parameters
        ----------
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            Observations for the first step of the new episode.
        dict
            Environment info.
        """
        self.renderer.reset()
        return super().reset()

    def render(self, *args, **kwargs):
        """
        Produce an image by hooking into the renderer.

        Parameters
        ----------
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        PIL.Image.Image
            A PIL format image of environment time step.
        """
        # operate on unwrapped parallel environment
        return self.renderer.render(self.get_sub_environments)

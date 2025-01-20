"""Loggers which output to console."""

import collections
import logging
import re
import socket

from collections import defaultdict
from collections.abc import Callable, Sequence
from math import log10
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rich import box, get_console
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text

from algatross.utils.debugging import get_debugging_ports
from algatross.utils.loggers.base_loggers import BaseHandler, BaseLogger
from algatross.utils.loggers.constants import LEVEL_COLORS
from algatross.utils.merge_dicts import merge_dicts

console = get_console()


def sort_by_string_number(key: str, sep: str = " ", prev: int = 0) -> str:
    """Sort a string lexographically unless it ends in a number.

    Parameters
    ----------
    key : str
        The key to sort.
    sep : str, optional
        A separator for splitting the key and determining whether it ends in a number, by default :python:`" "`.
    prev : int, optional
        The carried ordinal, default is 0. Users should not typically set this themselves.

    Returns
    -------
    str
        A key for sorting
    """
    if key.isdecimal() and float(key) != 0:
        l10 = log10(abs(float(key)))
        if l10 > 9:  # noqa: PLR2004
            # string sort would see
            return str(prev) + sort_by_string_number(str(int(l10)), prev=9) + key
        return str(prev) + str(int(l10)) + key
    if key.isdecimal() and float(key) == 0:
        return str(prev) + key
    return sep.join([sort_by_string_number(x) if x.isdecimal() else x for x in key.split(sep)])


class LogConsole(BaseLogger):
    """
    ConsoleLogger prints experiment results to the console.

    Parameters
    ----------
    log_file : Path
        The path to the log file for this experiment.
    reported_metrics : dict[Literal["mainland", "island", "archipelago"], dict[str, dict[Literal["alias", "key_length", "metric"], str | int]]] | None, optional
        The mapping of metrics to report for each component, default is :data:`python:None`.
    console : Console | None, optional
        The rich console object to log to, default is :data:`python:None`.
    live_display : bool, optional
        Whether to use rich's live-display interface with the console, default is :data:`python:False`.
    column_styles : dict[Literal["epoch", "id", "mainland", "island", "archipelago"], dict[str, Any]] | None, optional
        The custom styles to apply to the table columns.
    table_styles : dict[Literal["mainland", "island", "archipelago"], dict[str, Any]] | None, optional
        The the custom table style to use, default is :data:`python:None`
    config_file : str | None, optional
        The config file to report to the console, default is :data:`python:None`.
    log_dir : Path | str | None, optional
        The log directory to report to the console, default is :data:`python:None`.
    experiment_name : str | None, optional
        The experiment name to report to the console, default is :data:`python:None`.
    parallel_backend : str = "ray", optional
        The parallelization backend to use for determining what to print, default is :python:`"ray"`.
    dashboard_url : str | None, optional
        The URL to the cluster dashboard to report to the console, default is :data:`python:None`.
    session_dir : str | None, optional
        The directory of the parallelization session to report to the console, default is :data:`python:None`
    debug : bool, optional
        Whether to display the debugging table.
    `**kwargs`
        Additional keyword arguments.
    """  # noqa: E501

    default_reported_metrics: dict[
        Literal["mainland", "island", "archipelago"],
        dict[str, dict[Literal["alias", "key_length", "metric"], str | int]],
    ] = {  # noqa: RUF012
        "mainland": {
            "algorithm/evolve/fitness/team/total": {"alias": "fitness", "key_length": 1, "metric": "mean"},
            "fronts/total": {"alias": "fronts", "key_length": 0, "metric": "mean"},
            "current_population": {"alias": "current population", "key_length": 0, "metric": "mean"},
        },
        "island": {
            "conspecific_utility/cumulative": {"alias": "conspecific utility", "key_length": 0, "metric": "mean"},
            "archive/fitness": {"alias": "archive fitness", "key_length": 0, "metric": "mean"},
            "algorithm/evolve/fitness/team": {"alias": "last fitness", "key_length": 0, "metric": "mean"},
            "current_population": {"alias": "current population", "key_length": 0, "metric": "mean"},
        },
        "archipelago": {"topology/optimize_softmax/loss/mainland": {"alias": "softmax loss", "key_length": 2, "metric": "mean"}},
    }
    default_column_styles: dict[Literal["epoch", "id", "mainland", "island", "archipelago"], dict[str, Any]] = {  # noqa: RUF012
        "epoch": {"width": 10, "max_width": 20, "style": Style(color="magenta")},
        "id": {"width": 15, "max_width": 30, "style": Style(color="cyan")},
        "archipelago": {"style": Style(color="turquoise2"), "overflow": "fold", "ratio": 1},
        "mainland": {"style": Style(color="turquoise2"), "overflow": "fold", "ratio": 1},
        "island": {"style": Style(color="turquoise2"), "overflow": "fold", "ratio": 1},
    }
    default_table_styles: dict[Literal["mainland", "island", "archipelago"], dict[str, Any]] = {  # noqa: RUF012
        "mainland": {
            "title_justify": "left",
            "title_style": Style(bold=True, color="white", bgcolor="#6495ED"),
            "header_style": Style(bold=True),
            "box": box.ROUNDED,
            "expand": True,
        },
        "island": {
            "title_justify": "left",
            "title_style": Style(bold=True, color="white", bgcolor="#3CB371"),
            "header_style": Style(bold=True),
            "box": box.ROUNDED,
            "expand": True,
        },
        "archipelago": {
            "title_justify": "left",
            "title_style": Style(bold=True, color="white", bgcolor="#FF6347"),
            "header_style": Style(bold=True),
            "box": box.ROUNDED,
            "expand": True,
        },
    }

    archipelago_results: dict[str, Any]
    """The dictionary of archipelago results to report to the console."""
    island_results: dict[str, Any]
    """The dictionary of island results to report to the console."""
    mainland_results: dict[str, Any]
    """The dictionary of mainland results to report to the console."""
    sorted_arch_headers: list[str]
    """The archipelago headers, sorted."""
    sorted_isl: list[str]
    """The island headers, sorted."""
    sorted_ml_headers: list[str]
    """The mainland headers, sorted."""

    _archipelago_table: Table | None
    _mainland_table: Table | None
    _island_table: Table | None
    _port_table: Table | None = None
    _message_panel: Panel | None = None

    def __init__(
        self,
        log_file: Path,
        reported_metrics: (
            dict[Literal["mainland", "island", "archipelago"], dict[str, dict[Literal["alias", "key_length", "metric"], str | int]]] | None
        ) = None,
        console: Console | None = None,
        live_display: bool = False,
        column_styles: dict[Literal["epoch", "id", "mainland", "island", "archipelago"], dict[str, Any]] | None = None,
        table_styles: dict[Literal["mainland", "island", "archipelago"], dict[str, Any]] | None = None,
        config_file: str | None = None,
        log_dir: Path | str | None = None,
        experiment_name: str | None = None,
        parallel_backend: str = "ray",
        dashboard_url: str | None = None,
        session_dir: str | None = None,
        debug: bool = False,
        **kwargs,
    ):
        self._log_file = log_file
        self._reported_metrics = merge_dicts(self.default_reported_metrics, reported_metrics or {})
        self._column_styles = merge_dicts(self.default_column_styles, column_styles or {})
        self._table_styles = merge_dicts(self.default_table_styles, table_styles or {})
        self._console = get_console() if console is None else console

        self.init_layout(live_display, config_file, log_dir, experiment_name, parallel_backend, dashboard_url, session_dir, debug, **kwargs)

        self.sorted_arch_headers = sorted([alias["alias"] or name for name, alias in self.reported_metrics["archipelago"].items()])  # type: ignore[misc]
        self.sorted_ml_headers = sorted([alias["alias"] or name for name, alias in self.reported_metrics["mainland"].items()])  # type: ignore[misc]
        self.sorted_isl_headers = sorted([alias["alias"] or name for name, alias in self.reported_metrics["island"].items()])
        self._archipelago_table = None
        self._mainland_table = None
        self._island_table = None
        self.archipelago_results: dict[str, Any] = defaultdict(dict)
        self.mainland_results: dict[str, Any] = defaultdict(dict)
        self.island_results: dict[str, Any] = defaultdict(dict)
        self.reset_tables()

        self._layout["lower"]["result_table"].update(self.renderables)
        if live_display:
            live = Live(
                self._layout,
                refresh_per_second=0.2,
                console=self._console,
                screen=False,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="visible",
            ).__enter__()
            self._live: Callable = live.update

            def close_live(live=live):
                live.__exit__()

        else:
            self._live = get_console().print

            def close_live(live=None):
                pass

        self._close_live = close_live

    @property
    def renderables(self) -> Group:
        """
        A group of tables for the archipelago, island, mainland.

        Returns
        -------
        Group
            The grouped tables.
        """
        return Group(self._archipelago_table, self._island_table, self._mainland_table, fit=True)

    @property
    def reported_metrics(
        self,
    ) -> dict[Literal["mainland", "island", "archipelago"], dict[str, dict[Literal["alias", "key_length", "metric"], str | int]]]:
        """
        Dictionary of metrics to report to the console from the mainlands, islands, and archipelago.

        - Any logged results which contain the string segment are extracted and displayed in the console under the ``alias`` if one exists.
        - The matching results are further subdivided into groups where the last ``key_length`` segments of the metric key defines a group.
        - The metric type (``"min"``, ``"max"``, ``"mean"``) reported in the table is given by``metric``

        For example, consider a ``reported_metrics`` dictionary like so:

        .. code:: python

            reported_metrics = {
                "mainland": {
                    "algorithm/evolve/fitness/team/total": {
                        "alias": "fitness",
                        "key_length": 1,
                        "metric": "mean",
                    },
                }
            }

        We observe the following four behaviors:

        1. Any reported metrics beginning with :python:`"mainland"` and containing the string
            :python:`"algorithm/evolve/fitness/team/total"` will appear in the console report.
        2. The ``mean`` value is extracted for reporting. The metric key must end with this value.
        3. The metric will appear in the table under the column with an alias of "fitness"
        4. The last ``1`` segments of the metric key will be used to group the results into additional columns. So a set of keys like
            ``...algorithm/evolve/fitness/team/total/.../time_score_mean``, ``...algorithm/evolve/fitness/team/total/.../points_score_mean``
            will subdivide the results into two columns: ``fitness: time_score`` and ``fitness: points_score``

            Segments are separated by ``/``. In the case of :python:`key_length=0` no grouping will occur, otherwise :python:`key_length`
            segments define a single group.

        Returns
        -------
        dict[Literal["mainland", "island", "archipelago"], dict[str, dict[Literal["alias", "key_length", "metric"], str | int]]]
            A mapping from metric source (island, mainland, archipelago) to a partial metric string ending with ``metric``, as well as an
            ``alias`` to display and a ``key_length`` denoting the number of key segments to use for grouping.
        """
        return self._reported_metrics

    @reported_metrics.setter
    def reported_metrics(self, other: dict[str, dict[str, dict[str, Any]]]):
        self._reported_metrics = merge_dicts(self._reported_metrics, other)

    @property
    def column_styles(self) -> dict[Literal["epoch", "id", "mainland", "island", "archipelago"], dict[str, Any]]:
        """
        Mapping from table to style dictionary to use for each Column.

        Styles under ``epoch`` and ``id`` are global styles applied to all tables. See https://github.com/Textualize/rich
        documentation for more info about styling.

        Returns
        -------
        dict[Literal["epoch", "id", "mainland", "island", "archipelago"], dict[str, Any]]
            Mapping of styles to use with the columns of each table.
        """
        return self._column_styles

    @column_styles.setter
    def column_styles(self, other: dict[str, dict[str, Any]]):
        self._column_styles = merge_dicts(self._column_styles, other)

    @property
    def table_styles(self) -> dict[Literal["mainland", "island", "archipelago"], dict[str, Any]]:
        """
        Mapping of styles to use for each table.

        See [Rich](https://github.com/Textualize/rich) documentation for more info about styling.

        Returns
        -------
        dict[Literal["mainland", "island", "archipelago"], dict[str, Any]]
            Mapping from table to a style dictionary to pass to the constructor.
        """
        return self._table_styles

    @table_styles.setter
    def table_styles(self, other: dict[str, dict[str, Any] | None]):
        self._table_styles = merge_dicts(self._table_styles, other)

    @property
    def port_table(self) -> Table:
        """
        Get the table of debugging ports.

        Returns
        -------
        Table
            The table of debugging ports
        """
        if not self._port_table:
            self._port_table_entries: dict[str, str] = {}
            self.build_port_table()

        updates = {
            port: actor
            for port, actor in get_debugging_ports().items()
            if port not in self._port_table_entries or self._port_table_entries[port] != actor
        }

        if updates:
            self._port_table_entries.update(updates)
            self.build_port_table()

        return self._port_table

    @property
    def message_panel(self) -> Panel:
        """
        Get the panel of messages.

        Returns
        -------
        Panel
            The panel of messages
        """
        if not self._message_panel:
            self.build_message_panel()

        return self._message_panel

    def init_layout(
        self,
        live_display: bool = False,
        config_file: str | None = None,
        log_dir: Path | str | None = None,
        experiment_name: str | None = None,
        parallel_backend: str = "ray",
        dashboard_url: str | None = None,
        session_dir: str | None = None,
        debug: bool = False,
    ):
        """Initialize the console layout.

        Parameters
        ----------
        live_display : bool, optional
            Whether or not to use rich's LiveDisplay
        config_file : str | None, optional
            The path to the experiment configuration file, :data:`python:None`
        log_dir : Path | str | None, optional
            The log directory for the experiment, :data:`python:None`
        experiment_name : str | None, optional
            The name of the experiment, :data:`python:None`
        parallel_backend : str, optional
            The parallelization backend, by default "ray"
        dashboard_url : str | None, optional
            The URL to the cluster dashboard, :data:`python:None`
        session_dir : str | None
            The directory for the experiment session
        debug : bool, optional
            Whether to display the debug panel, :data:`python:False`
        """
        self._layout = Layout()

        progress = Progress(
            SpinnerColumn("moon", speed=1.0),
            TextColumn("Elapsed Time :stopwatch: "),
            TimeElapsedColumn(),
            console=self._console,
        )
        progress.add_task("main_task")
        if not live_display:
            progress.start()

        config_file_text = Text().from_markup(":file_folder:  Config File: ")
        config_file_text.append(str(Path(config_file).resolve()) if config_file else "", style=Style(bold=True, color="#FF79C6"))
        exp_name_text = Text().from_markup(":name_badge:  Experiment Name: ")
        exp_name_text.append(experiment_name, style=Style(bold=True, color="#BD93F9"))
        log_dir_text = Text().from_markup(":notebook_with_decorative_cover:  Log Directory: ")
        log_dir_text.append(str(Path(log_dir).resolve()) if log_dir else "", style=Style(bold=True, color="#6272A4"))
        left_panels: list[Text | Progress] = ([progress] if live_display else [Text()]) + [exp_name_text]  # type: ignore[list-item]
        right_panels = ([] if live_display else [Text()]) + [config_file_text, log_dir_text]
        if dashboard_url:
            _, host, port = self.make_port_table_row("", dashboard_url)
            dashboard_text = Text().from_markup(f":laptop_computer:  {parallel_backend.capitalize()} Dashboard: ")
            dashboard_text.append(host)
            dashboard_text.append(":")
            dashboard_text.append(port)
            left_panels.append(dashboard_text)
        if session_dir:
            session_text = Text().from_markup(f":wrench:  {parallel_backend.capitalize()} Session Directory: ")
            session_text.append(Text(session_dir, style=Style(color="#8BE9FD", bold=True)))
            right_panels.append(session_text)

        self._log_file = (Path(log_dir) / "app.log").resolve() if log_dir else None
        self._messages: collections.deque[dict[str, Any]] = collections.deque(maxlen=4)
        upper_layout = Layout(name="upper", visible=True, ratio=0, minimum_size=4)
        upper_layout.split_row(
            Layout(Group(*left_panels), name="left_panels", ratio=1),
            Layout(Group(*right_panels), name="right_panels", ratio=2),
        )
        lower_layout = Layout(name="lower", ratio=2, visible=True)
        lower_layout.split_row(Layout(name="result_table", ratio=5), Layout(self.port_table, name="port_table", visible=debug, ratio=2))
        self._layout.split_column(upper_layout, Layout(self.message_panel, name="messages", ratio=0, minimum_size=6), lower_layout)

    def build_port_table(self):
        """Rebuild the port table."""
        self._port_table = Table(
            Column(header="Actor", overflow="visible"),
            Column(header="Host :link:", max_width=15),
            Column(header="Port :electric_plug:", max_width=7),
            title=":lady_beetle: Debugging Ports :lady_beetle:",
            title_style=Style(bgcolor="#FFB86C", italic=True, bold=True),
            title_justify="left",
            box=box.ROUNDED,
            expand=True,
        )
        for port, actor in sorted(self._port_table_entries.items(), key=lambda x: sort_by_string_number(x[1])):
            self._port_table.add_row(*self.make_port_table_row(actor, port))

    def build_message_panel(self):
        """Rebuild the messages panel."""
        title = Text.from_markup(":page_with_curl:  Messages :page_with_curl:", style=Style(bgcolor="#44475A", bold=True))
        group = []
        for msg in self._messages:
            txt = Text()
            if (actor := msg["extra"].get("actor")) is None and (actor_pid := msg["extra"].get("process")):
                actor = f"pid={actor_pid}"
            if not actor:
                actor = "Unknown"
            actor_style = Style(color="cyan") if actor != "Unknown" else Style(color="grey0", dim=True, italic=True)
            txt.append(f"{actor} ", style=actor_style)
            txt.append(f"{msg['extra']['levelname']} ", style=Style(color=LEVEL_COLORS[msg["extra"]["levelno"]], italic=True))
            txt.append(self.parse_urls(msg["message"]))
            group.append(txt)
        self._message_panel = Panel(Group(*group), title=title, title_align="left")

    @staticmethod
    def make_port_table_row(actor: str, port: str) -> tuple[Text, Text, Text]:
        """Make a row for the port table using the given actor and port.

        Parameters
        ----------
        actor : str
            The name of the actor bound to the port for debugging
        port : str
            The URL or port

        Returns
        -------
        actor_text : Text
            The actor name with any trailing newlines trimmed
        host_text : Text
            The hostname text formatted as dim & italic if one was not included in ``port`` and thus had to be inferred.
        port_text : Text
            The port text formatted in bold
        """
        port_url = port.split(":")
        port_text = Text(port_url[-1], style=Style(bold=True, color="green"))
        if len(port_url) > 1:
            host_text = Text(port_url[0], style=Style(bold=False, color="green"))
        else:
            host_text = Text(socket.gethostbyname("localhost"), style=Style(bold=False, italic=True, dim=True, color="green"))
        return Text(actor.removesuffix("\n")), host_text, port_text

    def reset_tables(self, headers: dict[str, list[str]] | None = None):
        """Clear the data in the tables.

        We completely reconstruct the table each time so that previous rows are not re-printed.

        Parameters
        ----------
        headers : dict[str, list[str]] | None, optional
            Headers to use for the table after resetting, :data:`python:None`.
        """
        arch_columns = [Column(header="epoch", **self.column_styles["epoch"]), Column(header="id", **self.column_styles["id"])]
        ml_columns = [Column(header="epoch", **self.column_styles["epoch"]), Column(header="id", **self.column_styles["id"])]
        isl_columns = [Column(header="epoch", **self.column_styles["epoch"]), Column(header="id", **self.column_styles["id"])]

        arch_headers = self.sorted_arch_headers if headers is None or headers["archipelago"] is None else headers["archipelago"]
        ml_headers = self.sorted_ml_headers if headers is None or headers["mainland"] is None else headers["mainland"]
        isl_headers = self.sorted_isl_headers if headers is None or headers["island"] is None else headers["island"]

        arch_columns.extend([Column(header=str(head), **self.column_styles["archipelago"]) for head in arch_headers])
        ml_columns.extend([Column(header=str(head), **self.column_styles["mainland"]) for head in ml_headers])
        isl_columns.extend([Column(header=str(head), **self.column_styles["island"]) for head in isl_headers])

        self._archipelago_table = (
            Table(*arch_columns, title=":map_of_japan: Archipelago Results :map_of_japan:", **self.table_styles["archipelago"])
            if (headers is not None and headers["archipelago"] is not None) or self._archipelago_table is None
            else self._archipelago_table
        )
        self._mainland_table = (
            Table(*ml_columns, title=":desert_island:  Mainland Results :desert_island:", **self.table_styles["mainland"])
            if (headers is not None and headers["mainland"] is not None) or self._mainland_table is None
            else self._mainland_table
        )
        self._island_table = (
            Table(*isl_columns, title=":palm_tree: Island Results :palm_tree:", **self.table_styles["island"])
            if (headers is not None and headers["island"] is not None) or self._island_table is None
            else self._island_table
        )

    def update_debug_ports(self, debug_record: dict[str, str], extra: dict[str, str] | None = None):  # noqa: ARG002
        """Update the debugging ports from a dictionary.

        Parameters
        ----------
        debug_record : dict[str, str]
            The dictionary of updates.
        extra : dict[str, str] | None, optional
            A dictionary of extra information, :data:`python:None`.
        """
        port, actor = debug_record["port"], debug_record["actor"]
        if port not in self._port_table_entries or actor != self._port_table_entries[port]:
            self._port_table_entries[port] = actor
            self.build_port_table()
            self.refresh()

    def update_messages(self, message: str, extra: dict[str, str] | None = None):
        """Update the messages by appending the message to the message deque.

        Parameters
        ----------
        message : str
            The logged message.
        extra : dict[str, str] | None, optional
            Extra info from the log record, :data:`python:None`
        """
        self._messages.append({"message": message, "extra": extra})
        self.build_message_panel()
        self.refresh()

    def dump(self, result: dict[str, Any]):  # noqa: D102
        for key, val in result.items():
            reported_metrics_dict: dict | None = None
            table_keys = key.split("/")
            table_key = " ".join(table_keys[:2])
            result_key = "/".join(table_keys[2:])

            for name, res in zip(
                ("archipelago", "mainland", "island"),
                (self.archipelago_results, self.mainland_results, self.island_results),
                strict=True,
            ):
                if key.startswith(name):
                    if any(x in result_key for x in self.reported_metrics[name]):  # type: ignore[index]
                        reported_metrics_dict = self.reported_metrics[name]  # type: ignore[index]
                        results_dict = res[table_key]
                        break
                    if "epoch" in result_key.split("/") or "epoch_max" in result_key:
                        res[table_key]["epoch"] = (
                            int(val) if "epoch" not in res[table_key] or int(val) > res[table_key]["epoch"] else res[table_key]["epoch"]
                        )
                        break

            if reported_metrics_dict is not None:
                key_info = next(filter(lambda x: x[0] in result_key, reported_metrics_dict.items()))[1]
                if key.endswith(key_info["metric"]):
                    group_key = " ".join(
                        [key_info["alias"] + (":" if key_info["key_length"] > 0 else "")]
                        + table_keys[len(table_keys) - key_info["key_length"] :],
                    ).removesuffix("_mean")
                    results_dict[group_key] = val

        headers: dict[str, list[str] | None] = {}
        for name, res in zip(
            ("archipelago", "mainland", "island"),
            (self.archipelago_results, self.mainland_results, self.island_results),
            strict=True,
        ):
            if res:
                sorted_headers: list[str] | None = set()  # type: ignore[assignment]
                for hset in [set(key.keys()) - {"epoch"} for key in res.values()]:
                    sorted_headers.update(hset)  # type: ignore[union-attr]
                sorted_headers = sorted(sorted_headers, key=sort_by_string_number)
            else:
                sorted_headers = None
            headers[name] = sorted_headers

        self.reset_tables(headers=headers)

        for name, table, res in zip(
            ("archipelago", "mainland", "island"),
            (self._archipelago_table, self._mainland_table, self._island_table),
            (self.archipelago_results, self.mainland_results, self.island_results),
            strict=True,
        ):
            for t_key in sorted(res, key=sort_by_string_number):
                self._write_to_table(table, res[t_key]["epoch"], t_key, headers[name], res[t_key])

        self.refresh()

    @staticmethod
    def _write_to_table(table: Table, epoch: int, id_key: str, sorted_headers: Sequence[str], row_value_dict: dict[str, Any]):
        """
        Write the data to the results table.

        Parameters
        ----------
        table : Table
            The table object to write
        epoch : int
            The current epoch
        id_key : str
            The label to use for the row
        sorted_headers : Sequence[str]
            The table headers which have been sorted.
        row_value_dict : dict[str, Any]
            The dictionary of values to display in the table
        """
        row_values = [str(epoch), str(id_key)]
        for key in sorted_headers:
            vals = [row_value_dict[row_key] for row_key in row_value_dict if key in row_key and key != "epoch"]
            row_values.append(f"{np.array(vals).mean():0.4g}")
        table.add_row(*row_values)

    @staticmethod
    def parse_urls(string: str, sep: str = " ") -> Text:
        """Parse a string for URLs and format them as :class:`~rich.text.Text`.

        Parameters
        ----------
        string : str
            The string to parse.
        sep : str, optional
            The field separator of the string, by default :python:`" "`.

        Returns
        -------
        Text
            The formatted text.
        """
        strings = string.split(sep)
        text = Text()
        for ing in strings:
            if re.match(r"^([0-9]{1,3}\.){3}[0-9]{1,3}(:[0-9]{1,5})?$", ing):
                url = ing.split(":")
                text.append(Text(url[0], style=Style(bold=False, color="green")))
                if len(url) > 1:
                    text.append(":")
                    text.append(Text(url[1], style=Style(bold=True, color="green")))
            else:
                text.append(ing)
            text.append(sep)
        return text

    def close(self):  # noqa: D102
        self._close_live()

    def refresh(self):
        """Refresh the console by updating all the layouts."""
        self._layout["messages"].update(self.message_panel)
        self._layout["lower"]["result_table"].update(self.renderables)
        self._layout["lower"]["port_table"].update(self.port_table)
        self._live(self._layout)


class LogConsoleHandler(BaseHandler):
    """Handler for logging to a rich.Console.

    Calls the ``console_fn`` on the formatted log record

    Parameters
    ----------
    console_fn : Callable
        The console function to call when printing.
    level : int | str
        The logging level, default is 0.
    name : str | None
        The name for this handler, default is :data:`python:None`.
    strict_level : bool
        Whether to strictly obey logging level, default is :data:`python:True`.
    formatter : logging.Formatter | None
        The logging formatter to use with this handler.
    """

    def __init__(
        self,
        console_fn: Callable,
        level: int | str = 0,
        name: str | None = None,
        strict_level: bool = True,
        formatter: logging.Formatter | None = None,
    ) -> None:
        super().__init__(level=level, name=name, strict_level=strict_level)
        self.console_fn = console_fn
        if formatter:
            self.setFormatter(formatter)

    def _emit(self, record: logging.LogRecord) -> None:
        self.console_fn(self.format(record), extra=record.__dict__)
        self.flush()

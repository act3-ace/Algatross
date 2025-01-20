"""Module of yaml loaders."""

import collections
import copy
import importlib
import json
import os

from pathlib import Path
from typing import IO, Any
from urllib.parse import unquote, urlparse

import ray

import importlib_metadata
import yaml

from yaml.constructor import ConstructorError
from yaml.nodes import SequenceNode

from algatross.utils.types import ConstructorData


def get_pkgs_paths() -> set[Path]:
    """Return set of full paths to all pkgs.

    Returns
    -------
    set[Path]
        The set of full paths to all pkgs.
    """
    pkgs_paths = set()
    for distribution in importlib_metadata.distributions():
        if distribution.origin:
            pkgs_paths.add(Path(unquote(urlparse(distribution.origin.url).path)))
        else:
            pkgs_paths.add(Path(str(distribution.locate_file(""))))
    return pkgs_paths


def get_distribution_path(distribution: importlib_metadata.Distribution) -> Path:
    """Return set of full path to top level of a package.

    Parameters
    ----------
    distribution : importlib_metadata.Distribution
        The package distribution.

    Returns
    -------
    Path
        The path to the package.
    """
    dist_name = importlib_metadata.Prepared.normalize(distribution.name)
    if distribution.origin:
        return Path(unquote(urlparse(distribution.origin.url).path)) / Path(dist_name)
    return (Path(str(distribution.locate_file("")))) / Path(dist_name)


class Loader(yaml.SafeLoader):
    """
    YAML Loader with :yaml:`!include` constructor.

    Parameters
    ----------
    stream : IO
        The input stream to load.
    """

    cwd_path: Path  #: The path to the current working directory.
    """The path to the current working directory afterwards."""

    def __init__(self, stream: IO) -> None:
        try:
            self.root_path = Path(os.path.split(stream.name)[0])
        except AttributeError:
            self.root_path = Path(os.path.curdir)

        self.cwd_path = Path.cwd()

        self._pkg_paths = get_pkgs_paths()
        self._distribution = None

        for site_path in self._pkg_paths:
            if site_path in self.root_path.parents:
                str(Path(*self.root_path.relative_to(site_path).parts[:1]))
                if site_path != self.cwd_path:
                    self._distribution = importlib_metadata.distribution(str(Path(*self.root_path.relative_to(site_path).parts[:1])))

        super().__init__(stream)

        self._include_mapping: dict = {}
        self.deep_construct = True

    def build_file_path(self, node: yaml.ScalarNode, absolute: bool = False) -> tuple[Path, importlib_metadata.Distribution]:
        """Handle loading the strings behind the various corl !include extensions to handle complex loading.

        corl performs a prioritized search in the following order to
        locate a file and build the associated path. The first search
        that returns a valid file will be return the path. Subsequent
        searches will not be performed.

        1. Relative paths
            if an include filepath begins with a '.' the loader will use the file location as a basis
            no searches are performed beyond relative locations of current file.

            - file path: ``config/tasks/docking1d/agents/main_agent.yml``
                example1: :yaml:`!include ./glues/glue_set1.yml`
                would be resolved to ``config/tasks/docking1d/agents/glues/glue_set1.yml``

        2. CWD paths
            corl will look for files pathed from the current working directory

            - example: :yaml:`!include config/tasks/docking1d/agents/glue_set1.yml`
                would be resolved to load ``<path to working directory>/config/tasks/docking1d/agents/glue_set1.yml``

        3. Module paths
            corl sets a module path for each yaml conf. The module path is set to the cwd
            path, unless the current yml conf resides in a site_pkgs (i.e. is installed).
            In that case the module path will be set the the site package directory of the module.

            - example: ``/home/user/src/corl/config/tasks/docking1d/agents/glue_set1.yml``
                module_path: ``cwd``
            - example: ``opt/conda/lib/python3.10/site-packages/<pkg>/config/tasks/docking1d/agents/glue_set1.yml``
                module_path: ``opt/conda/lib/python3.10/site-packages/<pkg>``

            when parsing a module input config include path can now be defined relative the module
            for example in the config opt/conda/lib/python3.10/site-packages/<pkg>/config/sample_config1.yml

            - example: :yaml:`!include config/sample_config2.yml`
                would be resolved to load ``opt/conda/lib/python3.10/site-packages/<pkg>/config/sample_config2.yml``

        4. Site-package paths
            corl will search site package for includes not found in other paths

            - example: :yaml:`!include <pkg>/config/sampple_config1.yml`
                would be resolved to load ``opt/conda/lib/python3.10/<pkg>/config/sampple_config1.yml``

        Parameters
        ----------
        node : yaml.ScalarNode
            The yaml node.
        absolute : bool, optional
            Whether to resolve absolute paths, :data:`python:False`.

        Returns
        -------
        tuple[Path, importlib_metadata.Distribution]
            The (maybe absolute) path to the file and the package distribution.

        Raises
        ------
        RuntimeError
            If the filename was not found.
        """
        loader_string = self.construct_scalar(node)
        assert isinstance(loader_string, str)  # noqa: S101

        filename, distribution = self._find_file(loader_string)

        if not filename:
            msg = f"{loader_string} not found when yaml parsing"
            raise RuntimeError(msg)

        return filename.absolute() if absolute else filename.resolve(), distribution

    def _find_file(self, loader_string: str) -> tuple[Path | None, importlib_metadata.Distribution | None]:
        # Check if file is relative to current file
        if loader_string.startswith("."):
            tmp_path = Path(self.root_path, loader_string)
            if tmp_path.is_file():
                return tmp_path, self._distribution
            return None, None

        # Check if file is in current repo
        cwd_path = Path(self.cwd_path, loader_string)
        if cwd_path.is_file():
            return cwd_path, None

        # Check if file is in current module
        if self._distribution:
            module_path = Path(get_distribution_path(self._distribution), loader_string)
            if module_path.is_file():
                return module_path, self._distribution

        # Check if file is in a different site pkg module
        for pkg_path in self._pkg_paths:
            site_path = Path(pkg_path, loader_string)
            if site_path.is_file():
                return site_path, importlib_metadata.distribution(str(Path(*site_path.relative_to(pkg_path).parts[:1])))

        return None, None

    def construct_str(self, node: yaml.Node) -> Any:  # noqa: ANN401
        """Construct the path to the file as a string or returns the value of the node.

        Parameters
        ----------
        node : yaml.Node
            The yaml node to parse.

        Returns
        -------
        Any
            The path to the referenced file or the value of the yaml node itself.
        """
        # Implement custom string handling here
        if Path(node.value).suffix:
            new_file_path, _ = self._find_file(node.value)
            if new_file_path:
                return str(new_file_path)
        return node.value

    def construct_document(self, node: yaml.Node):  # noqa: ANN201, D102
        data = super().construct_document(node)
        self.deep_construct = True
        return data

    def construct_python_tuple(self, node: yaml.SequenceNode) -> tuple:
        """Add in the capability to process tuples in yaml files.

        Parameters
        ----------
        node : yaml.SequenceNode
            The sequence node to tuple-ize.

        Returns
        -------
        tuple
            The node as a python :class:`tuple`.
        """
        return tuple(self.construct_sequence(node))

    def construct_sequence(self, node: yaml.SequenceNode, deep: bool = False) -> list:
        """Construct a sequence from a YAML sequence node.

        This method extends yaml.constructor.BaseConstructor.construct_sequence by adding support for children with the tag
        :yaml:`!include-extend`.  Any object with this tag should be constructable to produce a sequence of objects.  Even though
        :yaml:`!include-extend` is a tag on the child object, the sequence produced by this child is not added as a single element to the
        sequence being produced by this method.  Rather, the output sequence is extended with this list.  Any children with other tags are
        appended into the list in the same manner as yaml.constructor.BaseConstructor.construct_sequence.

        Parameters
        ----------
        node : yaml.SequenceNode
            The yaml node to construct
        deep : bool, optional
            Whether to deep-construct the node, :data:`python:False`

        Returns
        -------
        list
            The constructed sequence.

        Raises
        ------
        ConstructorError
            If ``include-extend`` is not a sequence node.

        Examples
        --------
        >>> Loader.add_constructor("!include-extend", construct_include)
        >>> with open("primary.yml", "r") as fp:
        >>>     config = yaml.load(fp, Loader)
        {
            "root": {
                "tree1": ["apple", "banana", "cherry"],
                "tree2": [
                    {"type": "int", "value": 3},
                    {"type": "float", "value": 3.14},
                    {"type": "str", "value": "pi"},
                ],
                "tree3": [
                    "date",
                    "elderberry",
                    "fig",
                    "grape",
                    "honeydew",
                    "jackfruit",
                    "kiwi",
                    "lemon",
                    "mango",
                ],
            }
        }

        .. code-block:: yaml
            :caption: primary.yml

            root:
                tree1:
                    - apple
                    - banana
                    - cherry
                tree2:
                    - type: int
                      value: 3
                    - type: float
                      value: 3.14
                    - type: str
                      value: pi
                tree3:
                    - date
                    - elderberry
                    - !include-extend secondary.yml
                    - mango

        .. code-block:: yaml
            :caption: secondary.yml

            - fig
            - grape
            - honeydew
            - jackfruit
            - kiwi
            - lemon
        """
        if not isinstance(node, SequenceNode):
            return super().construct_sequence(node, deep=deep)

        output: list = []
        for child in node.value:
            this_output = self.construct_object(child, deep=deep)
            if child.tag == "!include-extend":
                if not isinstance(this_output, collections.abc.Sequence):
                    raise ConstructorError(
                        None,
                        None,
                        f"expected a sequence returned by 'include-extend', but found {type(this_output).__name__}",
                        child.start_mark,
                    )
                output.extend(this_output)
            else:
                output.append(this_output)

        return output

    def flatten_mapping(self, node: yaml.Node):  # noqa: D102
        merge = []
        index = 0
        while index < len(node.value):
            key_node, value_node = node.value[index]
            if key_node.tag == "tag:yaml.org,2002:merge":
                del node.value[index]
                if isinstance(value_node, yaml.MappingNode):
                    self.flatten_mapping(value_node)
                    merge.extend(value_node.value)
                elif isinstance(value_node, yaml.SequenceNode):
                    submerge = []
                    for subnode in value_node.value:
                        if not isinstance(subnode, yaml.MappingNode):
                            msgs = (
                                "while constructing a mapping",
                                node.start_mark,
                                f"expected a mapping for merging, but found {subnode.id}",
                                subnode.start_mark,
                            )
                            raise ConstructorError(*msgs)
                        self.flatten_mapping(subnode)
                        submerge.append(subnode.value)
                    submerge.reverse()
                    for value in submerge:
                        merge.extend(value)
                else:
                    # TODO: FIGURE OUT HOW TO DUMP AND ACCESS THE BASE NODE!!!!
                    # if value_node.tag == '!include-direct':
                    #     filename = os.path.realpath(os.path.join(self._root, self.construct_scalar(value_node)))  # noqa: ERA001
                    #     d = yaml.dump(self._include_mapping[filename])  # noqa: ERA001
                    #     # for k, v in .items():
                    #     #     sk = yaml.ScalarNode(tag='tag:yaml.org,2002:str', value=str(k))  # noqa: ERA001
                    #     #     if isinstance(v, int):
                    #     #         sv = yaml.ScalarNode(tag='tag:yaml.org,2002:int', value=str(v))  # noqa: ERA001
                    #     #     else:  # noqa: ERA001
                    #     #         sv = yaml.ScalarNode(tag='tag:yaml.org,2002:seq', value=str(v))  # noqa: ERA001
                    #     #     merge.extend([(sk, sv)])  # noqa: ERA001
                    # else:  # noqa: ERA001
                    msgs = (
                        "while constructing a mapping",
                        node.start_mark,
                        f"expected a mapping or list of mappings for merging, but found {value_node.id}",
                        value_node.start_mark,
                    )
                    raise ConstructorError(*msgs)
            elif key_node.tag == "tag:yaml.org,2002:value":
                key_node.tag = "tag:yaml.org,2002:str"
                index += 1
            else:
                index += 1
        if merge:
            node.value = merge + node.value


def construct_function(loader: yaml.SafeLoader, node: yaml.Node) -> Any:  # noqa: ANN401, ARG001
    """Import a function or class from the given node.

    Parameters
    ----------
    loader : yaml.SafeLoader
        The yaml loader (unused)
    node : yaml.Node
        The node to construct

    Returns
    -------
    Any
        The python object
    """
    parts = node.value.split(".")
    module = ".".join(parts[:-1])
    fn = parts[-1]
    module = importlib.import_module(module)  # type: ignore[assignment]
    return getattr(module, fn)


def include_file(filename: Path) -> Any:  # noqa: ANN401
    """Include nodes from another yaml file.

    Parameters
    ----------
    filename : Path
        The path of the yaml file to load

    Returns
    -------
    Any
        The loaded contents of the file.
    """
    extension = filename.suffix

    with open(filename, encoding="utf-8") as fp:  # noqa: PTH123
        if extension in {".yaml", ".yml"}:
            return yaml.load(fp, Loader)  # noqa: S506
        elif extension == ".json":  # noqa: RET505
            return json.load(fp)
        else:
            return "".join(fp.readlines())


def construct_include(loader: Loader, node: yaml.ScalarNode) -> Any:  # noqa: ANN401
    """Include file referenced at node.

    Parameters
    ----------
    loader : Loader
        The yaml loader
    node : yaml.ScalarNode
        The noad to construct

    Returns
    -------
    Any
        The constructed node contents
    """
    filename, _ = loader.build_file_path(node)
    return include_file(filename)


def construct_constructordata(loader: Loader, node: yaml.MappingNode) -> ConstructorData:
    """Create a ConstructorData object from the node.

    Parameters
    ----------
    loader : Loader
        The yaml loader
    node : yaml.MappingNode
        The node to load as a :class:`~algatross.utils.types.ConstructorData`

    Returns
    -------
    ConstructorData
        The constructor data object.
    """
    return ConstructorData(**loader.construct_mapping(node))  # type: ignore[misc]


def construct_path(loader: Loader, node: yaml.ScalarNode) -> Path:
    """Construct file path associated with node.

    Parameters
    ----------
    loader : Loader
        The yaml loader
    node : yaml.ScalarNode
        The node containing the path.

    Returns
    -------
    Path
        The file path
    """
    filename, _ = loader.build_file_path(node)
    return filename


def construct_include_direct(loader: Loader, node: yaml.ScalarNode) -> Any:  # noqa: ANN401
    """Include file referenced at node.

    Parameters
    ----------
    loader : Loader
        The yaml loader
    node : yaml.ScalarNode
        The node to include

    Returns
    -------
    Any
        The included yaml contents
    """
    filename, _file_module = loader.build_file_path(node)
    extension = filename.suffix

    with open(filename, encoding="utf-8") as fp:  # noqa: PTH123
        if extension in {".yaml", ".yml"}:
            temp = yaml.load(fp, Loader)  # noqa: S506
            loader._include_mapping[filename] = temp  # noqa: SLF001
            return temp
        elif extension == ".json":  # noqa: RET505
            return json.load(fp)
        else:
            return "".join(fp.readlines())


def construct_merge_dict(loader: Loader, node: yaml.SequenceNode) -> Any:  # noqa: ANN401
    """Merge two dictionaries.

    Parameters
    ----------
    loader : Loader
        The yaml loader
    node : yaml.SequenceNode
        The node containing the mappings to merge

    Returns
    -------
    Any
        The merged dictionaries
    """
    sequence = loader.construct_sequence(node)
    return apply_patches(sequence)


Loader.add_constructor("!include", construct_include)
Loader.add_constructor("!include-direct", construct_include_direct)
Loader.add_constructor("!include-extend", construct_include)
Loader.add_constructor("!function", construct_function)
Loader.add_constructor("!ConstructorData", construct_constructordata)
Loader.add_constructor("tag:yaml.org,2002:python/tuple", Loader.construct_python_tuple)
Loader.add_constructor("tag:yaml.org,2002:str", Loader.construct_str)
Loader.add_constructor("!path", construct_path)
Loader.add_constructor("!merge", construct_merge_dict)


def apply_patches(config: dict | list[dict | None]) -> dict:
    """Update the base setup with patches.

    Parameters
    ----------
    config : dict | list[dict | None]
        The base and patch if list, else dict.

    Returns
    -------
    dict | list
        The combined dict.
    """

    def merge(source: dict, destination: dict) -> dict:
        """
        Run me with nosetests --with-doctest file.py.

        Parameters
        ----------
        source : dict
            The source dictionary to merged into ``destination``.
        destination : dict
            The destination dictionary to be deep-merged with values from source.

        Returns
        -------
        dict
            The ``destination`` recursively deep-updated with values in ``source``

        Examples
        --------
        >>> a = {"first": {"all_rows": {"pass": "dog", "number": "1"}}}
        >>> b = {"first": {"all_rows": {"fail": "cat", "number": "5"}}}
        >>> merge(b, a) == {"first": {"all_rows": {"pass": "dog", "fail": "cat", "number": "5"}}}
        True
        """
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                merge(value, node)
            else:
                destination[key] = value

        return destination

    if isinstance(config, list):
        config_new = copy.deepcopy(config[0])
        for item in config[1:]:
            if item is not None:
                config_new = merge(item, config_new)
        return config_new

    return config


def load_config(config_file: str) -> dict:
    """Load the configuration from a yaml file.

    Parameters
    ----------
    config_file : str
        The path of the file to load

    Returns
    -------
    dict
        The parsed yaml contents in ``config_file``
    """
    with Path(config_file).open("rb") as f:
        return yaml.load(f, Loader=Loader)  # noqa: S506


if __name__ == "__main__":
    with Path("config/simple_spread/algatross.yml").open("rb") as f:
        stuff = yaml.load(f, Loader=Loader)  # noqa: S506
    print(stuff)
    island_constructors = collections.defaultdict(list)
    mainland_constructors = collections.defaultdict(list)
    for isl_data in stuff["islands"]:
        if stuff["seed"]:
            isl_data["problem_constructor"].config["seed"] = isl_data["problem_constructor"].config.get("seed") or stuff["seed"]
            isl_data["algorithm_constructor"].config["seed"] = isl_data["algorithm_constructor"].config.get("seed") or stuff["seed"]
            isl_data["population_constructor"].config["seed"] = isl_data["population_constructor"].config.get("seed") or stuff["seed"]
        island_constructors["constructors"].append(isl_data["island_constructor"])
        island_constructors["problem_constructors"].append(isl_data["problem_constructor"])
        island_constructors["algorithm_constructors"].append(isl_data["algorithm_constructor"])
        island_constructors["population_constructors"].append(isl_data["population_constructor"])
    for isl_data in stuff["mainlands"]:
        if stuff["seed"]:
            isl_data["problem_constructor"].config["seed"] = isl_data["problem_constructor"].config.get("seed") or stuff["seed"]
            isl_data["algorithm_constructor"].config["seed"] = isl_data["algorithm_constructor"].config.get("seed") or stuff["seed"]
            isl_data["population_constructor"].config["seed"] = isl_data["population_constructor"].config.get("seed") or stuff["seed"]
        mainland_constructors["constructors"].append(isl_data["island_constructor"])
        mainland_constructors["problem_constructors"].append(isl_data["problem_constructor"])
        mainland_constructors["algorithm_constructors"].append(isl_data["algorithm_constructor"])
        mainland_constructors["population_constructors"].append(isl_data["population_constructor"])
    topology = stuff["topology_constructor"].construct()
    archipelago = stuff["archipelago_constructor"]
    archipelago.config["topology"] = topology
    archipelago.config["island_constructors"] = island_constructors
    archipelago.config["mainland_constructors"] = mainland_constructors
    archipelago = archipelago.constructor.remote(**archipelago.config)

    ray.get(archipelago.evolve.remote(stuff["island_iterations"], stuff["mainland_iterations"], stuff["epochs"]))

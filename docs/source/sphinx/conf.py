# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import math
import os
import re
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from matplotlib import __version__ as mpl_version
from pandas import __version__ as pandas_version

from numpy import __version__ as numpy_version

from ray import __version__ as ray_version

from torch import __version__ as torch_version

from PIL import __version__ as pillow_version
from gymnasium import __version__ as gym_version
from pettingzoo import __version__ as pz_version
from scipy import __version__ as scipy_version
from sphinx.application import Sphinx
from yaml import safe_load

version = ""
announcment = ""

# REGEX Patterns
TYPE_PATTERN = re.compile(r"^(?:\s*):(?:r)?(?:type|value)(?:\s\w*)?: {0,2}(?P<types>.*(?=[,|]?))")
RAISES_PATTERN = re.compile(r"^(?:\s*):(?:raises) (?P<types>.*(?=[,|]?)):")
SIGNATURE_PATTERN = re.compile(r"^\s*.. py:\w+::\s+.*\((?P<types>.*(?=[,|]?))\)")
SIGNATURE_PATTERN = re.compile(r"(?P<types>(?:\.|\w)+)")
EQ_PATTERN = re.compile(r".*((?= = )(?:.*)).*")
NEWS_HEADER_PATTERN = re.compile(r"^newsHeader:\s?(?P<header>.*)$")
PROJECT_URL = "https://git.act3-ace.com/stalwart/ascension/mo-marl/"
PAGES_BASE_URL = "https://stalwart.git.act3-ace.com/ascension/mo-marl/"

PAGES_PREFIX = os.environ.get("PAGES_PREFIX", "")

print(f"Using PAGES_PREFIX: {PAGES_PREFIX}")

# Get the version from the version file
with Path("../../../VERSION").open("r", encoding="utf-8") as f:
    version = f.readline().strip("\n")

# # Get the type aliases from the type alias file
with Path("type_aliases.yaml").open("r", encoding="utf-8") as f:
    type_aliases = safe_load(f)

TAG_URL = f"{PROJECT_URL}/-/tree/v{version}/"

# Parse the news file for Announcments
with Path("../hugo/content/news.md").open("r", encoding="utf-8") as f:
    for line in f:
        if (news_header := NEWS_HEADER_PATTERN.match(line)) and (announcment := news_header.group("header").strip('"')):
            announcment = (
                f'<div class="news-container"><div class="news-title"><a href="/ascension/mo-marl/news">{announcment}</a></div></div>'
            )
            break


# Defune the settings for links and intersphinx
def majmin_ver(version: str) -> str:
    return ".".join(version.split(".")[:2])


depends = {
    "python": {"url": ".".join(map(str, sys.version_info[:2])), "pip": ".".join(map(str, sys.version_info))},
    "ray": {"url": f"releases-{ray_version}", "pip": ray_version},
    "numpy": {"url": majmin_ver(numpy_version), "pip": numpy_version},
    "pytorch": {"url": majmin_ver(torch_version), "pip": torch_version},
    "scipy": {"url": "devdocs" if "rc" in scipy_version else f"scipy-{scipy_version}", "pip": scipy_version},
    "matplotlib": {"url": "devdocs", "pip": mpl_version},
    "pandas": {"url": majmin_ver(pandas_version), "pip": pandas_version},
    "pettingzoo": {"url": pz_version, "pip": pz_version},
    "PIL": {"url": "stable", "pip": pillow_version},
    "rich": {"url": "stable", "pip": ""},
    "gymnasium": {"url": f"v{gym_version}", "pip": gym_version},
}


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MO-MARL"
copyright = f"{datetime.now(timezone.utc).year}, Air Force Research Laboratory"  # noqa: A001
author = "Air Force Research Laboratory"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "myst_parser",
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "autoapi.extension",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.linkcode",  # Support for adding links to source code under each objects header
    "sphinxemoji.sphinxemoji",  # Support for using emojis
    "sphinx_togglebutton",  # Support for dropdowns
    "matplotlib.sphinxext.plot_directive",
    # "numpydoc",
    "docs.source.sphinx.ext.python_domain",
    "docs.source.sphinx.ext.fontawesome_cards",
    "docs.source.sphinx.ext.linkcode",
]

templates_path = ["_templates"]
# don't include any static CSS -> we will let HUGO handle styles
exclude_patterns: list[str] = [
    "**/_static/css/*",
    "**/pagefind/*",
    "**/js/bundle.js",
]
modindex_common_prefix: list[str] = ["algatross."]
# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"
# define RST to insert to every page when building
rst_prolog = """
.. role:: python(code)
    :language: python

.. role:: yaml(code)
    :language: yaml

"""
numfig = True  # auto number figures, tables, code, etc. if they include :caption:
add_module_names = False
add_function_parentheses = False
python_display_short_literal_types = True
pygments_style = "manni"
pygments_dark_style = "dracula"

# If you want to use Markdown files with extensions other than .md, adjust the
# source_suffix variable. The following example configures Sphinx to parse all
# files with the extensions .md and .txt as Markdown:
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html
html_theme = "pydata_sphinx_theme"
html_logo = "../hugo/static/images/text-logo-2.svg"
html_favicon = "_static/images/favicon.ico"
html_baseurl = urljoin(PAGES_BASE_URL, PAGES_PREFIX)
# prefix the dummy paths with '../' so they point to the actual locations in the Hugo build directory
html_alt_link_options = {}
html_use_modindex = True
html_copy_source = False
html_static_path = ["_static"]
html_css_files = []
html_js_files = [
    "js/fa-custom-icons.js",
    "js/hugo_compat.js",
]
html_context = {
    "gitlab_url": "https://git.act3-ace.com",  # or your self-hosted GitLab
    "gitlab_user": "stalwart/ascension",
    "gitlab_repo": "mo-marl",
    "gitlab_version": "main",
    "doc_path": "docs/source/sphinx",
}
html_sidebars = {
    "**": ["components/search-button-field.html", "sidebar-nav-bs"],
}
html_theme_options = {
    "logo": {
        "link": "/ascension/mo-marl",
    },
    "external_links": [
        {"name": "Home", "url": "/ascension/mo-marl/"},
        {"name": "Installation", "url": "/ascension/mo-marl/installation"},
        {"name": "API Documentation", "url": "/ascension/mo-marl/api"},
        {"name": "User Guide", "url": "/ascension/mo-marl/user_guide"},
        {"name": "Developer Guide", "url": "/ascension/mo-marl/developer_guide"},
    ],
    "icon_links": [
        {
            "name": "Gitlab",
            "url": "https://git.act3-ace.com/stalwart/ascension/mo-marl/",  # required
            "icon": "fa-brands fa-gitlab",
            "type": "fontawesome",
        },
        {
            "name": "Gitlab",
            "url": "https://github.com/gresavage/mo-marl/",  # required
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Docker",
            "url": "https://reg.git.act3-ace.com/stalwart/ascension/mo-marl/",  # required
            "icon": "fa-brands fa-docker",
            "type": "fontawesome",
        },
        {
            "name": "ruff",
            "url": "https://astral.sh/docs/ruff/",  # required
            "icon": "fa-custom fa-ruff",
            "type": "fontawesome",
            "header": False,
            "footer": True,
        },
        {
            "name": "uv",
            "url": "https://astral.sh/docs/uv/",  # required
            "icon": "fa-custom fa-uv",
            "type": "fontawesome",
            "header": False,
            "footer": True,
        },
    ],
    "announcement": announcment,
    "header_links_before_dropdown": 7,
    "footer_center": ["components/footer-icon-links"],
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "components/icon-links"],
    "navbar_persistent": [],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "primary_sidebar_end": ["indices", "sidebar-ethical-ads"],
    "navbar_align": "right",
    "use_edit_page_button": True,
    "article_header_start": "components/breadcrumbs.html",
    "pygments_light_style": "manni",
    "pygments_dark_style": "dracula",
    # "switcher": {  # noqa: ERA001
    #     "json_url": "version_list.json",  # noqa: ERA001
    #     "version_match": version,  # noqa: ERA001
    # },
    # "check_switcher": False,  # noqa: ERA001
    # "footer_end": ["theme-version", "version-switcher"],  # noqa: ERA001
}

# Mappings for Intersphinx to pull-in other projects. These projects must have
# been build using Sphinx to work with Intersphinx
# See: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#confval-intersphinx_mapping
# The examples below link directly to the documentations of the actual versions used by this project
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{depends['python']['url']}/", None),
    "pytorch": (f"https://pytorch.org/docs/{depends['pytorch']['url']}/", None),
    # scipy is silly about their dev pages
    "scipy": (
        f"https://{'docs.scipy.org/doc' if depends['scipy']['url'] != 'devdocs' else 'scipy.github.io'}/{depends['scipy']['url']}/",
        None,
    ),
    "ray": (f"https://docs.ray.io/en/{depends['ray']['url']}/", None),
    "matplotlib": (f"https://matplotlib.org/{depends['matplotlib']['url']}/", None),
    "PIL": (f"https://pillow.readthedocs.io/en/{depends['PIL']['url']}/", None),
    "numpy": (f"https://numpy.org/doc/{depends['numpy']['url']}/", None),
    "pandas": (f"https://pandas.pydata.org/pandas-docs/version/{depends['pandas']['url']}/", None),
    "gymnasium": (f"https://gymnasium.farama.org/{depends['gymnasium']['url']}/", None),
    "pettingzoo": (f"https://pettingzoo.farama.org/{depends['pettingzoo']['url']}/", None),
    "rich": (f"https://rich.readthedocs.io/en/{depends['rich']['url']}/", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tf2_py_objects.inv",
    ),
    "tensorflow_probability": (
        "https://www.tensorflow.org/probability/api_docs/python",
        "https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tfp_py_objects.inv",
    ),
}


# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------
autodoc_typehints = "both"  # typehint both the signature and docstring
autodoc_type_aliases = {}
for _k, v in filter(lambda it: it[0] != "builtins", type_aliases.items()):
    autodoc_type_aliases |= v
autoclass_content = "class"  # both the class docstring and __init__ docstrings will be used
autodoc_class_signature = "separated"  # separate the signature
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "undoc-members": True,
    "show-inheritance": True,
    "private-members": False,
    "special-members": False,
    "exclude-members": ["__init__, __new__"],
    "inherited-members": False,
}


# -----------------------------------------------------------------------------
# Autoapi
# -----------------------------------------------------------------------------
autoapi_dirs = ["../../../algatross"]
autoapi_ignore = ["*test_*_experiment*"]
autoapi_template_dir = "_templates"
autoapi_root = "generated"
autoapi_member_order = "groupwise"
autoapi_keep_files = True
autoapi_own_page_level = "function"
autoapi_python_class_content = "class"
autoapi_options = [
    "members",
    "undoc-members",
    # "show-inheritance",  # auto-referencing fails because of using short names
    "show-inheritance-diagram",
    "show-module-summary",
]


# -----------------------------------------------------------------------------
# MyST
# -----------------------------------------------------------------------------
myst_title_to_header = True  # true to set header to title
myst_heading_anchors = 3
myst_enable_extensions = {
    "amsmath",
    "dollarmath",
    "strikethrough",
    "linkify",
    "colon_fence",
    "deflist",
}


# -----------------------------------------------------------------------------
# Autosectionlabel
# -----------------------------------------------------------------------------
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 1

# -----------------------------------------------------------------------------
# Sphinx Copybutton
# -----------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.{3,}: | {5,8}: "
copybutton_prompt_is_regexp = True

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
plot_pre_code = """
import numpy as np
np.random.seed(0)
"""
plot_include_source = True
plot_formats = [("png", 100), "pdf"]

phi = (math.sqrt(5) + 1) / 2

plot_rcparams = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3 * phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

# -----------------------------------------------------------------------------
# NitPicky
# -----------------------------------------------------------------------------
suppress_warnings = []
nitpick_ignore = [("py:class", "type"), ("py:class", "Ellipsis")]
nitpick_ignore_regex = [
    (r"py:.*", r"optional"),
    (r"py:.*", r"yaml\..*"),
    (r"py:.*", r"algatross\.utils\.types\..*"),
    (r"py:.*", r"dask\..*"),
    (r"py:.*", r"ribs\..*"),
    (r"py:.*", r"ray\..*"),
    (r"py:.*", r"torch\..*"),
    (r"py:.*", r"np\..*"),
    (r"py:.*", r"numpy\..*"),
    (r"py:.*", r"importlib_metadata\..*"),
    (r"py:.*", r"json\..*"),
    (r"py:.*", r"_collections(_abc)?\..*"),
    (r"py:.*", r"treelib\..*"),
    (r"py:.*", r"pettingzoo\..*"),
    (r"py:.*", r"gymnasium\..*"),
    (r"py:.*", r"safe_autonomy_sim(ulation|s)\..*"),
    (r"py:.*", r"sim\..*"),
    (r"py:.*", r"harl\..*"),
    (r"py:.*", r"pymoo\..*"),
    (r"py:.*", r"logging\..*"),
    (r"py:.*", r"^T$"),
]

# -----------------------------------------------------------------------------
# Napoleon
# -----------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_type_aliases = {}
for v in type_aliases.values():
    napoleon_type_aliases |= v


def process_docstring(app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: list[str]):
    """Replace the short refs in docstrings with aliases from the yaml file."""
    for line_idx, line in enumerate(lines):
        if replace := _replace_types(line, TYPE_PATTERN, app.config.autodoc_type_aliases):
            lines[line_idx] = replace
            continue
        if replace := _replace_types(line, RAISES_PATTERN, app.config.autodoc_type_aliases):
            lines[line_idx] = replace
            continue


def _replace_types(line: str, matcher: re.Pattern, type_aliases: dict[str, str]) -> str:
    """Replace the types in the parsed docstrings - for some reason (probably AutoAPI) the automatic cross-refs aren't generated."""  # noqa: DOC201
    if type_match := matcher.match(line):
        replace_types = {}
        group = type_match.group("types")
        for group_type in set(group.split(" | ")):
            for ind_type in {dirty.strip() for dirty in group_type.replace("[", ",").replace("]", ",").split(",") if "~" not in dirty}:
                if replace_eq := EQ_PATTERN.match(ind_type):
                    line = line.replace(replace_eq.group(1), "")
                if not ind_type or ind_type == "optional" or ind_type in replace_types:
                    continue
                if replace_type := type_aliases.get(ind_type):
                    replace_types[ind_type] = f"{replace_type} "
        for in_type, out_type in replace_types.items():
            line = line.replace(in_type, out_type)
        return line
    return None


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    code_url = f"{urljoin(TAG_URL, filename)}.py?ref_type=tags"
    if not (viewcode_tags := info.get("viewcode_tags", None)):
        return code_url
    line_start, line_end = viewcode_tags[1:3]
    return f"{code_url}#L{line_start}-L{line_end}"


def setup(app: Sphinx):
    app.connect("autodoc-process-docstring", process_docstring)

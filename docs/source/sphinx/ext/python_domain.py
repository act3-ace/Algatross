from typing import TYPE_CHECKING, ClassVar

from docutils import nodes
from sphinx.domains.python import PythonDomain
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec

if TYPE_CHECKING:
    from sphinx.application import Sphinx


logger = logging.getLogger(__name__)


class PyCurrentClass(SphinxDirective):
    """Directive is just to tell Sphinx that we're documenting stuff in module foo, but links to module foo won't lead here."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: ClassVar[OptionSpec] = {}

    def run(self) -> list[nodes.Node]:
        # adds the directive argument as a class to the context
        classname = self.arguments[0].strip()
        if classname == "None":
            self.env.ref_context.pop("py:class", None)
        else:
            self.env.ref_context["py:class"] = classname
        return []


class CustomPythonDomain(PythonDomain):
    name = "py"

    def get_objects(self):
        # Modify the index entries as needed
        for obj in super().get_objects():
            # Yield just the object name for the index
            yield (obj[0], obj[1].split(".")[-1], *obj[2:])


def setup(app: "Sphinx"):
    # Add custom builder and domain utilities
    app.add_domain(CustomPythonDomain, override=True)
    app.add_directive_to_domain("py", "currentclass", PyCurrentClass)
    return {"parallel_read_safe": True}

from docutils import nodes
from sphinx_design.cards import DIRECTIVE_NAME_CARD, CardDirective
from sphinx_design.grids import DIRECTIVE_NAME_GRID_ITEM_CARD, GridItemCardDirective
from sphinx_design.shared import WARNING_TYPE, PassthroughTextElement, create_component, is_component

from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)


class fa_icon(nodes.General, nodes.Inline, nodes.Element):
    tagname = "i"


def visit_fa_icon(self, node: nodes.Element):
    self.body.append(self.starttag(node, "i"))


def depart_fa_icon(self, node: nodes.Element):
    self.body.append("</i>\n")


def get_fa_icon_node(icon_src: str, classes: list[str] | None = None, img_alt: str = "") -> nodes.Element:
    classes = classes or []
    extra_classes = icon_src.replace(":fontawesome:", "").replace("`", "").strip().split(" ")
    return fa_icon(
        classes=[*classes, *extra_classes],
        alt=img_alt,
    )


class FontawesomeCardDirective(CardDirective):
    @classmethod
    def create_card(  # noqa: PLR0915
        cls,
        inst: SphinxDirective,
        arguments: list | None,
        options: dict,
    ) -> nodes.Node:
        """Commandeer this method from Sphinx-Design to in order to support Fontawesome icons in cards."""
        card_classes = ["sd-card", "sd-sphinx-override"]
        if "width" in options:
            card_classes += [f"sd-w-{options['width'].rstrip('%')}"]
        card_classes += options.get("margin", ["sd-mb-3"])
        card_classes += [f"sd-shadow-{options.get('shadow', 'sm')}"]
        if "link" in options:
            card_classes += ["sd-card-hover"]
        card = create_component(
            "card",
            card_classes + options.get("text-align", []) + options.get("class-card", []),
        )
        inst.set_source_info(card)

        img_alt = options.get("img-alt") or ""

        container = card
        if "img-background" in options:
            if options["img-background"].startswith(":fontawesome:"):
                card.append(get_fa_icon_node(options["img-background"], ["sd-card-img"], img_alt=img_alt))
            else:
                card.append(
                    nodes.image(
                        uri=options["img-background"],
                        classes=["sd-card-img"],
                        alt=img_alt,
                    ),
                )
            overlay = create_component("card-overlay", ["sd-card-img-overlay"])
            inst.set_source_info(overlay)
            card += overlay
            container = overlay

        if "img-top" in options:
            if options["img-top"].startswith(":fontawesome:"):
                image_top = get_fa_icon_node(options["img-top"], ["sd-card-img-top", *options.get("class-img-top", [])], img_alt=img_alt)
            else:
                image_top = nodes.image(
                    "",
                    uri=options["img-top"],
                    alt=img_alt,
                    classes=["sd-card-img-top", *options.get("class-img-top", [])],
                )
            container.append(image_top)

        components = cls.split_content(inst.content, inst.content_offset)

        if components.header:
            container.append(
                cls._create_component(
                    inst,
                    "header",
                    options,
                    components.header[0],
                    components.header[1],
                ),
            )

        body = cls._create_component(
            inst,
            "body",
            options,
            components.body[0],
            components.body[1],
        )
        if arguments:
            title = create_component(
                "card-title",
                [
                    "sd-card-title",
                    "sd-font-weight-bold",
                    *options.get("class-title", []),
                ],
            )
            textnodes, _ = inst.state.inline_text(arguments[0], inst.lineno)
            title_container = PassthroughTextElement()
            title_container.extend(textnodes)
            inst.set_source_info(title_container)
            title.append(title_container)
            body.insert(0, title)
        container.append(body)

        if components.footer:
            container.append(
                cls._create_component(
                    inst,
                    "footer",
                    options,
                    components.footer[0],
                    components.footer[1],
                ),
            )

        if "img-bottom" in options:
            if options["img-bottom"].startswith(":fontawesome:"):
                image_bottom = get_fa_icon_node(
                    options["img-bottom"],
                    ["sd-card-img-bottom", *options.get("class-img-bottom", [])],
                    img_alt=img_alt,
                )
            else:
                image_bottom = nodes.image(
                    "",
                    uri=options["img-bottom"],
                    alt=img_alt,
                    classes=["sd-card-img-bottom", *options.get("class-img-bottom", [])],
                )
            container.append(image_bottom)

        if "link" in options:
            link_container = PassthroughTextElement()
            _classes = ["sd-stretched-link", "sd-hide-link-text"]
            _rawtext = options.get("link-alt") or options["link"]
            if options.get("link-type", "url") == "url":
                link = nodes.reference(
                    _rawtext,
                    "",
                    nodes.inline(_rawtext, _rawtext),
                    refuri=options["link"],
                    classes=_classes,
                )
            else:
                options = {
                    # TODO the presence of classes raises an error if the link cannot be found
                    "classes": _classes,
                    "reftarget": options["link"],
                    "refdoc": inst.env.docname,
                    "refdomain": "" if options["link-type"] == "any" else "std",
                    "reftype": options["link-type"],
                    "refexplicit": "link-alt" in options,
                    "refwarn": True,
                }
                link = addnodes.pending_xref(
                    _rawtext,
                    nodes.inline(_rawtext, _rawtext),
                    **options,
                )
            inst.set_source_info(link)
            link_container += link
            container.append(link_container)

        return card


class FontawesomeGridItemCardDirective(GridItemCardDirective):
    def run_with_defaults(self) -> list[nodes.Node]:
        """Commandeer this method from Sphinx-Design to in order to support Fontawesome icons in grid-item cards."""
        if not is_component(self.state_machine.node, "grid-row"):
            logger.warning(
                f"The parent of a 'grid-item' should be a 'grid-row' [{WARNING_TYPE}.grid]",
                location=(self.env.docname, self.lineno),
                type=WARNING_TYPE,
                subtype="grid",
            )
        column = create_component(
            "grid-item",
            [
                "sd-col",
                "sd-d-flex-row",
                *self.options.get("columns", []),
                *self.options.get("margin", []),
                *self.options.get("padding", []),
                *self.options.get("class-item", []),
            ],
        )
        card_options = {
            key: value
            for key, value in self.options.items()
            if key
            in {
                "width",
                "text-align",
                "img-background",
                "img-top",
                "img-bottom",
                "img-alt",
                "link",
                "link-type",
                "link-alt",
                "shadow",
                "class-card",
                "class-body",
                "class-title",
                "class-header",
                "class-footer",
                "class-img-top",
                "class-img-bottom",
            }
        }
        if "width" not in card_options:
            card_options["width"] = "100%"
        card_options["margin"] = []
        card = FontawesomeCardDirective.create_card(self, self.arguments, card_options)
        column += card
        return [column]


def setup(app: "Sphinx"):
    # Add fontawesome stuff
    app.add_directive(DIRECTIVE_NAME_CARD, FontawesomeCardDirective, override=True)
    app.add_directive(DIRECTIVE_NAME_GRID_ITEM_CARD, FontawesomeGridItemCardDirective, override=True)
    app.add_node(fa_icon, html=(visit_fa_icon, depart_fa_icon))

    return {"parallel_read_safe": True}

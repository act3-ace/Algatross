+++
date = '2024-12-03T18:10:18-05:00'
draft = false
title = 'Ruff'
+++

[ruff](http://docs.astral.sh/ruff) is a python linter and formatter written in Rust. It is extremely fast and ensures all developers produce consistent code.

The easiest way to integrate with ruff is through an IDE such as [VS Code](https://code.visualstudio.com/) and the official [ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

In most cases, ruff will auto-fix linting and formatting errors for you. It's fast enough to do this even as you type. This is achieved with the following Settings in VS Code (`settings.json`)

```json
{
    "[python]": {
        "editor.formatOnType": true,
        "editor.renderWhitespace": "trailing",
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "always",
            "source.organizeImports.ruff": "always"
        },
    },
    "ruff.showNotifications": "always",
    "ruff.lint.args": [
        "--config",
        "pyproject.toml"
    ],
}
```

Additionally, if using a toml linter such as [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml), set the following in `settings.json`

```json
{
    "evenBetterToml.schema.catalogs": [
        "https://www.schemastore.org/api/json/catalog.json",
        "https://raw.githubusercontent.com/astral-sh/ruff/main/ruff.schema.json"
    ],
}
```

## Further Reading

* [Core Tools]({{< ref "/developer_guide/core" >}})
* [Linting Tools]({{< ref "/developer_guide/lint" >}})
* [Testing Tools]({{< ref "/developer_guide/test" >}})
* [Documentation Tools]({{< ref "/developer_guide/docs" >}})
* [Pipeline Tools]({{< ref "/developer_guide/pipeline" >}})

---
title: uv
---

[uv](https://docs.astral.sh/uv/) is essential for any development work in MO-MARL. It manages the entire project and ensures your python environment is up-to-date.

Once installed, you can set up a virtual development environment using the command below

```bash
uv sync --frozen --all-extras
```

By default, this will install the core dependencies as well as the following groups `dev`, `test`, `lint`.

The `lint` group contains the linter dependencies such as [ruff](https://docs.astral.sh/ruff/) and [mypy](https://mypy.readthedocs.io/en/stable/), while the `test` group contains the testing dependencies such as [pytest](https://docs.pytest.org/en/stable/).

## Further Reading

* [Core Tools]({{< ref "/developer_guide/core" >}})
* [Linting Tools]({{< ref "/developer_guide/lint" >}})
* [Testing Tools]({{< ref "/developer_guide/test" >}})
* [Documentation Tools]({{< ref "/developer_guide/docs" >}})
* [Pipeline Tools]({{< ref "/developer_guide/pipeline" >}})

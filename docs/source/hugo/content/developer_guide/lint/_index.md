---
title: Linting Tools
---
<!-- markdownlint-disable MD034 -->

Linters ensure code from different developers conforms to consistent formatting rules and coding best practices.

Ruff is used for both formatting and linting, and MyPy is used for type-checking. It is strongly encouraged to run these tools on your codebase before commiting (see: [pre-commit]({{< ref "/developer_guide/core/pre_commit" >}})) in order to ensure your code meets the development standards.

{{< grid columns="1 2 2 3" >}}
[[item]]
type = "card"
title = "Ruff"
link = "{{< ref "ruff" >}}"
body = '''
{{< image >}}
src = "https://github.com/astral-sh/ruff/raw/refs/heads/main/docs/assets/bolt.svg"
height = "64"
align = 'right'
loading = 'lazy'
{{< /image >}}
An extremely fast Python linter and code formatter, written in Rust.
'''

[[item]]
type = "card"
title = "MyPy"
link = "{{< ref "mypy" >}}"
body = '''
{{< image >}}
src = "https://github.com/python/mypy/raw/refs/heads/master/docs/source/mypy_light.svg"
height = "64"
align = 'right'
loading = 'lazy'
{{< /image >}}
Optional static typing for Python
'''

{{< /grid >}}

## Further Reading

* [Core Tools]({{< ref "/developer_guide/core" >}})
* [Linting Tools]({{< ref "/developer_guide/lint" >}})
* [Testing Tools]({{< ref "/developer_guide/test" >}})
* [Documentation Tools]({{< ref "/developer_guide/docs" >}})
* [Pipeline Tools]({{< ref "/developer_guide/pipeline" >}})

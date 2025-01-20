---
title: Core Tools
---
<!-- markdownlint-disable MD034 -->

These tools are considered "core tools" since they ensure a consistent development flow.

{{< grid columns="1 2 2 3" >}}
[[item]]
type = "card"
title = "uv"
link = "{{< ref "uv" >}}"
body = '''
{{< image >}}
src = "https://raw.githubusercontent.com/astral-sh/uv/refs/heads/main/docs/assets/logo-letter.svg"
height = "48"
align = 'right'
loading = 'lazy'
{{< /image >}}
An extremely fast Python package and project manager, written in Rust.
'''

[[item]]
type = "card"
title = "pre-commit"
link = "{{< ref "pre_commit" >}}"
body = '''
{{< image >}}
src = "https://raw.githubusercontent.com/pre-commit/pre-commit.com/refs/heads/main/logo.svg"
height = "64"
align = 'right'
loading = 'lazy'
{{< /image >}}
A framework for managing and maintaining multi-language pre-commit hooks.
'''

[[item]]
type = "card"
title = "ACT3 Project Tool"
link = "{{< ref "act3_pt" >}}"
body = '''
ACT3's project management toolbox.
'''

[[item]]
type = "card"
title = "Commitlint"
link = "{{< ref "commitlint" >}}"
body = '''
{{< image >}}
src = "https://github.com/conventional-changelog/commitlint/raw/refs/heads/master/docs/public/assets/icon.svg"
height = "56"
align = 'right'
loading = 'lazy'
{{< /image >}}
Helps your team adhere to a commit convention
'''

[[item]]
type = "card"
title = "markdownlint-cli2"
link = "{{< ref "test" >}}"
body = '''
A fast, flexible, configuration-based command-line interface for linting Markdown/CommonMark files with the markdownlint library
'''
{{< /grid >}}

{{< admonition hint >}}
An IDE such as [VS Code](https://code.visualstudio.com/) is strongly encouraged.
{{< /admonition >}}

## Further Reading

* [Core Tools]({{< ref "/developer_guide/core" >}})
* [Linting Tools]({{< ref "/developer_guide/lint" >}})
* [Testing Tools]({{< ref "/developer_guide/test" >}})
* [Documentation Tools]({{< ref "/developer_guide/docs" >}})
* [Pipeline Tools]({{< ref "/developer_guide/pipeline" >}})

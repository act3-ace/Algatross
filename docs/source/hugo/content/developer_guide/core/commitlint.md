+++
date = '2024-12-03T18:08:55-05:00'
draft = false
title = 'Commitlint'
+++

[Commitlint](https://commitlint.js.org/) is a commit-message linter. This tool is important to ensure the [Semantic Release](https://semantic-release.gitbook.io/semantic-release) mechanism works correctly. The commit format is derived from [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit).

Make sure your commit message passes the commitlint checks before submitting. This is most easily achieved using an IDE such as [VS Code](https://code.visualstudio.com/) and the [commitlint extension](https://marketplace.visualstudio.com/items?itemName=joshbolduc.commitlint). This extension will highlight issues in your commit message as you type it.

## Further Reading

* [Core Tools]({{< ref "/developer_guide/core" >}})
* [Linting Tools]({{< ref "/developer_guide/lint" >}})
* [Testing Tools]({{< ref "/developer_guide/test" >}})
* [Documentation Tools]({{< ref "/developer_guide/docs" >}})
* [Pipeline Tools]({{< ref "/developer_guide/pipeline" >}})

---
title: Pre-Commit
---

[Pre-Commit](https://pre-commit.com/) is an extremely important tool in guaranteeing that errors are not introduced into the codebase and that commits are consistent between authors. The pre-commit binary will be installed when [uv]({{< ref "uv" >}}) installs the development dependencies. Once the binary is installed, pre-commit can be setup with the following command:

```bash
pre-commit install --install-hooks
```

This will install a [git hook](https://git-scm.com/book/ms/v2/Customizing-Git-Git-Hooks) which runs pre-commit checks before every call to `git commit`, `git merge`, etc.

## Handling Errors

In general, errors flagged by pre-commit should be resolved legitimately before resorting to these steps. For example, linting errors should be corrected, and only ignored in the case that a fix is not possible or the error is a false-positive.

### Detect Secrets

[detect-secrets](https://github.com/Yelp/detect-secrets) looks for high-entropy strings and keywords to make sure secure credentials are not committed to the repository. In the case of a false-positive, try the following:

1. Add `pragma: allowlist secret` as an inline comment on the flagged line
2. If the above fails, update the `.secrets.baseline` file to whitelist the secret:

```bash
detect-secrets scan --baseline .secrets.baseline
```

### Markdownlint

[markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2) lints markdown files for syntax and formatting. In the case of false-positives or errors which, if resolved, would break how the markdown is rendered, try the following:

1. Add `<!-- markdownlint-disable #error code# -->` as a comment on the line before the flagged error
2. Update `.markdownlint-cli2.yaml` to disable the specific rule for the file
3. Update `.markdownlint-cli2.yaml` to ignore the file entirely

### Ruff

[ruff](https://docs.astral.sh/ruff/) is a python linter and formatter. In many cases, ruff is able to auto-fix errors. In the case where ruff does not auto-fix errors, and the error is considered a false-positive, try the follwing:

1. Add `noqa: <error code>` as an inline comment
2. Modify `pyproject.toml` to disable the rule for the file in the `[tool.ruff.lint.per-file-ignores]` section.

{{< admonition important >}}
Always include the error code in the `noqa` directive.
{{< /admonition >}}

### MyPy

[mypy](https://mypy.readthedocs.io/en/stable/) is a static type-checker for python. It does its best to determine the typing for your code but due to multiple-dispatch and duck-typing in python false-positives are inevitable. If the code cannot be modified in a way which satisfies mypy, try the following:

1. Add a `type: ignore[<error code>]` as an inline comment
2. Modify `pyproject.toml` to exclude the file path using the `exclude` table under the `[tool.mypy]` section.

{{< admonition attention >}}
The `type: ignore` directive must appear before any other directives on the same line, such as `noqa`
{{< /admonition >}}
{{< admonition important >}}
Always include the error code in the `type: ignore` directive
{{< /admonition >}}

### ACT3 Project Tool

[act3-pt](https://git.act3-ace.com/devsecops/act3-pt) is a project templating tool for projects on ACT3 Gitlab. If the act3-pt linter identifies missing files try the following:

1. Add the file path to the `ignore` field under the `blueprints` key.

## Further Reading

* [Core Tools]({{< ref "/developer_guide/core" >}})
* [Linting Tools]({{< ref "/developer_guide/lint" >}})
* [Testing Tools]({{< ref "/developer_guide/test" >}})
* [Documentation Tools]({{< ref "/developer_guide/docs" >}})
* [Pipeline Tools]({{< ref "/developer_guide/pipeline" >}})

#!/usr/bin/env python3
"""
A script for updating the version list JSON for the version switcher.

See: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html
"""
import json
import re

from pathlib import Path

version_list_source = "version_list.json"
base_url = "https://stalwart.git.act3-ace.com/ascension/mo-marl"
url_suffix = "api"

with Path("VERSION").open("r", encoding="utf-8") as f:
    new_version = f.readline().strip("\n").strip()

with Path(version_list_source).open("r", encoding="utf-8") as f:
    version_list: list[dict[str, str]] = json.load(f)

main_matcher = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)$")
alternate_matcher = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)-(.*)$")

new_stable = main_matcher.match(new_version)
new_alternate = alternate_matcher.match(new_version)

# create the new version specification
new_version_spec = {"name": f"v{new_version}", "version": new_version}
if new_stable:
    new_version_spec["name"] = f"{new_version_spec['name']} (stable)"
    new_version_spec["url"] = f"{base_url}/{url_suffix}"
    new_version_spec["preferred"] = "true"
    new_version_spec["channel"] = "main"
elif new_alternate:
    alternate_channel = new_alternate.group(4).split(".")[0]
    new_version_spec["name"] = alternate_channel
    new_version_spec["url"] = f"{base_url}/{alternate_channel}/{url_suffix}"
    new_version_spec["channel"] = alternate_channel

for idx, version_spec in enumerate(version_list):
    if new_stable:
        if "preferred" in version_spec:
            del version_spec["preferred"]
        # replace the stable url with a permalink to the tagged version
        # and remove the '(stable)' from the name
        if "name" in version_spec and "(stable)" in version_spec["name"]:
            version_spec["name"] = version_spec["name"].replace("(stable)", "").strip()
            version_spec["url"] = f"{base_url}/v{version_spec['version']}/{url_suffix}"
            break
    if new_alternate:
        if version_spec["url"] == new_version_spec["url"]:
            break

# all alternate versions should point to a single documentation
if new_alternate:
    version_list.pop(idx)

if new_stable or new_alternate:
    version_list.insert(0, new_version_spec)

    with Path(version_list_source).open("w", encoding="utf-8") as f:
        json.dump(version_list, f, indent=4)

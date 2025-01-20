#!/usr/bin/env python3

from pathlib import Path
from importlib import import_module
import inspect
import sys

print(sys.executable)
for mp in sys.argv[1:]:
    print(mp)
    p = Path(mp).relative_to(Path().cwd())
    imp = ".".join(p.parts[:-1])
    imp += "." + p.stem
    m = import_module(str(imp))
for o_name, o in inspect.getmembers(m, inspect.isclass):
    print(o_name)
    for om_name, om in inspect.getmembers(o, inspect.isfunction):
        if not om_name.startswith("_"):
            print(om_name)
    print()

from contextlib import suppress
import os
from pathlib import Path


def gather_empty(
    base_dir: str | Path = "experiments",
    clear: list[Path] | None = None,
    /,
    keep: list[str | Path] | None = None,
    keep_latest: bool = True,
):
    clear = clear or []
    latest: Path | None = None
    for file in Path(base_dir).rglob("**/results.json"):
        # if file.is_dir():
        #     clear = clear_empty(file, clear, keep=keep, keep_latest=keep_latest)
        # else:
        # for subfile in file.glob("**/results.json"):
        # print(file)
        if file.is_dir():
            clear = gather_empty(file, clear, keep=keep, keep_latest=keep_latest)
            continue
        if file.stat().st_size == 0:
            clear.append(file.absolute().parent)
        if keep_latest and ((latest and file.stat().st_mtime > latest.stat().st_mtime) or not latest):
            latest = file.absolute().parent
    if keep_latest and latest:
        with suppress(ValueError):
            clear.remove(latest)
    if keep:
        for k in keep:
            if (k_path := Path(k).absolute()) in clear:
                clear.remove(k_path)
    # for file in sorted(clear):
    return clear


def clear_empty(
    base_dir: str | Path = "experiments",
    /,
    keep: list[str | Path] | None = None,
    keep_latest: bool = True,
):
    def empty_dir(path: Path):
        if path.exists():
            if path.is_file():
                os.remove(path)
            elif path.is_dir():
                for sub_path in path.iterdir():
                    empty_dir(sub_path)
                path.rmdir()

    for file in gather_empty(base_dir, keep=keep, keep_latest=keep_latest):
        empty_dir(file)


if __name__ == "__main__":
    print("\n".join(map(str, gather_empty(keep=["experiments/06f4fb18", "experiments/194d9532"]))))
    clear_empty()

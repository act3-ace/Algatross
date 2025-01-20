#!/usr/bin/env python3
"""Update the content in ``news.md`` with the latest changes from the changelog."""

__description__ = """Any existing content from the preamble is carried over unless it is an updated
field (`newsHeader` & `date`). The incoming content is inserted between the
preamble and the old content.

Summarizes the content by grouping the commits under identical commit messages in
the same change type. E.G.:

    >\t### Bug Fixes
    >
    >\t* <note> (<commit 1>)
    >\t* <note> (<commit 2>)
    >
    >\t### Features
    >
    >\t* <note> (<commit 3>)

This will be summarized as:

    >\t### Bug Fixes
    >
    >\t* <note> (<commit 1>, <commit 2>)
    >
    >\t### Features
    >
    >\t* <note> (<commit 3>)

Also slaps some emojis on the headings depending on the type of semantic release label

Before anything is written, a backup of the existing file is made.

This does not check to make sure the changelog has not already been added so this
should only be ran during a release on the main branch.

"""

import argparse
from collections import defaultdict
import datetime
import logging
import os
from pathlib import Path
import re
import shutil

logger = logging.getLogger(__file__)

RELEASE_NOTE_PATTERN = re.compile(r"^\*\s*(?P<note>.*)\s\((?P<commit>.*)\)$")
PREAMBLE_PATTERN = re.compile(r"^(?:newsHeader|title|sidebar|date|(?P<preamble_block>-{3})).*$")
NON_DEFAULT_BRANCH_PATTERN = re.compile(r"^#\s\[?\d+\.\d+\.\d+(?:-(.*?))?\]?\(.*\).*$")
HEADING_PATTERN = re.compile(r"^(?P<heading_level>#+)\s*(?P<heading>.*?)\s*$")
RELEASE_PATTERN = re.compile(r"(\[)?(?P<heading>\d+\.\d+\.\d+(?:-(?:\w*\.?)*)?)(?(1)\]).*\((?P<date>\d{4}-\d{1,2}-\d{1,2})\)")

# some nice eye-candy
EMOJI_MAP = {
    "Bug Fixes": "🐞",
    "Features": "🎉",
    "Performance Improvements": "⚡️",
}

# default lines which have fixed values
PREAMBLE_LINES = ["title: News\n", "sidebar: false\n"]

def summarize_section(summary_section: str, summary_content: dict[str, defaultdict[str, list]], line: str, content: list[str], stop: bool = False) -> str:
    # aggregate the commits with the same message into a single line
    if summary_section in summary_content and summary_content[summary_section]:
        for note, commits in summary_content[summary_section].items():
            content.append(f"* {note} ({', '.join(commits)})\n")
        del summary_content[summary_section]

    if stop:
        return ""

    if heading_match := HEADING_PATTERN.match(line):
        summary_section, heading_level = heading_match.group("heading", "heading_level")
        summary_content[summary_section] = defaultdict(list)

    emoji = EMOJI_MAP.get(summary_section, "🛠️")
    if emoji:
        emoji += " "
    # section heading level must at least be 4 since the main heading is level 3
    content.extend(["\n", f"{'#' * max(len(heading_level) + 1, 4)} {emoji}{summary_section}\n", "\n"])

    return summary_section

def gather_news_content(changelog: Path) -> tuple[str, str, list[str], bool]:
    content = []
    summary_content: dict[str, defaultdict[str, list]] = {}
    summary_section = ""
    hl = 0

    with changelog.open("rt", encoding="utf-8") as f:
        for line in f:

            if line == "\n":
                continue

            # the most recent changes in the changelog are not for the default branch
            # so we exit early
            if not hl and (branch_match := NON_DEFAULT_BRANCH_PATTERN.match(line)):
                logging.info(f"Most recent changes are for {branch_match.group(1)} which is not a default branch: exiting early...")
                return "", "", [], False

            if heading_match := HEADING_PATTERN.match(line):
                hl = len(heading_match.group("heading_level"))
                head = heading_match.group("heading")

                # title
                if hl == 1:

                    if summary_content:
                        logger.info("Finalizing content")
                        summarize_section(summary_section, summary_content, line, content, stop=True)
                        break

                    logger.info("Generating header")
                    if (release_match := RELEASE_PATTERN.match(head)):
                        header, date = release_match.group("heading", "date")
                        if not header:
                            return "", "", [], False

                        if not date:
                            today = datetime.datetime.now(datetime.timezone.utc)
                            date = f"{today.year:04d}-{today.month:02d}-{today.day:02d}"
                        title = f"🎉 MO-MARL {header} released! 🎉"
                        content.append(f"### MO-MARL {header} released\n")
                    continue

                # semantic release notes header
                if hl > 1:
                    logging.debug("New section found while working on another section... summarizing the current section")
                    summary_section = summarize_section(summary_section, summary_content, line, content)
                    continue

            # semantic release notes content
            if note_match := RELEASE_NOTE_PATTERN.match(line):
                note, commit = note_match.group("note", "commit")
                summary_content[summary_section][note].append(commit)
                continue

            content.append(line)
    return title, date, content, True

def write_news_content(news_log: Path, updates: tuple[str, str, list[str], bool]):
    # Insert news content into the news file.

    try:
        bak = news_log.with_suffix(".bak")
        logging.debug(f"Making a backup of {news_log}")
        src_path = Path(shutil.move(news_log, bak))
        logging.debug(f"Backup of {news_log} created at {src_path}")
        scanning_preamble = None
        current_preamble_lines = []
        source_line_count = 0
        new_line_count = 0

        header, date, lines_to_write, _ = updates
        new_line_count = len(lines_to_write)

        with news_log.open("w", encoding="utf-8") as dest:
            with src_path.open("r", encoding="utf-8") as src:
                logger.info("Inserting new content")
                for src_line in src:
                    if scanning_preamble in {None, True} and (preamble_match := PREAMBLE_PATTERN.match(src_line)):

                        # we've not been here before, flip the latch
                        if preamble_match.group("preamble_block") and scanning_preamble is None:
                            logging.debug("Entering preamble block")
                            scanning_preamble = True
                            continue
                        # we've found the closure to the preamble block
                        if preamble_match.group("preamble_block") and scanning_preamble:
                            scanning_preamble = False
                            logging.debug(f"Done scanning preamble block, found {len(current_preamble_lines)} lines to carry over into new document")
                            dest.writelines(["---\n", *PREAMBLE_LINES, f"newsHeader: {header}\n", f"date: {date}\n", *current_preamble_lines, "---\n", "\n", *lines_to_write])
                            logging.debug(f"Inserted updated preamble and new content")
                            carry_line_count = len(current_preamble_lines)
                            current_preamble_lines.clear()
                            lines_to_write.clear()
                            continue
                        continue
                    if scanning_preamble:
                        current_preamble_lines.append(src_line)
                        continue
                    source_line_count += 1
                    dest.write(src_line)
        if current_preamble_lines or lines_to_write:
            msg = f"Copied all old content back into the original file but never wrote any new content! Carried preamble lines: {len(current_preamble_lines)}, new content lines: {len(new_line_count)}"
            raise RuntimeError(msg)
        logging.debug(f"{carry_line_count}\tpreamble lines carried over")
        logging.debug(f"{source_line_count}\tcontent lines carried over")
        logging.debug(f"{new_line_count}\tnew lines inserted")
        logging.info(f"Cleaning up")
        os.remove(src_path)
        logging.debug(f"Removed backup file: {src_path}")
    except:
        logging.exception("Exception occured while trying to generate the content. Restoring from backup...")
        if news_log.exists() and src_path.exists():
            os.remove(news_log)
            logging.debug(f"Removed partial file: {news_log}")

        if src_path.exists():
            # we had an oopsie, restore the backup
            shutil.move(src_path, news_log)
            logging.debug(f"Restored backup from: {src_path}")
        raise

def main():

    parser = argparse.ArgumentParser(prog="auto_update_news", description=__doc__, epilog=__description__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file", default="CHANGELOG.md", type=Path, help="The path to the changelog to parse for new content (default is `CHANGELOG.md`)")
    parser.add_argument("-o", "--output", "--output-file", default="docs/source/hugo/content/news.md", type=Path, help="The path to the news file to update (default is `docs/source/hugo/content/news.md`)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="The verbosity level. Specify the flag multiple times to increase the verbosity, (default is 0 / `logging.CRITICAL`)")

    parsed = parser.parse_args()

    logging.basicConfig(level=max(50 - (10 * parsed.verbose), 0), format='%(asctime)s : %(levelname)s : %(message)s')

    updates = gather_news_content(parsed.file)

    # only generate news content if the most recent changes are for the default branch
    if updates[-1] and updates[2]:
        write_news_content(parsed.output, updates)
        return

    logging.info("No new content to write")

if __name__ == "__main__":
    main()

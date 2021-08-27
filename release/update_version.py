import argparse
import re
from pathlib import Path


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--init_file_path', type=Path, required=True)
    parser.add_argument('-v', '--new_version', type=str, required=True)
    parsed_args = vars(parser.parse_args())

    return parsed_args


def update_init_file(file_path: Path,
                     new_version: str):
    file_text = file_path.open().read()
    file_text_mod = re.sub(r"__version__ = '.*'",
                           f"__version__ = '{new_version}'",
                           file_text)
    file_path.write_text(file_text_mod)


if __name__ == '__main__':
    args = parse_args()
    update_init_file(file_path=args['init_file_path'],
                     new_version=args['new_version'])

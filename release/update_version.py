import argparse
import re
from pathlib import Path


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=Path, required=True)
    parser.add_argument('-v', '--new_version', type=str, required=True)
    parsed_args = vars(parser.parse_args())

    return parsed_args


def update_file(file_path: Path, new_version: str):
    file_text = file_path.read_text()
    old_version = re.search(r'_{0,2}version_{0,2} = (\'|\")(.*)(\'|\")', file_text).group(2)

    file_text_mod = file_text.replace(old_version, new_version, 1)
    file_path.write_text(file_text_mod)


def main():
    args = parse_args()
    update_file(file_path=args['file_path'], new_version=args['new_version'])


if __name__ == '__main__':
    main()

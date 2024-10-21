import gzip
import json

from territories import Territory


def test_parse_without_error():
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))

    t = Territory.from_names("DEP:69")
    json.dumps(t)
    json.dumps(t.descendants())
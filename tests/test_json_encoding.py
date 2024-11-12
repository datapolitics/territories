import gzip
import json

from territories import Territory


def test_parse_without_error(benchmark):
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))

    t = Territory.from_names("DEP:69")
    res = benchmark.pedantic(json.dumps, t, rounds=100)
    assert json.loads(res) == {"name": "Rhône", "tu_id": "DEP:69", "atomic": False, "partition_type": "DEP", "postal_code": None}


def test_parse_fast(benchmark):
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))

    t = Territory.from_names("DEP:69")
    benchmark.pedantic(json.dumps, args=(t.descendants(), ), rounds=100)
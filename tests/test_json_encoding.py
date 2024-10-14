import gzip
import json
import pickle

from territories import Territory, MissingTreeCache
from territories.partitions import Node


def test_parse_without_error():
    split = lambda x: (arg if arg != 'null' else None for arg in x[:-1].split('; '))

    try:
        Territory.load_tree()
    except MissingTreeCache:
        with open("docs/tree_large.gzip", "rb") as file:
            lines = pickle.loads(gzip.decompress(file.read()))

        stream = ([Node(*split(x) )for x in lines])
        Territory.build_tree(data_stream=stream, save_tree=False)

    t = Territory.from_name("DEP:69")
    s = Territory.successors(t)
    json.dumps(s)
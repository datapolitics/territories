import gzip

from territories import Territory


def test_legacy_codes():
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))

    assert Territory.from_tu_ids("f81d4fae-7dec-11d0-a765-00a0c91e6bf6") == Territory.from_tu_ids("DEP:69", "DEP:75")


            # if "EPCI" in name and not name in epci_to_comm:
            #     name = name.split('-')[0]
            # return epci_to_comm.get(name, name)

import gzip
import json
import pytest

import rustworkx as rx

from random import sample, choice
from territories import Territory, NotOnTreeError, MissingTreeException
from territories.partitions import TerritorialUnit, Partition, Node


lyon = TerritorialUnit("Lyon", tu_id="Lyon")
marseille = TerritorialUnit("Marseille", tu_id="Marseille")
paris = TerritorialUnit("Paris", tu_id="Paris")
nogent = TerritorialUnit("Nogent", tu_id="Nogent")
pantin = TerritorialUnit("Pantin", tu_id="Pantin")
villeurbane = TerritorialUnit("Villeurbane", tu_id="Villeurbane")
sté = TerritorialUnit("Saint Etienne", tu_id="Etienne")

metropole = TerritorialUnit("Grand Lyon", "metro", False, Partition.DEP)

sud = TerritorialUnit("Sud", "Sud", False, Partition.REG)
idf = TerritorialUnit("Île-de-France", "idf", False, Partition.REG)
rhone = TerritorialUnit("Rhône", "Rhône", False, Partition.DEP)

france = TerritorialUnit("France", "France", False, Partition.CNTRY)


entities = (france, sud, idf, rhone, metropole, nogent, pantin, paris, marseille, sté, villeurbane, lyon)

tree= rx.PyDiGraph()
entities_indices = tree.add_nodes_from(entities)

mapper = {o : idx for o, idx in zip(entities, entities_indices)}
edges = [
    (france, idf),
    (france, sud),

    (idf, nogent),
    (idf, pantin),
    (idf, paris),

    (sud, marseille),
    (sud, rhone),

    (rhone, metropole),
    (rhone, sté),

    (metropole, villeurbane),
    (metropole, lyon),
    ]

tree.add_edges_from([
    (mapper[parent], mapper[child], None) for parent, child in edges
])


Territory.assign_tree(tree)


a = Territory(sté, marseille)
b = Territory(lyon, france)
c = Territory(paris, nogent, pantin, lyon, lyon, metropole)
d = Territory(lyon, villeurbane, marseille)
e = Territory(rhone, idf)
f = Territory(idf, marseille, metropole)

exemples = (a, b, c, d, e, f)


@pytest.fixture
def load_tree():
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))


def test_imports():
    from territories import MissingTreeException, MissingTreeCache, NotOnTreeError, TerritorialUnit, Partition, Territory


def test_creation():
    Territory.reset()
    with pytest.raises(MissingTreeException):
        Territory()
    with pytest.raises(MissingTreeException):
        Territory.from_tu_ids("DEP:69")
    Territory.assign_tree(tree)
    t = Territory()
    assert t.is_empty()



@pytest.mark.filterwarnings("ignore:This method is deprecated")
def test_from_names():
    Territory.assign_tree(tree)
    new = Territory.from_names("Pantin", "Rhône")
    assert new == Territory(pantin, rhone)

    with pytest.raises(NotOnTreeError, match=r"^([\w\s]+,)*[\w\s]+ where not found in the territorial tree$"):
        new = Territory.from_names("not exist", "Rhône", "yolo")

    with pytest.raises(NotOnTreeError, match='not exist was not found in the territorial tree'):
        new = Territory.from_names("not exist", "Rhône")


def test_from_tu_ids():
    Territory.assign_tree(tree)

    assert Territory(pantin, rhone) == Territory.from_tu_ids("Pantin", "Rhône")
    assert Territory(pantin) == Territory.from_tu_ids("Pantin")
    assert Territory(pantin, rhone) == Territory.from_tu_ids((i for i in ("Pantin", "Rhône")))
    assert Territory(pantin, rhone) == Territory.from_tu_ids(["Pantin", "Rhône"])
    assert Territory(pantin, rhone) == Territory.from_tu_ids(("Pantin", "Rhône"))
    assert Territory(pantin, rhone) == Territory.from_tu_ids({"Pantin", "Rhône"})

    # the final test (not sure this monstruosity should be valid)
    assert Territory(france) == Territory.from_tu_ids([{"Pantin", "Rhône"}, "Villeurbane"], ("Marseille", "Lyon"), "Marseille", "Île-de-France")

    assert Territory() == Territory.from_tu_ids({})
    assert Territory() == Territory.from_tu_ids([])
    assert Territory() == Territory.from_tu_ids(set())
    assert Territory() == Territory.from_tu_ids(tuple())


    with pytest.raises(NotOnTreeError, match=r"^([\w\s]+,)*[\w\s]+ were not found in the territorial tree$"):
        _ = Territory.from_tu_ids("not exist", "Rhône", "yolo")
    with pytest.raises(NotOnTreeError, match=r"^([\w\s]+,)*[\w\s]+ were not found in the territorial tree$"):
            _ = Territory.from_tu_ids(["not exist", "Rhône", "yolo"])
    with pytest.raises(NotOnTreeError, match=r"^([\w\s]+,)*[\w\s]+ were not found in the territorial tree$"):
        _ = Territory.from_tu_ids(["not exist", "Rhône", "yolo", "Pantin"])
    with pytest.raises(NotOnTreeError, match='not exist was not found in the territorial tree'):
        _ = Territory.from_tu_ids({"not exist", "Rhône"})
    with pytest.raises(TypeError, match='tu_ids are string, you provided a int : 5'):
        _ = Territory.from_tu_ids(5)
    with pytest.raises(TypeError, match='tu_ids are string, you provided a int : 5'):
        _ = Territory.from_tu_ids(["Rhône", 5, "Pantin"])
    with pytest.raises(TypeError, match='tu_ids are string, you provided a int : 5'):
        _ = Territory.from_tu_ids("Rhône", 5, "yolo")
        
        
def test_from_name():
    Territory.assign_tree(tree)
    new = Territory.from_name("Pantin")
    assert new == Territory(pantin)

    with pytest.raises(NotOnTreeError, match='not exist'):
        new = Territory.from_name("not exist")


def test_base_init():
    Territory.assign_tree(tree)
    pantin = Territory.from_name("Pantin")
    t = Territory(pantin)
    tt = Territory(t)
    assert t == tt
    assert t == Territory(t, tt)
    
    rhone = Territory.from_name("Rhône")
    t = Territory(pantin, rhone)
    assert t == Territory(pantin, t)
    assert t == Territory(rhone, t, tt)
    

def test_union():
    Territory.assign_tree(tree)
    assert Territory(france) == Territory.union(c, f, a)
    assert Territory(france) == Territory.union([c, f, a])
    assert Territory(france) == Territory.union({c, f, a})
    assert Territory(france) == Territory.union(c, [f, a])
    assert Territory(france) == Territory.union((c, f), a)


def test_intersection():
    Territory.assign_tree(tree)
    assert Territory(metropole) == Territory.intersection(b, c, d)
    assert Territory(metropole, idf) == Territory.intersection(c, b, e, f)
    assert Territory(metropole) == Territory.intersection([b, c, d])
    assert Territory(metropole, idf) == Territory.intersection({c, b, e, f})
    assert Territory(metropole) == Territory.intersection(b, (c, d))
    assert Territory(metropole, idf) == Territory.intersection([c, b, e], f)
    
    
def test_iteration():
    Territory.assign_tree(tree)
    for i in a:
        assert i in a.territorial_units


def test_load_from_bytes():
    with pytest.raises(Exception):
        Territory.load_tree_from_bytes(b"bad data")


def test_sort_tus(load_tree):
    names = ("COM:69132", "DEP:75", "CNTRY:F", "DEP:69")
    sorted_names = ("CNTRY:F", "DEP:69", "DEP:75", "COM:69132")
    tus = [Territory.from_name(name) for name in names]
    tus.sort()
    assert tus == [Territory.from_name(name) for name in sorted_names]


def test_type(load_tree):
    names = ("COM:69132", "DEP:75", "CNTRY:F", "DEP:69")
    ter = Territory.from_tu_ids(*names)
    assert ter.type == Partition.CNTRY
    empty = Territory()
    assert empty.type == Partition.EMPTY


def test_parents(load_tree):
    ter = Territory.from_tu_ids("DEP:69", "COM:69132")
    assert ter.parents() == Territory.from_tu_ids("REG:84")

    ter = Territory.from_tu_ids("DEP:75", "COM:69132")
    assert ter.parents() == Territory.from_tu_ids("REG:11", "DEP:69")

    tu = Territory.from_name("DEP:69")
    assert Territory.get_parent(tu) == Territory.from_name("REG:84")


def test_children(load_tree):
    cntr = Territory.from_tu_ids("CNTRY:F")
    assert all(t.level == Partition.REG for t in cntr.children())
        
    reg = Territory.from_tu_ids("REG:11")
    assert all(t.level == Partition.DEP for t in reg.children())

    dep = Territory.from_tu_ids("DEP:69")
    assert all(t.level == Partition.COM for t in dep.children())

    child_union = set(reg.children() + dep.children())
    assert child_union == set((reg | dep).children())


def setup_large():
    s = sample(Territory.tree.nodes(), 1000)
    ter = Territory(*s)
    names = [tu.tu_id for tu in ter.descendants(include_itself=True) if tu.level == Partition.COM]
    return names, {}


def setup_small():
    ter = Territory(choice(Territory.tree.nodes()))
    names = [tu.tu_id for tu in ter.descendants(include_itself=True) if tu.level <= Partition.COM]
    return names, {}


def test_creation_large(load_tree, benchmark):
    benchmark.pedantic(Territory.from_tu_ids, setup=setup_large, rounds=10)
    
    
def test_creation_small(load_tree, benchmark):
    benchmark.pedantic(Territory.from_tu_ids, setup=setup_small, rounds=1000)


def setup_minimized():
    ter = Territory(*sample(Territory.tree.nodes(), 4))
    names = [tu.tu_id for tu in ter]
    return names, {}


def test_creation_already_minimized(load_tree, benchmark):
    benchmark.pedantic(Territory.from_tu_ids, setup=setup_minimized, rounds=100)


def test_tu_ids(load_tree):
    ter = Territory.from_tu_ids("DEP:69", "COM:69132", "DEP:75")
    assert ter.tu_ids == ["DEP:69", "DEP:75"] # the order must be deterministic
    assert Territory().tu_ids == []


def test_tu_path(load_tree):
    ter = Territory.from_tu_ids("DEP:69", "COM:69132", "DEP:75")
    assert ter.tu_path == ['CNTRY:F', 'REG:11', 'REG:84', 'DEP:69', 'DEP:75'] # the order must be deterministic

    assert Territory().tu_path == []

def test_tu_names(load_tree):
    ter = Territory.from_tu_ids("DEP:69", "COM:69132", "DEP:75")
    assert ter.tu_names == ["Rhône", "Paris"] # the order must be deterministic


def test_hash(load_tree):
    """this test mobilize every test before"""
    print(Territory.from_tu_ids("REG:11", "DEP:69"))
    # assert hash(Territory(france)) == hash(Territory.union([c, f, a]))
    # assert hash(Territory(metropole)) == hash(Territory.intersection([b, c, d]))
    assert hash(Territory.get_parent(Territory.from_name("DEP:69"))) == hash(Territory.from_name("REG:84"))
    assert hash(Territory.from_tu_ids("DEP:75", "COM:69132").parents()) == hash(Territory.from_tu_ids("REG:11", "DEP:69"))
    
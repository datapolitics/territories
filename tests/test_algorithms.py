import gzip

import pytest
import rustworkx as rx

from territories import Territory
from territories.partitions import TerritorialUnit, Partition


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

tree = rx.PyDiGraph()
entities_indices = tree.add_nodes_from(entities)

mapper = {o: idx for o, idx in zip(entities, entities_indices)}
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

tree.add_edges_from([(mapper[parent], mapper[child], None) for parent, child in edges])

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
    Territory.reset()
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))


def test_lca():
    Territory.assign_tree(tree)

    for entity in entities:
        t = Territory(entity)
        assert t.lowest_common_ancestor() == entity

    assert sud == a.lowest_common_ancestor()
    assert france == b.lowest_common_ancestor()
    assert france == c.lowest_common_ancestor()
    assert sud == d.lowest_common_ancestor()
    assert france == e.lowest_common_ancestor()
    assert france == f.lowest_common_ancestor()

    assert sud == Territory.LCA(lyon, marseille)
    assert france == Territory.LCA(lyon, Territory(marseille, paris))


def test_lca_includes_ancestor_itself_on_real_tree(load_tree):
    france = next(iter(Territory.from_tu_ids("CNTRY:F")))
    auvergne_rhone_alpes = next(iter(Territory.from_tu_ids("REG:84")))
    rhone = next(iter(Territory.from_tu_ids("DEP:69")))
    lyon = next(iter(Territory.from_tu_ids("COM:69123")))

    assert Territory(lyon).parents() == Territory(rhone)
    assert Territory.LCA(rhone, lyon) == rhone
    assert Territory.LCA(auvergne_rhone_alpes, rhone) == auvergne_rhone_alpes
    assert Territory.LCA(france, lyon) == france


def test_unit_distance():
    Territory.assign_tree(tree)

    assert Territory.unit_distance(lyon, lyon) == 0
    assert Territory.unit_distance(metropole, lyon) == 1
    assert Territory.unit_distance(lyon, metropole) == 1
    assert Territory.unit_distance(lyon, villeurbane) == 2
    assert Territory.unit_distance(lyon, marseille) == 4
    assert Territory.unit_distance(pantin, marseille) == 4
    assert Territory.unit_distance(france, lyon) == 4


def test_unit_distance_on_real_tree(load_tree):
    france = next(iter(Territory.from_tu_ids("CNTRY:F")))
    auvergne_rhone_alpes = next(iter(Territory.from_tu_ids("REG:84")))
    rhone = next(iter(Territory.from_tu_ids("DEP:69")))
    lyon = next(iter(Territory.from_tu_ids("COM:69123")))
    paris = next(iter(Territory.from_tu_ids("DEP:75")))
    brest = next(iter(Territory.from_tu_ids("COM:29019")))
    nancy = next(iter(Territory.from_tu_ids("COM:54395")))

    assert Territory.unit_distance(lyon, lyon) == 0
    assert Territory.unit_distance(rhone, lyon) == 1
    assert Territory.unit_distance(auvergne_rhone_alpes, lyon) == 2
    assert Territory.unit_distance(france, lyon) == 3
    assert Territory.unit_distance(rhone, paris) == 4
    assert Territory.unit_distance(brest, lyon) == 6
    assert Territory.unit_distance(brest, nancy) == 6


def test_distance():
    Territory.assign_tree(tree)

    assert Territory(lyon).distance(Territory(lyon)) == 0
    assert Territory(france).distance(Territory(lyon)) == 4
    assert Territory(metropole).distance(lyon) == 1
    assert Territory(lyon).distance(Territory(villeurbane)) == 2
    assert Territory(lyon).distance(Territory(marseille)) == 4
    assert Territory(pantin, lyon).distance(Territory(marseille, villeurbane)) == 4


def test_distance_properties():
    Territory.assign_tree(tree)
    territories = [
        Territory(france),
        Territory(sud),
        Territory(idf),
        Territory(rhone),
        Territory(metropole),
        Territory(lyon),
        Territory(villeurbane),
        Territory(marseille),
        Territory(pantin, lyon),
        Territory(marseille, villeurbane),
    ]

    for left in territories:
        for right in territories:
            assert (left.distance(right) == 0) == (left == right)
            assert left.distance(right) == right.distance(left)

            for third in territories:
                assert left.distance(third) <= left.distance(right) + right.distance(third)


def test_distance_on_real_tree(load_tree):
    france = Territory.from_tu_ids("CNTRY:F")
    rhone = Territory.from_tu_ids("DEP:69")
    lyon = Territory.from_tu_ids("COM:69123")
    paris = Territory.from_tu_ids("DEP:75")
    brest = Territory.from_tu_ids("COM:29019")
    nancy = Territory.from_tu_ids("COM:54395")

    assert lyon.distance(lyon) == 0
    assert france.distance(lyon) == 3
    assert rhone.distance(lyon) == 1
    assert rhone.distance(paris) == 4
    assert brest.distance(lyon | nancy) == 6


def test_ancestors():
    Territory.assign_tree(tree)

    assert a.ancestors() == [france, sud, rhone]
    assert b.ancestors() == []
    assert c.ancestors() == [france, sud, rhone]
    assert d.ancestors() == [france, sud, rhone]

    assert c.ancestors(include_itself=True) == [france, idf, sud, rhone, metropole]
    assert d.ancestors(include_itself=True) == [france, sud, rhone, metropole, marseille]

    assert Territory.all_ancestors(paris, marseille) == [france, idf, sud]
    assert Territory.all_ancestors(paris, Territory(villeurbane, sté)) == [france, idf, sud, rhone, metropole]

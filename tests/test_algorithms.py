import rustworkx as rx

from itertools import product

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


def test_lca():
    Territory.assign_tree(tree)

    assert sud == a.lowest_common_ancestor()
    assert None == b.lowest_common_ancestor()
    assert france == c.lowest_common_ancestor()
    assert sud == d.lowest_common_ancestor()
    assert france == e.lowest_common_ancestor()
    assert france == f.lowest_common_ancestor()


    assert sud == Territory.LCA(lyon, marseille)
    assert france == Territory.LCA(lyon, Territory(marseille, paris))
    assert rhone != Territory.LCA(rhone)
    assert sud == Territory.LCA(rhone)


def test_ancestors():
    Territory.assign_tree(tree)

    assert a.ancestors() == {rhone, sud, france}
    assert b.ancestors() == set()
    assert c.ancestors() == {rhone, sud, france}
    assert d.ancestors() == {rhone, sud, france}

    assert c.ancestors(include_itself=True) == {rhone, sud, france, idf, metropole}
    assert d.ancestors(include_itself=True) == {rhone, sud, france, metropole, marseille}

    assert Territory.all_ancestors(paris, marseille) == {sud, idf, france}
    assert Territory.all_ancestors(paris, Territory(villeurbane, sté)) == {metropole, rhone, sud, idf, france}

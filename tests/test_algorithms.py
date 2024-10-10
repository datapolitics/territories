import rustworkx as rx

from itertools import product

from territories import Territory
from territories.partitions import TerritorialUnit, Partition


lyon = TerritorialUnit("Lyon")
marseille = TerritorialUnit("Marseille", tu_id="COM:2909") # you can specify an ElasticSearch code
paris = TerritorialUnit("Paris")
nogent = TerritorialUnit("Nogent")
pantin = TerritorialUnit("Pantin")
villeurbane = TerritorialUnit("Villeurbane")
sté = TerritorialUnit("Saint Etienne")

metropole = TerritorialUnit("Grand Lyon", False, Partition.DEP)

sud = TerritorialUnit("Sud", False, Partition.REG)
idf = TerritorialUnit("Île-de-France", False, Partition.REG)
rhone = TerritorialUnit("Rhône", False, Partition.DEP)

france = TerritorialUnit("France", False, Partition.CNTRY)



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

    assert Territory(sud) == a.lowest_common_ancestor()
    assert Territory() == b.lowest_common_ancestor()
    assert Territory(france) == c.lowest_common_ancestor()
    assert Territory() == b.lowest_common_ancestor()
    assert Territory(france) == c.lowest_common_ancestor()

    assert Territory(sud) == Territory.LCA(lyon, marseille)
    assert Territory(france) == Territory.LCA(lyon, Territory(marseille, paris))
    assert Territory(sud) == Territory.LCA(rhone)


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

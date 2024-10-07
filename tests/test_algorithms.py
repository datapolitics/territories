import rustworkx as rx

from itertools import product

from territories import Territory
from territories.partitions import Part, Partition


lyon = Part("Lyon")
marseille = Part("Marseille", es_code="COM:2909") # you can specify an ElasticSearch code
paris = Part("Paris")
nogent = Part("Nogent")
pantin = Part("Pantin")
villeurbane = Part("Villeurbane")
sté = Part("Saint Etienne")

metropole = Part("Grand Lyon", False, Partition.DEP)

sud = Part("Sud", False, Partition.REGION)
idf = Part("Île-de-France", False, Partition.REGION)
rhone = Part("Rhône", False, Partition.DEP)

france = Part("France", False, Partition.COUNTRY)



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


def test_ancestors():
    Territory.assign_tree(tree)

    assert a.ancestors() == {rhone, sud, france}
    assert b.ancestors() == set()
    assert c.ancestors() == {rhone, sud, france}
    assert d.ancestors() == {rhone, sud, france}

    assert Territory.all_ancestors(paris, marseille) == {sud, idf, france}
    assert Territory.all_ancestors(paris, Territory(villeurbane, sté)) == {metropole, rhone, sud, idf, france}

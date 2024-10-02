import rustworkx as rx

from itertools import product

from territories import Part, Territory, Partition


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

france = Part("France", False, Partition.PAYS)


def build_tree() -> rx.PyDiGraph:
    print("BUILDING TREE : this is a very long operation")
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

    return tree


Territory.assign_tree(build_tree())

a = Territory(sté, marseille)
b = Territory(lyon, france)
c = Territory(paris, nogent, pantin, lyon, lyon, metropole)
d = Territory(lyon, villeurbane, marseille)
e = Territory(rhone, idf)
f = Territory(idf, marseille, metropole)


exemples = (a, b, c, d, e, f)

def test_eqality():
    assert b == Territory(france)

    for i, j in product(exemples, exemples):
        assert (j == i) == (i == j)

def test_addition():
    assert d + a == Territory(sud)
    assert c + a == Territory(idf, sud)
    assert d + c == Territory(metropole, marseille, idf)

    for i, j in product(exemples, exemples):
        assert i + j == j + i


def test_inclusion():
    assert a in b
    assert a in c + a
    assert a not in d
    assert d in f

    for i, j in zip(exemples, exemples):
        assert j in i
        assert i in j


def test_union():
    assert a | d == Territory(sud)
    assert c | d == Territory(idf, marseille, metropole)

    for i, j in product(exemples, exemples):
        assert i | j == j | i


def test_intersection():
    assert a & b == a
    assert a & d == Territory(marseille)
    assert e & f == Territory(idf, metropole)
    
    for i, j in product(exemples, exemples):
        assert i & j == j & i


def test_substraction():
    assert a - b == Territory()
    assert b - a == Territory(metropole, idf)
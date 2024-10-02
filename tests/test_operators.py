import networkx as nx

from itertools import product

from territories import Part, Territory, Partition


lyon = Part("Lyon")
marseille = Part("Marseille", es_code="COM:2909") # you can specify an ElasticSearch code
paris = Part("Paris")
nogent = Part("Nogent")
pantin = Part("Pantin")
villeurbane = Part("Villeurbane")
sté = Part("Saint Etienne")

metropole = Part("Grand Lyon", False, Partition.EPCI)

sud = Part("Sud", False, Partition.REGION)
idf = Part("Île-de-France", False, Partition.REGION)
rhone = Part("Rhône", False, Partition.DEP)

france = Part("France", False, Partition.PAYS)


# buiding the reference territory tree
tree = nx.DiGraph([
    (france, sud),
    (france, idf),

    (idf, nogent),
    (idf, pantin),
    (idf, paris),

    (sud, marseille),
    (sud, rhone),

    (rhone, metropole),
    (rhone, sté),

    (metropole, villeurbane),
    (metropole, lyon),
])


Territory.assign_tree(tree)

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
        if i == j:
            assert j == i

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
import networkx as nx

from territories import Entity, Territory, Type

lyon = Entity("Lyon")
marseille = Entity("Marseille", es_code="COM:2909") # you can specify an ElasticSearch code
paris = Entity("Paris")
nogent = Entity("Nogent")
pantin = Entity("Pantin")
villeurbane = Entity("Villeurbane")
sté = Entity("Saint Etienne")

metropole = Entity("Grand Lyon", False, Type.EPCI)

sud = Entity("Sud", False, Type.REGION)
idf = Entity("Île-de-France", False, Type.REGION)
rhone = Entity("Rhône", False, Type.DEP)

france = Entity("France", False, Type.PAYS)


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


def test_eqality():
    assert b == Territory(france)


def test_addition():
    assert d + a == Territory(sud)
    assert c + a == Territory(idf, sud)
    assert d + c == Territory(metropole, marseille, idf)


def test_inclusion():
    assert a in b
    assert a in c + a
    assert a not in d


def test_union():
    assert a | d == Territory(sud)
    assert c | d == Territory(idf, marseille, metropole)


def test_intersection():
    assert a & b == a
    assert a & d == Territory(marseille)
    assert e & f == Territory(idf, metropole)


def test_substraction():
    assert a - b == Territory()
    print(b - a)
    assert b - a == Territory(metropole, idf)
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

a = Territory(entities=(sté, marseille))
b = Territory(entities=(lyon, france))
c = Territory(entities=(paris, nogent, pantin, lyon, lyon, metropole))
d = Territory(entities=(lyon, villeurbane, marseille))


def test_eqality():
    assert b == Territory(entities=(france, ))


def test_addition():
    assert d + a == Territory(entities=(sud, ))
    assert c + a == Territory(entities=(idf, sud))
    assert d + c == Territory(entities=(metropole, marseille, idf ))


def test_inclusion():
    assert a in b
    assert a in c + a
    assert a not in d


def test_union():
    assert a | d == Territory(entities=(sud, ))
    assert c | d == Territory(entities=(idf, marseille, metropole))


def test_intersection():
    assert a & b == a
    assert a & d == Territory(entities=(marseille, ))


def test_substraction():
    assert a - b == Territory()
    assert b - a == Territory(entities=(metropole, idf))
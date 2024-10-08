import rustworkx as rx

from territories import Territory
from territories.partitions import TerritorialUnit, Partition, Node


lyon = TerritorialUnit("Lyon")
marseille = TerritorialUnit("Marseille", es_code="COM:2909") # you can specify an ElasticSearch code
paris = TerritorialUnit("Paris")
nogent = TerritorialUnit("Nogent")
pantin = TerritorialUnit("Pantin")
villeurbane = TerritorialUnit("Villeurbane")
sté = TerritorialUnit("Saint Etienne")

metropole = TerritorialUnit("Grand Lyon", False, Partition.DEP)

sud = TerritorialUnit("Sud", False, Partition.REGION)
idf = TerritorialUnit("Île-de-France", False, Partition.REGION)
rhone = TerritorialUnit("Rhône", False, Partition.DEP)

france = TerritorialUnit("France", False, Partition.COUNTRY)


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


def test_creation():
    # Territory.assign_tree(tree)
    Territory()


def test_from_es():
    # Territory.assign_tree(tree)
    new = Territory.from_name("Pantin", "Rhône")
    assert new == Territory(pantin, rhone)


def test_union():
    Territory.assign_tree(tree)
    t = Territory.union(c, f, a)
    assert t == Territory(france)


def test_intersection():
    Territory.assign_tree(tree)

    t = Territory.intersection(b, c, d)
    assert t == Territory(metropole)
    t = Territory.intersection(c, b, e, f)
    assert t == Territory(metropole, idf)



def test_build_tree():

    nodes = [
        Node(id='CNTRY:France', label='France', level='CNTRY', parent_id=None),
        Node(id='REG:Sud', label='Sud', level='REG', parent_id='CNTRY:France'),
        Node(id='REG:idf', label='île-de-france', level='REG', parent_id='CNTRY:France'),

        Node(id='DEP:Rhone', label='Rhône', level='DEP', parent_id='REG:Sud'),
        Node(id='DEP:metropole', label='Grand Lyon', level='DEP', parent_id='REG:Sud'),

        Node(id='COM:Pantin', label='Pantin', level='COM', parent_id="REG:idf"),
        Node(id='COM:Nogent', label='Nogent', level='COM', parent_id="REG:idf"),
        Node(id='COM:Paris', label='Paris', level='COM', parent_id="REG:idf"),

        Node(id='COM:sté', label='Saint Étienne', level='COM', parent_id="DEP:Rhone"),
        Node(id='COM:Lyon', label='Lyon', level='COM', parent_id="DEP:metropole"),
        Node(id='COM:Villeurbane', label='Villeurbane', level='COM', parent_id="DEP:metropole"),

        Node(id='COM:Marseille', label='Marseille', level='COM', parent_id="REG:Sud"),
    ]

    Territory.build_tree(nodes)

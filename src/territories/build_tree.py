import rustworkx as rx

from territories.territories import Part, Partition

def build_test_tree() -> rx.PyDiGraph:
    print("BUILDING TREE : this is a very long operation")

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
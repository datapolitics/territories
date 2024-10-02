import rustworkx as rx

from dataclasses import dataclass
from more_itertools import batched
from typing import Optional, Iterable

from territories.territories import Part, Partition


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    level: str
    parent_id: Optional[str]
    postal_code: Optional[str]


    def to_part(self):
        match self.level:
            case "COM":
                partition = Partition.COMMUNE
            case "DEP":
                partition = Partition.DEP
            case "REG":
                partition = Partition.REGION
            case "CNTRY":
                partition = Partition.COUNTRY
            case _:
                partition = None

        return Part(
            name=self.label,
            es_code=self.id,
            partition_type=partition
        )


def build_tree(data_stream: Iterable[Node]) -> rx.PyDiGraph:
    tree = rx.PyDiGraph()

    batch_size = 32
    mapper = {}
    for batch in batched(data_stream, batch_size):

        entities_indices = tree.add_nodes_from(node.to_part() for node in batch)
        
        for node, tree_idx in zip(batch, entities_indices):
            mapper[node.id] = tree_idx

        tree.add_edges_from((mapper[node.parent_id], tree_id, None) for node, tree_id in zip(batch, entities_indices))

    return tree

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



from territories import Territory
from territories.database import NodeTuple


def test_build_tree():
    nodes = [
        NodeTuple(id="CNTRY:France", label="France", level="CNTRY", parent_id=None),
        NodeTuple(id="REG:Sud", label="Sud", level="REG", parent_id="CNTRY:France"),
        NodeTuple(id="REG:idf", label="île-de-france", level="REG", parent_id="CNTRY:France"),
        NodeTuple(id="DEP:Rhone", label="Rhône", level="DEP", parent_id="REG:Sud"),
        NodeTuple(id="DEP:metropole", label="Grand Lyon", level="DEP", parent_id="REG:Sud"),
        NodeTuple(id="COM:Pantin", label="Pantin", level="COM", parent_id="REG:idf"),
        NodeTuple(id="COM:Nogent", label="Nogent", level="COM", parent_id="REG:idf"),
        NodeTuple(id="COM:Paris", label="Paris", level="COM", parent_id="REG:idf"),
        NodeTuple(id="COM:sté", label="Saint Étienne", level="COM", parent_id="DEP:Rhone"),
        NodeTuple(id="COM:Lyon", label="Lyon", level="COM", parent_id="DEP:metropole"),
        NodeTuple(id="COM:Villeurbane", label="Villeurbane", level="COM", parent_id="DEP:metropole"),
        NodeTuple(id="COM:Marseille", label="Marseille", level="COM", parent_id="REG:Sud"),
    ]

    Territory.build_tree(nodes, save_tree=True, filepath='/tmp/foo.pickle')

import pickle

from territories.partitions import Node
from territories.territories import Territory
from territories.database import create_connection, read_stream


def build_tree_from_db():
    with create_connection("crawling") as cnx:
        data_stream = (Node(
            id=e[0],
            level=e[1],
            label=e[2],
            parent_id=e[3]) for e in read_stream(cnx, "tu", ['id', 'level', 'label', 'parent_id']))
        Territory.build_tree(data_stream)



if __name__ == "__main__":
    build_tree_from_db()
    
    with open("tree.pickle", "wb") as file:
        pickle.dump(Territory.tree, file)
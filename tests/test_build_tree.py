import pickle

from territories import Territory, build_tree_from_db


if __name__ == "__main__":
    build_tree_from_db()
    
    with open("tree.pickle", "wb") as file:
        pickle.dump(Territory.tree, file)
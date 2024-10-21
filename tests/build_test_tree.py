from collections import namedtuple

from territories import Territory    


def build_tree():
    Node = namedtuple('Node', ('id', 'parent_id', 'label', 'level'))
    split = lambda x: (arg if arg != 'null' else None for arg in x[:-1].split('; '))
    
    with open("tree.txt", "r") as file:
        lines = file.readlines()
        stream = ([Node(*split(x) )for x in lines])
        Territory.build_tree(data_stream=stream, save_tree=False)

# bad idea
# def test_nodes():
#     a = Territory(sté, marseille)
#     b = Territory(lyon, france)
#     c = Territory(paris, nogent, pantin, lyon, lyon, metropole)
#     d = Territory(lyon, villeurbane, marseille)
#     e = Territory(rhone, idf)
#     f = Territory(idf, marseille, metropole)
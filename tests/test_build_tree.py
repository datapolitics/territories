from territories import Territory, MissingTreeCache
from territories.database import create_connection, stream_tu_table


if __name__ == "__main__":
    try:
        Territory.load_tree()
    except MissingTreeCache:
        with create_connection("crawling") as cnx:
            Territory.build_tree(data_stream=stream_tu_table(cnx))


    arrs = [node for node in Territory.tree.nodes() if node.partition_type is None]
    for arr in arrs:
        parent = Territory.tree.predecessors(arr.tree_id)
        parent = Territory.tree.get_node_data(parent.pop().tree_id)
        print(f"parent of {arr} is {parent}")
    print(arrs)


# save tree to file
# raw_tree = Territory.save_tree(return_bytes=True)

# with open("full_territorial_tree.gzip", "wb") as file:
#     file.write(gzip.compress(raw_tree))
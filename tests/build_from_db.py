import gzip

from territories import Territory
from territories.database import create_connection, stream_tu_table

if __name__ == "__main__":
    with create_connection("crawling") as cnx:
        Territory.build_tree(stream_tu_table(cnx))
        raw_tree = Territory.save_tree(return_bytes=True)
        assert raw_tree is not None
        with open("full_territorial_tree.gzip", "wb") as file:
            file.write(gzip.compress(raw_tree))

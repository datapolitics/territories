from territories import Territory, MissingTreeCache
from territories.database import create_connection, stream_tu_table


if __name__ == "__main__":
    try:
        Territory.load_tree()
    except MissingTreeCache:
        with create_connection("crawling") as cnx:
            Territory.build_tree(data_stream=stream_tu_table(cnx))
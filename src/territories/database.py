import os
import psycopg2

from dotenv import load_dotenv
from typing import Iterable, Optional
from contextlib import contextmanager

from territories.partitions import Node

load_dotenv()

user = os.environ.get("DB_USER")
pswd = os.environ.get("DB_PSWD")
port = os.environ.get("DB_PORT")
host = os.environ.get("DB_HOST")


@contextmanager
def create_connection(database: str):
    """Yield a connection to a database.
    You need to following env. variables to use this function :
    ```
    DB_USER = *your_username*
    DB_PSWD = *your_password*
    DB_PORT = 20184
    DB_HOST = postgresql-88ff04ce-oc6ad7ab2.database.cloud.ovh.net
    ```
    Args:
        database (str): The name of the database you want to connect to.

    Yields:
        A psycopg2 connection object
    """
    connection = psycopg2.connect(
        user=user,
        password=pswd,
        host=host,
        port=port,
        database=database
        )
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def borrow_connection(connection):
    with connection as c:
        try:
            cursor = c.cursor()
            yield cursor
        finally:
            cursor.close()


def read_stream(
        connection,
        table: str,
        elements: Optional[Iterable] = None,
        conditions: Optional[dict]=None,
        operator: Optional[str]=None,
        batch_size: int = 1024
        ):
    cursor = connection.cursor()
    values = []
    if elements:
        if isinstance(elements, str):
            elements = [elements]
        elements = ', '.join(elements)
    else:
        elements = '*'
    if conditions:
        is_enumeration = lambda x: isinstance(x, Iterable) and not isinstance(x, str)
        equality = lambda value: "in" if is_enumeration(value) else '='
        where = " WHERE " + f" {operator} ".join(f"{k} {equality(v)} %s" for k, v in conditions.items() if v)
        values.extend((tuple(x) if is_enumeration(x) else x for x in conditions.values()))
    else:
        where = ''
    req = f"SELECT {elements} FROM {table} {where};"
    # print(req)
    # print(values)
    cursor.itersize = batch_size
    cursor.execute(req, values)
    return cursor


def stream_tu_table(cnx) -> Iterable[Node]:
    """Stream the content of the tu table as `territories.partition.Node` objects

    Args:
        cnx : A psycopg2 cursor

    Returns:
        Iterable[Node]: Objects with id, parent_id, level and label.
    """
    data_stream = (Node(
        id=e[0],
        level=e[1],
        label=e[2],
        parent_id=e[3]) for e in read_stream(cnx, "tu", ['id', 'level', 'label', 'parent_id']))
    return data_stream



if __name__ == "__main__":
    with create_connection("crawling") as cnx:
        for element in read_stream(cnx, "tu", ['id', 'level', 'label', 'parent_id', 'postal_code']):
            print(element)
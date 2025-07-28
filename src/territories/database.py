import os
import warnings

try:
    import psycopg
except ModuleNotFoundError:
    raise Exception("Install the postgres optional dependency to load the tree from the DB (uv add 'territories[postgres]')") from None

try:
    import asyncpg
except ModuleNotFoundError:
    asyncpg = None

from dotenv import load_dotenv
from typing import Iterable, Optional, AsyncIterable
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass

from territories.partitions import Node

@dataclass
class NodeTuple:
    id: str
    level: str
    label: str
    parent_id: str | None = None
    inhabitants: int | None = None

load_dotenv()


@contextmanager
def create_connection(database: str, connection_url: str | None = None):
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
        connection_url (Optional[str]): The URL to connect to the db, as postgresql://foo@localhost:5432/bar. Will attempt to read CRAWLING_DB_URL first.

    Yields:
        A psycopg connection object
    """
    warnings.warn("You should not use this internal module, stream the database by yourself", UserWarning)
    
    connection_url = connection_url or os.environ["CRAWLING_DB_URL"]

    if connection_url:
        connection = psycopg.connect(connection_url)
    else:
        username = os.environ["DB_USER"]
        password = os.environ["DB_PSWD"]
        port = os.environ["DB_PORT"]
        hostname = os.environ["DB_HOST"]
        connection_string = os.environ["DB_URL"] or f"dbname={database} user={username} password={password} host={hostname} port={port}"
        connection = psycopg.connect(connection_string)
    try:
        yield connection
    finally:
        connection.close()


@asynccontextmanager
async def async_create_connection(database: str, connection_url: str | None = None):
    """Yield an async connection to a database using asyncpg.
    You need to following env. variables to use this function :
    ```
    DB_USER = *your_username*
    DB_PSWD = *your_password*
    DB_PORT = 20184
    DB_HOST = postgresql-88ff04ce-oc6ad7ab2.database.cloud.ovh.net
    ```
    Args:
        database (str): The name of the database you want to connect to.
        connection_url (Optional[str]): The URL to connect to the db, as postgresql://foo@localhost:5432/bar. Will attempt to read CRAWLING_DB_URL first.

    Yields:
        An asyncpg connection object
    """
    warnings.warn("You should not use this internal module, stream the database by yourself", UserWarning)
    
    if asyncpg is None:
        raise Exception("Install the async optional dependency to use async database operations (uv add 'territories[async]')")

    connection_url = connection_url or os.environ["CRAWLING_DB_URL"]

    if connection_url:
        # connection = await asyncpg.connect(connection_url.replace("postgresql://", "postgresql+psycopg://"))
        connection = await asyncpg.connect(connection_url)
    else:
        username = os.environ["DB_USER"]
        password = os.environ["DB_PSWD"]
        port = os.environ["DB_PORT"]
        hostname = os.environ["DB_HOST"]
        connection = await asyncpg.connect(
            database=database,
            user=username,
            password=password,
            host=hostname,
            port=port
        )
    try:
        yield connection
    finally:
        await connection.close()


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
    # cursor.itersize = batch_size
    cursor.execute(req, values)
    return cursor


def stream_tu_table(cnx) -> Iterable[Node]:
    """Stream the content of the tu table as `territories.partition.Node` objects

    Args:
        cnx : A psycopg2 cursor

    Returns:
        Iterable[Node]: Objects with id, parent_id, level and label.
    """
    warnings.warn("You should not use this internal module, stream the database by yourself", UserWarning)
    
    data_stream = (NodeTuple(
        id=e[0],
        level=e[1],
        label=e[2],
        parent_id=e[3],
        inhabitants=e[4]
        ) for e in read_stream(cnx, "tu", ['id', 'level', 'label', 'parent_id', 'inhabitants']))
    return iter(data_stream)


async def async_stream_tu_table(cnx) -> AsyncIterable[Node]:
    """Async stream the content of the tu table as `territories.partition.Node` objects

    Args:
        cnx : An asyncpg connection

    Returns:
        AsyncIterable[Node]: Objects with id, parent_id, level and label.
    """
    warnings.warn("You should not use this internal module, stream the database by yourself", UserWarning)
    
    if asyncpg is None:
        raise Exception("Install the async optional dependency to use async database operations (uv add 'territories[async]')")

    query = "SELECT id, level, label, parent_id, inhabitants FROM tu;"
    async for record in cnx.cursor(query):
        yield NodeTuple(
            id=record[0],
            level=record[1],
            label=record[2],
            parent_id=record[3],
            inhabitants=record[4]
        )


# Usage example for async functions:
# async def example_usage():
#     async with async_create_connection("crawling") as cnx:
#         async for node in async_stream_tu_table(cnx):
#             print(f"Node: {node.id} - {node.label}")

if __name__ == "__main__":
    with create_connection("crawling") as cnx:
        for element in read_stream(cnx, "tu", ['id', 'level', 'label', 'parent_id', 'postal_code']):
            print(element)

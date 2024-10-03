import os
import psycopg2


from dotenv import load_dotenv
from typing import Iterable, Optional
from contextlib import contextmanager

load_dotenv()

user = os.environ["DB_USER"]
pswd = os.environ["DB_PSWD"]
port = os.environ["DB_PORT"]
host = os.environ["DB_HOST"]

@contextmanager
def create_connection(database: str):
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


if __name__ == "__main__":


    with create_connection("crawling") as cnx:
        for element in read_stream(cnx, "tu", ['id', 'level', 'label', 'parent_id', 'postal_code']):
            print(element)
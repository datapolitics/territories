import os
import psycopg2


from typing import Iterable, Optional
from dotenv import load_dotenv
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


# def create(connection, table: str, elements, columns=None):
#     with borrow_connection(connection) as cursor:
#         req = f"INSERT INTO %s {'%s' if columns else ''} VALUES {elements};"
#         values = (table, columns, elements)if columns else (table, elements)
#         cursor.execute(req,  values)
#         return cursor.fetchall()

def read(
        connection,
        table: str,
        elements: Optional[Iterable] = None,
        conditions: Optional[dict]=None,
        operator: Optional[str]=None
        ):
    with borrow_connection(connection) as cursor:
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
        cursor.execute(req,  values)
        return cursor.fetchall()


# def update(connection, table: str, element: str, value: str, conditions=None, operator=None):
#     if conditions:
#         where = " WHERE " + f" {operator} ".join(f"{k} = %s" for k, v in conditions.items() if v)
#         value = None
#     with borrow_connection(connection) as cursor:
#         cursor.execute(f"UPDATE {table} SET {element} = {value} {where} ;")
#         return cursor.fetchall()

# def delete(connection, table: str, elements):
#     with borrow_connection(connection) as cursor:
#         cursor.execute(f"CREATE {elements} FROM {table};")
#         return cursor.fetchall()


# def update_db(object: dict, entity_id: str):
#     with create_connection("entity") as connection:
#         update(connection, "entity", "cluster", {"id" : entity_id})


if __name__ == "__main__":
    pass
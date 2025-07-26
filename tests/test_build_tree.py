import gzip
import pytest

from territories import Territory
from territories.database import stream_tu_table, create_connection, async_stream_tu_table, async_create_connection, NodeTuple


def test_build_tree():

    nodes = [
        NodeTuple(id='CNTRY:France', label='France', level='CNTRY', parent_id=None),
        NodeTuple(id='REG:Sud', label='Sud', level='REG', parent_id='CNTRY:France'),
        NodeTuple(id='REG:idf', label='île-de-france', level='REG', parent_id='CNTRY:France'),

        NodeTuple(id='DEP:Rhone', label='Rhône', level='DEP', parent_id='REG:Sud'),
        NodeTuple(id='DEP:metropole', label='Grand Lyon', level='DEP', parent_id='REG:Sud'),

        NodeTuple(id='COM:Pantin', label='Pantin', level='COM', parent_id="REG:idf"),
        NodeTuple(id='COM:Nogent', label='Nogent', level='COM', parent_id="REG:idf"),
        NodeTuple(id='COM:Paris', label='Paris', level='COM', parent_id="REG:idf"),

        NodeTuple(id='COM:sté', label='Saint Étienne', level='COM', parent_id="DEP:Rhone"),
        NodeTuple(id='COM:Lyon', label='Lyon', level='COM', parent_id="DEP:metropole"),
        NodeTuple(id='COM:Villeurbane', label='Villeurbane', level='COM', parent_id="DEP:metropole"),

        NodeTuple(id='COM:Marseille', label='Marseille', level='COM', parent_id="REG:Sud"),
    ]

    Territory.build_tree(nodes, save_tree=False)


# @pytest.mark.filterwarnings("ignore:You should not use this internal module")
# def test_create_from_stream():
#     with create_connection("crawling") as cnx:
#         Territory.build_tree(data_stream=stream_tu_table(cnx))


# @pytest.mark.filterwarnings("ignore:You should not use this internal module")
# @pytest.mark.asyncio
# async def test_create_from_async_stream():
#     async with async_create_connection("crawling") as cnx:
#         await Territory.async_build_tree(async_data_stream=async_stream_tu_table(cnx))


# def test_to_bytes():
#     # save tree to file
#     raw_tree = Territory.save_tree(return_bytes=True)
#     assert raw_tree is not None
#     with open("full_territorial_tree.gzip", "wb") as file:
#         file.write(gzip.compress(raw_tree))

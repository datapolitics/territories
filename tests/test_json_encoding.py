import gzip
import json
import pytest

from pydantic import BaseModel
from territories import Territory


@pytest.fixture
def load_tree():
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))



def test_parse_without_error(benchmark, load_tree):
    t = Territory.from_tu_ids("DEP:69")
    res = benchmark.pedantic(json.dumps, t, rounds=100)
    assert json.loads(res) == {
        "name": "Rhône",
        "tu_id": "DEP:69",
        "atomic": False,
        "level": "DEP",
        "postal_code": None,
        "inhabitants": 1883437
    }


def test_parse_fast(benchmark, load_tree):
    t = Territory.from_tu_ids("DEP:69")
    benchmark.pedantic(json.dumps, args=(t.descendants(), ), rounds=100)
    
    
# def test_to_list(load_tree):
#     t = Territory.from_tu_ids("DEP:69")
#     assert t.to_list() == [Territory.from_name("DEP:69")]
    

def test_pydantic(load_tree):

    class TerritoryModel(BaseModel):
        terr: Territory

    # tus = [t for t in Territory.tree.nodes() if t.name in ("Paris", "Lyon")]
    terr = Territory.from_tu_ids("DEP:75", "COM:69132")

    # TerritoryModel(terr='["DEP:69", "COM:69132"]')
    # TerritoryModel(terr='[]')
    # TerritoryModel(terr=tus)
    TerritoryModel(terr={"DEP:69", "COM:69132"})
    TerritoryModel(terr=["DEP:69", "COM:69132"])
    TerritoryModel(terr=("DEP:69", "COM:69132"))
    TerritoryModel(terr=[])
    model = TerritoryModel(terr=terr)
    
    assert model.model_dump() == {"terr": terr}
    json_dump = model.model_dump_json() 
    print(json_dump)
    json_reference = json.dumps({"terr": terr})
    assert json.loads(json_dump) == json.loads(json_reference)



if __name__ == "__main__":
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))
        
        
    class TerritoryModel(BaseModel):
        terr: Territory

    terr = Territory.from_tu_ids("DEP:75", "COM:69132")
    model = TerritoryModel(terr=terr)
    
    json_dump = model.model_dump_json() 
    print(json_dump)

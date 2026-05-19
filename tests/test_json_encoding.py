from datetime import date
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
        "inhabitants": 1883437,
    }


def test_parse_fast(benchmark, load_tree):
    t = Territory.from_tu_ids("DEP:69")
    benchmark.pedantic(json.dumps, args=(t.descendants(),), rounds=100)


# def test_to_list(load_tree):
#     t = Territory.from_tu_ids("DEP:69")
#     assert t.to_list() == [Territory.from_name("DEP:69")]


def test_pydantic(load_tree):
    class TerritoryModel(BaseModel):
        foo: int
        bar: date
        terr: Territory

    d = date(1999, 11, 1)
    model_inputs = [
        {"foo": 1, "bar": d, "terr": {"DEP:69", "COM:69132"}},
        {"foo": 2, "bar": d, "terr": ["DEP:69", "COM:69132"]},
        {"foo": 3, "bar": d, "terr": ("DEP:69", "COM:69132")},
        {"foo": 4, "bar": d, "terr": []},
        {"foo": 5, "bar": d, "terr": Territory.from_tu_ids("DEP:75", "COM:69132")},
        {"foo": 6, "bar": d, "terr": Territory.from_tu_ids("DEP:69")},
    ]

    for payload in model_inputs:
        model = TerritoryModel.model_validate(payload)
        dict_repr = {"terr": model.terr, "foo": model.foo, "bar": model.bar}

        assert model.model_dump() == dict_repr
        assert TerritoryModel.model_validate_json(model.model_dump_json()) == model
        assert TerritoryModel.model_validate(model.model_dump(mode="json")) == model
        assert TerritoryModel.model_validate(model.model_dump()) == model


def test_pydantic_serialized_roundtrip(load_tree):
    class TerritoryModel(BaseModel):
        foo: int
        bar: date
        terr: Territory

    d = date(1999, 11, 1)
    base_models = [
        TerritoryModel(foo=1, bar=d, terr=[]),
        TerritoryModel(foo=2, bar=d, terr=["DEP:69"]),
        TerritoryModel(foo=3, bar=d, terr=["DEP:75", "COM:69132"]),
    ]

    for model in base_models:
        json_payload = model.model_dump(mode="json")
        assert TerritoryModel.model_validate(json_payload) == model
        assert TerritoryModel.model_validate_json(json.dumps(json_payload)) == model


@pytest.mark.parametrize(
    "payload",
    [
        {"foo": 1, "bar": "1999-11-01", "terr": [{"name": "Paris"}]},
        {"foo": 1, "bar": "1999-11-01", "terr": [123]},
        {"foo": 1, "bar": "1999-11-01", "terr": ["DEP:75", 123]},
        {"foo": 1, "bar": "1999-11-01", "terr": [{"tu_id": "DEP:75"}, 123]},
    ],
)
def test_pydantic_invalid_payloads(payload, load_tree):
    class TerritoryModel(BaseModel):
        foo: int
        bar: date
        terr: Territory

    with pytest.raises(Exception):
        TerritoryModel.model_validate(payload)


if __name__ == "__main__":
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))

    class TerritoryModel(BaseModel):
        terr: Territory

    terr = Territory.from_tu_ids("DEP:75", "COM:69132")
    model = TerritoryModel(terr=terr)

    json_dump = model.model_dump_json()
    print(json_dump)

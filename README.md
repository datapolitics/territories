Territories
================


This package allows to represent and operate on territories.

The only assumptions of the model are :
- The entire world can be partitioned with atomic entities
- There are several partition of the world in sets of atomic entities.
- A **territory** is a collection of those atomic entities


A Territory object can be any combination of entities, such as municipalities, countries, county, länders, states, etc, as long as it belongs to the DAG of entities. The package guarantee that the representation of a territory will always be efficient. For instance, if I create a `Territory` object with all regions from a country, it will simplify it to only the country object.

## Usage

At startup, you need to initialize a tree of all known entities. Ths can be done in two ways :
- from a local file. By default, the tree is build from a file in the API_CACHE_DIR
- from a database : there are utility functions in territories.database to create a tree from the current database. You need to specify some environements variables to use such functions.


```python
from territories import Territory, MissingTreeCache
from territories.database import create_connection, stream_tu_table

try:
    Territory.load_tree()
except MissingTreeCache:
    with create_connection("crawling") as cnx:
        Territory.build_tree(data_stream=stream_tu_table(cnx))
```

The `build_tree()` function will read the TU table, and create a territory tree out of it, with all its 35099 elements



## Exemple of a potential usage of such a package

```python
# es code are received from the UI, for instance
topic_territory = Territory.from_tu_ids("COM:234", "COM:943", "DEP:23")


# lowest common ancestor of the territorial units
lca = topic_territory.lowest_common_ancestor()

# all ancestors of the territorial units
all_ancestors = topic_territory.ancestors()

# union of the territories
bu_teritory = Territory.union(*(topic.territory for topic in bu))

for article in articles:
    for territory in article.territories:
        if territory in bu_territory:
            article.send() # send article in brief
        if territory in topic_territory:
            # do something else
```


## The pydantic feature


Territories are valid Pydantic types. It means you can validate input data with type hints ; in a web server for instance. Exemples to come.



## How to develop

This package depends on [uv](https://docs.astral.sh/uv/getting-started/installation/). This is a better and simpler package manager than pip, and I strongly encourage you to make the switch.

- run ```uv sync``` to create a virtual env and install all dependencies.
- update the code in **src/territories/**
- add tests in **tests/**
- run the tests with `uv run pytest`
- you can quickly iterate and test with the notebooks in **docs/usage.ipynb** by installing the package in the virtual env with `uv sync`

## Tests

The tests checks the behavior of **Territory** objects. You can change whatever you want internaly as long as the tests passes.
The tests relies on the **full_territorial_tree.gzip** file to load a territorial tree. If you need to update it, please use the `test_build_tree.py` file.



## Deployment

CI automatically deploy the package when there are pushes on the main branch. Do not deploy yourself.

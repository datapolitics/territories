Territories
================


This package allows to represent and operate on territories.

The only assumptions of the model are :
- the entire world can be partitioned with atomic entities
- those atomic entities can be grouped in larger entities into a Directed Acyclic Graph (DAG)
- a **territory** is a collection of those atomic entities


A Territory object can be any combination of entities, such as municipalities, countries, county, länders, states, etc, as long as it belongs in the DAG of entities.

## Usage

At startup, build a tree of all known entities.
```python
# buiding the reference territory tree
tree = nx.DiGraph([
    (france, sud),
    (france, idf),

    (idf, nogent),
    (idf, pantin),
    (idf, paris),

    (sud, marseille),
    (sud, rhone),

    (rhone, metropole),
    (rhone, sté),

    (metropole, villeurbane),
    (metropole, lyon),
])

Territory.assign_tree(tree)
```

But it can be done as simply as :
```python
from territories import build_tree_from_db

build_tree_from_db()
```

This function will read the TU table, and create a territory tree out of it, with all its 35099 elements. It only takes a few seconds btw. (it takes 20s to construct the perfect hash function to go from es_code -> node in the tree)

```python
def build_tree_from_db():
    with create_connection("crawling") as cnx:
        data_stream = (Node(
            id=e[0],
            level=e[1],
            label=e[2],
            parent_id=e[3]) for e in read_stream(cnx, "tu", ['id', 'level', 'label', 'parent_id']))
        Territory.build_tree(data_stream)
```

## Exemple of a potential usage of such a package.

```python
# es code are received from the UI, for instance
topic_territory = Territory.from_es_codes("COM:234", "COM:943", "DEP:23")

# union of the territories
bu_teritory = Territory.union(*(topic.terriotry for topic in bu))

for article in articles:
    for territory in article.territories:
        if territory in bu_territory:
            article.send() # send article in brief
        if territory in topic_territory:
            # do something else

# filter territories in an ElasticSearch query
query = {"ids" : {
    "values" : ids
    }}

query['bool']['should'].extend(topic_territory.to_es_query())

documents = HArticle.search(using=target.es, index=target.index)\
        .query(query)\
        .execute()
```


## Questions left to answer

If a child has several parents (Grand Lyon or département du Rhône, île-de-france or IDF mobilité), how should we chose ?


## Package structure

I used [this](https://py-pkgs.org/01-introduction) website as the main ressource for the structure of the package. Also [this](https://docs.python-guide.org/writing/structure/) one is useful.


## Tests

The tests checks the behavior of the package. You can change whatever you want internaly as long as the tests passes.

```sh
$ pip install -r requirements-dev.txt
$ pip install .
$ pytest

>>> tests/test_interface.py .....   [ 45%]
>>> tests/test_operators ......     [100%]
>>> 11 passed in 0.14s 
```


## Deployment

CI to come
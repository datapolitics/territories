# LLM.md — Project Guide for AI Agents

## What is this project?

`territories` is a Python library for representing and operating on **territories** — collections of administrative entities (communes, departments, regions, countries) organized in a directed acyclic graph (DAG). It models the French administrative hierarchy.

A `Territory` is any combination of these entities. The library guarantees a **canonical minimal representation**: if you create a Territory with all departments of a region, it auto-simplifies to just the region.

## Key concepts

- **Territorial tree**: A `rustworkx.PyDiGraph` shared as class-level state on `Territory`. Must be initialized before creating any `Territory` instances.
- **TerritorialUnit**: A frozen dataclass representing a single node (e.g. one commune, one department). Has a `tu_id` string like `"COM:69132"` or `"DEP:69"`.
- **Partition**: An enum of hierarchy levels: `EMPTY(0)`, `ARR(1)`, `COM(2)`, `DEP(3)`, `REG(4)`, `CNTRY(5)`, `UE(6)`.
- **Minimization**: Every `Territory` stores a `frozenset[TerritorialUnit]` that is always in minimal form — no redundant nodes. The `minimize()` method enforces this.
- **Set algebra**: Territories support `|` (union), `&` (intersection), `-` (subtraction), `+` (union), `in` (containment), `==`, and `hash`.

## Project layout

```
src/territories/
  __init__.py          # Public API: Territory, TerritorialUnit, Partition, exceptions
  territories.py       # Core Territory class (~1000 lines) — the heart of the project
  partitions.py        # TerritorialUnit dataclass, Partition enum, Node protocol
  exceptions.py        # MissingTreeException, MissingTreeCache, NotOnTreeError, EmptyTerritoryError
  database.py          # PostgreSQL loading utilities (deprecated, sync + async)
  data/
    epci_to_comm.json  # Legacy code mapping (UUIDs/EPCI → current TU IDs)

tests/
  test_interface.py    # Main integration tests + benchmarks (small tree + full 35K-node tree)
  test_operators.py    # Exhaustive algebraic property tests (commutativity, associativity, De Morgan, etc.)
  test_algorithms.py   # LCA and ancestor traversal tests
  test_build_tree.py   # Tree construction pipeline test
  test_json_encoding.py # JSON serialization + Pydantic integration tests
  test_data.py         # Legacy code mapping test
  full_territorial_tree.gzip  # Serialized 35K-node production tree (for tests)
  entities.gzip        # 45K lines of TU ID sets (for benchmarks)

profiles/
  profile_from_tu_ids.py  # cProfile script for from_tu_ids performance
```

## How to run

```
uv run pytest              # all tests + benchmarks
uv run pytest --benchmark-disable  # tests only, skip benchmarks
uv run pytest --benchmark-only     # benchmarks only
```

## Architecture details

### Tree lifecycle

The tree is class-level state on `Territory`. It must be initialized before use via one of:
- `Territory.build_tree(data_stream)` — builds from an iterable of `Node`-protocol objects
- `Territory.load_tree(filepath)` — loads from a pickled cache file
- `Territory.load_tree_from_bytes(data)` — loads from bytes (used in tests with gzip)
- `Territory.assign_tree(graph)` — directly assigns a PyDiGraph (tests only)

`Territory.reset()` clears all state including the `_ancestors` LRU cache.

### Territory creation

- `Territory(*territorial_units)` — from TerritorialUnit objects (calls `minimize`)
- `Territory.from_tu_ids(*codes)` — from string codes like `"DEP:69"`, supports nested iterables via `collapse()`. Uses `LEGACY_CODES` for backward-compatible UUID mapping. Internally uses `_from_indices()` fast path.
- `Territory.from_name(tu_id)` — returns a single TerritorialUnit by its ID

### The minimize algorithm

`minimize()` is the most performance-critical method. It uses a **bottom-up** algorithm:
1. **Remove descendants**: For each item, check if any of its ancestors are also in the set. If so, remove the item (the ancestor subsumes it).
2. **Merge siblings**: Iteratively check if all children of a parent are present. If so, replace them with the parent. Repeat until stable.

This replaced an earlier top-down approach that had a bug causing it to traverse the entire tree on every call.

### Performance-sensitive areas

- `minimize()` is called on every Territory creation. The bottom-up approach is O(items × depth) rather than O(tree_size).
- `_ancestors()` is an LRU-cached wrapper around `rx.ancestors()` with `maxsize=1024`. Don't make it unbounded (web server memory concerns).
- `from_tu_ids()` uses `_from_indices()` to bypass TerritorialUnit→tree_id roundtrips.
- The full tree has ~35K nodes and ~35K edges. The benchmark file has ~45K lines of territories to create.

### Pydantic integration

`Territory` is a valid Pydantic type (conditional on pydantic being installed). It validates from `list[str]`, `str`, or `Territory` instances and serializes to a list of TU dicts.

## Dependencies

- **rustworkx** — Rust-backed graph library (PyDiGraph for the territorial DAG)
- **more-itertools** — `collapse()` for flattening nested iterables
- **json-fix** — enables `__json__()` protocol for json.dumps
- **python-dotenv** — env variable loading
- **pydantic** (optional) — web framework validation support
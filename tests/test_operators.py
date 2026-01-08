import gzip
import pytest

import rustworkx as rx

from itertools import product

from territories import Territory
from territories.partitions import TerritorialUnit, Partition


lyon = TerritorialUnit("Lyon", tu_id="Lyon")
marseille = TerritorialUnit("Marseille", tu_id="Marseille")
paris = TerritorialUnit("Paris", tu_id="Paris")
nogent = TerritorialUnit("Nogent", tu_id="Nogent")
pantin = TerritorialUnit("Pantin", tu_id="Pantin")
villeurbane = TerritorialUnit("Villeurbane", tu_id="Villeurbane")
sté = TerritorialUnit("Saint Etienne", tu_id="Etienne")

metropole = TerritorialUnit("Grand Lyon", "metro", False, Partition.DEP)

sud = TerritorialUnit("Sud", "Sud", False, Partition.REG)
idf = TerritorialUnit("Île-de-France", "idf", False, Partition.REG)
rhone = TerritorialUnit("Rhône", "Rhône", False, Partition.DEP)

france = TerritorialUnit("France", "France", False, Partition.CNTRY)


entities = (france, sud, idf, rhone, metropole, nogent, pantin, paris, marseille, sté, villeurbane, lyon)

tree: rx.PyDiGraph[TerritorialUnit, None] = rx.PyDiGraph()
entities_indices = tree.add_nodes_from(entities)

mapper = {o: idx for o, idx in zip(entities, entities_indices)}
edges = [
    (france, idf),
    (france, sud),
    (idf, nogent),
    (idf, pantin),
    (idf, paris),
    (sud, marseille),
    (sud, rhone),
    (rhone, metropole),
    (rhone, sté),
    (metropole, villeurbane),
    (metropole, lyon),
]

tree.add_edges_from([(mapper[parent], mapper[child], None) for parent, child in edges])

Territory.assign_tree(tree)

a = Territory(sté, marseille)
b = Territory(lyon, france)
c = Territory(paris, nogent, pantin, lyon, lyon, metropole)
d = Territory(lyon, villeurbane, marseille)
e = Territory(rhone, idf)
f = Territory(idf, marseille, metropole)

all_nodes = [Territory(n) for n in tree.nodes()]
examples = (a, b, c, d, e, f)


@pytest.fixture
def load_tree():
    Territory.reset()
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))


@pytest.fixture
def load_tree():
    Territory.reset()
    with open("tests/full_territorial_tree.gzip", "rb") as file:
        Territory.load_tree_from_bytes(gzip.decompress(file.read()))


def test_equality():
    Territory.assign_tree(tree)

    assert b == Territory(france)

    for i, j in product(examples, examples):
        assert (j == i) == (i == j)


def test_addition():
    Territory.assign_tree(tree)

    assert d + a == Territory(sud)
    assert c + a == Territory(idf, sud)
    assert d + c == Territory(metropole, marseille, idf)

    for i, j in product(examples, examples):
        assert i + j == j + i


def test_inclusion():
    Territory.assign_tree(tree)

    assert a in b
    assert a in c + a
    assert a not in d
    assert d in f

    for i, j in zip(examples, examples):
        assert j in i
        assert i in j


def test_union():
    Territory.assign_tree(tree)

    assert a | d == Territory(sud)
    assert c | d == Territory(idf, marseille, metropole)

    for i, j in product(examples, examples):
        assert i | j == j | i

    h = Territory(paris, nogent, pantin, lyon)
    hu = Territory(paris, nogent) | Territory(pantin, lyon)
    assert h == hu

    assert Territory.union(a, d) == Territory(sud)
    assert Territory.union([a, d]) == Territory(sud)
    assert Territory.union({a, d}) == Territory(sud)


def test_intersection():
    Territory.assign_tree(tree)

    assert a & b == a
    assert a & d == Territory(marseille)
    assert e & f == Territory(idf, metropole)

    for i, j in product(examples, examples):
        assert i & j == j & i


@pytest.fixture
def complex_territory(load_tree):
    with open("tests/big_territory.txt", "r") as f:
        file_ids = {s.strip() for s in f.readlines()} - {""}
    return Territory.from_tu_ids(file_ids)


@pytest.fixture
def simple_territory(load_tree):
    return Territory.from_tu_ids("REG:76")


def test_benchmark_intersection(complex_territory, simple_territory, benchmark):
    benchmark.pedantic(lambda: simple_territory & complex_territory, rounds=32, iterations=1)


def test_benchmark_union(complex_territory, simple_territory, benchmark):
    benchmark.pedantic(lambda: simple_territory | complex_territory, rounds=32, iterations=1)


def test_benchmark_subtraction(complex_territory, simple_territory, benchmark):
    benchmark.pedantic(lambda: complex_territory - simple_territory, rounds=32, iterations=1)


def test_benchmark_addition(complex_territory, simple_territory, benchmark):
    benchmark.pedantic(lambda: simple_territory + complex_territory, rounds=32, iterations=1)


def test_benchmark_contains(complex_territory, simple_territory, benchmark):
    benchmark.pedantic(lambda: simple_territory in complex_territory, rounds=32, iterations=1)


def test_substraction():
    Territory.assign_tree(tree)

    assert a - b == Territory()
    assert b - a == Territory(metropole, idf)


# =============================================================================
# Property-based tests for algebraic properties
# =============================================================================


class TestEqualityProperties:
    """Tests for equality properties: reflexivity, symmetry, transitivity."""

    def test_reflexivity(self):
        """a == a for all territories."""
        Territory.assign_tree(tree)
        for t in examples:
            assert t == t

    def test_symmetry(self):
        """(a == b) == (b == a) for all territories."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            assert (i == j) == (j == i)

    def test_transitivity(self):
        """If a == b and b == c, then a == c."""
        Territory.assign_tree(tree)
        # Create equivalent territories
        t1 = Territory(france)
        t2 = Territory(lyon, france)  # simplifies to france
        t3 = Territory(idf, sud)  # also simplifies to france
        assert t1 == t2
        assert t2 == t3
        assert t1 == t3

    def test_equality_with_empty(self):
        """Empty territory equals itself."""
        Territory.assign_tree(tree)
        empty1 = Territory()
        empty2 = Territory()
        assert empty1 == empty2


class TestUnionProperties:
    """Tests for union properties: commutativity, associativity, identity, idempotence."""

    def test_commutativity(self):
        """a | b == b | a for all territories."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            assert i | j == j | i

    def test_associativity(self):
        """(a | b) | c == a | (b | c) for all territories."""
        Territory.assign_tree(tree)
        for i, j, k in product(examples, examples, examples):
            assert (i | j) | k == i | (j | k)

    def test_identity(self):
        """a | empty == a for all territories."""
        Territory.assign_tree(tree)
        empty = Territory()
        for t in examples:
            assert t | empty == t
            assert empty | t == t

    def test_idempotence(self):
        """a | a == a for all territories."""
        Territory.assign_tree(tree)
        for t in examples:
            assert t | t == t

    def test_absorption_with_superset(self):
        """If a is contained in b, then a | b == b."""
        Territory.assign_tree(tree)
        # a (sté, marseille) is contained in b (france)
        assert a in b
        assert a | b == b

    def test_union_class_method(self):
        """Territory.union works the same as | operator."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            assert Territory.union(i, j) == i | j

    def test_union_multiple(self):
        """Union of multiple territories."""
        Territory.assign_tree(tree)
        assert Territory.union(a, c, d) == a | c | d
        assert Territory.union([a, c, d]) == a | c | d


class TestIntersectionProperties:
    """Tests for intersection properties: commutativity, associativity, identity, idempotence."""

    def test_commutativity(self):
        """a & b == b & a for all territories."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            assert i & j == j & i

    def test_associativity(self):
        """(a & b) & c == a & (b & c) for all territories."""
        Territory.assign_tree(tree)
        for i, j, k in product(examples, examples, examples):
            assert (i & j) & k == i & (j & k)

    def test_idempotence(self):
        """a & a == a for all territories."""
        Territory.assign_tree(tree)
        for t in examples:
            assert t & t == t

    def test_intersection_with_empty(self):
        """a & empty == empty for all territories."""
        Territory.assign_tree(tree)
        empty = Territory()
        for t in examples:
            assert t & empty == empty
            assert empty & t == empty

    def test_absorption_with_subset(self):
        """If a is contained in b, then a & b == a."""
        Territory.assign_tree(tree)
        # a (sté, marseille) is contained in b (france)
        assert a in b
        assert a & b == a

    def test_intersection_class_method(self):
        """Territory.intersection works the same as & operator."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            assert Territory.intersection(i, j) == i & j

    def test_intersection_multiple(self):
        """Intersection of multiple territories."""
        Territory.assign_tree(tree)
        assert Territory.intersection(a, b, e) == a & b & e


class TestAdditionProperties:
    """Tests for addition (combines then minimizes): commutativity, associativity."""

    def test_commutativity(self):
        """a + b == b + a for all territories."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            assert i + j == j + i

    def test_associativity(self):
        """(a + b) + c == a + (b + c) for all territories."""
        Territory.assign_tree(tree)
        for i, j, k in product(examples, examples, examples):
            assert (i + j) + k == i + (j + k)

    def test_identity(self):
        """a + empty == a for all territories."""
        Territory.assign_tree(tree)
        empty = Territory()
        for t in examples:
            assert t + empty == t
            assert empty + t == t

    def test_addition_equals_union(self):
        """Addition should produce the same result as union."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            assert i + j == i | j


class TestSubtractionProperties:
    """Tests for subtraction properties."""

    def test_known(self, load_tree):
        some_municipalities = [
            'COM:64559',
            'COM:64558',
            'COM:64547',
            'COM:64546',
            'COM:64476',
            'COM:64468',
            'COM:64441',
            'COM:64437',
            'COM:64436',
            'COM:64435',
            'COM:64432',
            'COM:69123'
        ]
        dep = Territory.from_tu_ids("DEP:64")
        t = Territory.from_tu_ids(some_municipalities)
        assert t - dep == Territory.from_tu_ids("COM:69123")

    def test_self_subtraction(self):
        """a - a == empty for all territories."""
        Territory.assign_tree(tree)
        empty = Territory()
        for t in examples:
            assert t - t == empty

    def test_subtraction_of_empty(self):
        """a - empty == a for all territories."""
        Territory.assign_tree(tree)
        empty = Territory()
        for t in examples:
            assert t - empty == t

    def test_empty_minus_anything(self):
        """empty - a == empty for all territories."""
        Territory.assign_tree(tree)
        empty = Territory()
        for t in examples:
            assert empty - t == empty

    def test_subtraction_of_superset(self):
        """If a is contained in b, then a - b == empty."""
        Territory.assign_tree(tree)
        empty = Territory()
        # a (sté, marseille) is contained in b (france)
        assert a in b
        assert a - b == empty

    def test_subtraction_of_superset_bis(self):
        """If a is contained in b, then a - b == empty."""
        Territory.assign_tree(tree)
        for a, b in product(all_nodes, all_nodes):
            if a in b:
                assert (a - b).is_empty()

    def test_subtraction_complement(self):
        """(a | b) - b should be contained in a."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            result = (i | j) - j
            assert result in i or result == Territory()


class TestContainmentProperties:
    """Tests for containment (in) properties: reflexivity, antisymmetry, transitivity."""

    def test_reflexivity(self):
        """a in a for all territories."""
        Territory.assign_tree(tree)
        for t in examples:
            assert t in t

    def test_antisymmetry(self):
        """If a in b and b in a, then a == b."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            if i in j and j in i:
                assert i == j

    def test_transitivity(self):
        """If a in b and b in c, then a in c."""
        Territory.assign_tree(tree)
        # Create a chain: lyon < metropole < rhone < sud < france
        t_lyon = Territory(lyon)
        t_metro = Territory(metropole)
        t_rhone = Territory(rhone)
        t_sud = Territory(sud)
        t_france = Territory(france)

        assert t_lyon in t_metro
        assert t_metro in t_rhone
        assert t_rhone in t_sud
        assert t_sud in t_france
        # Transitivity
        assert t_lyon in t_rhone
        assert t_lyon in t_sud
        assert t_lyon in t_france
        assert t_metro in t_sud
        assert t_metro in t_france
        assert t_rhone in t_france

    def test_empty_contained_in_all(self):
        """Empty territory is contained in all territories."""
        Territory.assign_tree(tree)
        empty = Territory()
        for t in examples:
            assert empty in t

    def test_all_contained_in_root(self):
        """All territories are contained in the root (france)."""
        Territory.assign_tree(tree)
        root = Territory(france)
        for t in examples:
            assert t in root


class TestDistributiveLaws:
    """Tests for distributive laws between union and intersection."""

    def test_union_over_intersection(self):
        """a | (b & c) == (a | b) & (a | c)."""
        Territory.assign_tree(tree)
        for i, j, k in product(examples, examples, examples):
            left = i | (j & k)
            right = (i | j) & (i | k)
            assert left == right

    def test_intersection_over_union(self):
        """a & (b | c) == (a & b) | (a & c)."""
        Territory.assign_tree(tree)
        for i, j, k in product(examples, examples, examples):
            left = i & (j | k)
            right = (i & j) | (i & k)
            assert left == right


class TestSubtractionAdditionalProperties:
    """Additional tests for subtraction behavior.

    The classical De Morgan laws hold for territories:
        - `a - (b | c) == (a - b) & (a - c)`
        - `a - (b & c) == (a - b) | (a - c)`

    Example:
        i = {Marseille, Saint Etienne}
        j = {Marseille, Grand Lyon}
        k = {Rhône, Île-de-France}
        j | k = France
        i - (j | k) = ø (since i is contained in France)
        (i - j) = Saint Etienne
        (i - k) = Marseille
        (i - j) & (i - k) = Marseille & Saint Etienne = ø ✓
    """

    def test_de_morgan_law_1(self):
        """Test that a - (b | c) == (a - b) & (a - c)."""
        Territory.assign_tree(tree)
        for i, j, k in product(examples, examples, examples):
            assert i - (j | k) == (i - j) & (i - k)

    def test_de_morgan_law_2(self):
        """Test that a - (b & c) == (a - b) | (a - c)."""
        Territory.assign_tree(tree)
        for i, j, k in product(examples, examples, examples):
            assert i - (j & k) == (i - j) | (i - k)

    def test_subtraction_preserves_containment(self):
        """If a is contained in b, then (c - b) is contained in (c - a)."""
        Territory.assign_tree(tree)
        # lyon is contained in metropole
        t_lyon = Territory(lyon)
        t_metro = Territory(metropole)
        assert t_lyon in t_metro

        # rhone - metropole should be contained in rhone - lyon
        t_rhone = Territory(rhone)
        result1 = t_rhone - t_metro  # rhone minus metropole = sté
        result2 = t_rhone - t_lyon  # rhone minus lyon = sté | villeurbane
        assert result1 in result2

    def test_subtraction_specific_cases(self):
        """Test specific subtraction results."""
        Territory.assign_tree(tree)
        # b - a where a is contained in b
        # b (france) - a (sté, marseille) = france without those two
        result = b - a
        assert Territory(sté) not in result or result == Territory()
        assert Territory(marseille) not in result or result == Territory()

    def test_subtraction_with_overlapping_territories(self):
        """Test subtraction when territories partially overlap."""
        Territory.assign_tree(tree)
        # d = lyon, villeurbane, marseille (simplifies to metropole, marseille)
        # c = paris, nogent, pantin, lyon, metropole (simplifies to idf, metropole)
        # d - c should remove metropole, leaving marseille
        result = d - c
        assert result == Territory(marseille)


class TestHashProperties:
    """Tests for hash consistency with equality."""

    def test_equal_territories_have_equal_hashes(self):
        """If a == b, then hash(a) == hash(b)."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            if i == j:
                assert hash(i) == hash(j)

    def test_hash_is_consistent(self):
        """Hash of a territory doesn't change."""
        Territory.assign_tree(tree)
        for t in examples:
            h1 = hash(t)
            h2 = hash(t)
            assert h1 == h2

    def test_territories_usable_in_sets(self):
        """Territories can be used in sets."""
        Territory.assign_tree(tree)
        s = {a, b, c, d, e, f}
        assert len(s) == len(examples)
        for t in examples:
            assert t in s

    def test_territories_usable_as_dict_keys(self):
        """Territories can be used as dictionary keys."""
        Territory.assign_tree(tree)
        d = {t: i for i, t in enumerate(examples)}
        assert len(d) == len(examples)
        for i, t in enumerate(examples):
            assert d[t] == i


class TestBooleanProperties:
    """Tests for boolean conversion."""

    def test_empty_is_falsy(self):
        """Empty territory is falsy."""
        Territory.assign_tree(tree)
        empty = Territory()
        assert not empty
        assert bool(empty) is False

    def test_non_empty_is_truthy(self):
        """Non-empty territories are truthy."""
        Territory.assign_tree(tree)
        for t in examples:
            assert t
            assert bool(t) is True


class TestIterationAndLength:
    """Tests for iteration and length operations."""

    def test_len_consistency(self):
        """Length should be the number of minimal territorial units."""
        Territory.assign_tree(tree)
        # b simplifies to france
        assert len(b) == 1
        # a has sté and marseille
        assert len(a) == 2

    def test_iteration(self):
        """Iteration should yield territorial units."""
        Territory.assign_tree(tree)
        for t in examples:
            units = list(t)
            assert len(units) == len(t)
            for unit in units:
                assert isinstance(unit, TerritorialUnit)


class TestSpecificScenarios:
    """Tests for specific scenarios and edge cases."""

    def test_simplification(self):
        """Territory with all children simplifies to parent."""
        Territory.assign_tree(tree)
        # All children of idf: nogent, pantin, paris
        all_idf_children = Territory(nogent, pantin, paris)
        assert all_idf_children == Territory(idf)

    def test_duplicate_units_simplified(self):
        """Duplicate units are simplified."""
        Territory.assign_tree(tree)
        t1 = Territory(lyon, lyon, lyon)
        t2 = Territory(lyon)
        assert t1 == t2

    def test_parent_absorbs_child(self):
        """Adding parent to child results in just parent."""
        Territory.assign_tree(tree)
        t1 = Territory(lyon, metropole)
        t2 = Territory(metropole)
        assert t1 == t2

    def test_disjoint_territories(self):
        """Intersection of disjoint territories is empty."""
        Territory.assign_tree(tree)
        # nogent (in idf) and marseille (in sud) are disjoint
        t1 = Territory(nogent)
        t2 = Territory(marseille)
        assert t1 & t2 == Territory()

    def test_union_then_intersection_identity(self):
        """(a | b) & a == a when a and b are related."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            result = (i | j) & i
            # Result should always equal i (since i is contained in i | j)
            assert result == i

    def test_subtraction_then_union_identity(self):
        """(a - b) | (a & b) == a."""
        Territory.assign_tree(tree)
        for i, j in product(examples, examples):
            left = (i - j) | (i & j)
            assert left == i

from territories import Territory
from territories.database import NodeTuple
from territories.territories import _acquire_file_lock, _release_file_lock


def test_build_tree(tmp_path):
    nodes = [
        NodeTuple(id="CNTRY:France", label="France", level="CNTRY", parent_id=None),
        NodeTuple(id="REG:Sud", label="Sud", level="REG", parent_id="CNTRY:France"),
        NodeTuple(id="REG:idf", label="île-de-france", level="REG", parent_id="CNTRY:France"),
        NodeTuple(id="DEP:Rhone", label="Rhône", level="DEP", parent_id="REG:Sud"),
        NodeTuple(id="DEP:metropole", label="Grand Lyon", level="DEP", parent_id="REG:Sud"),
        NodeTuple(id="COM:Pantin", label="Pantin", level="COM", parent_id="REG:idf"),
        NodeTuple(id="COM:Nogent", label="Nogent", level="COM", parent_id="REG:idf"),
        NodeTuple(id="COM:Paris", label="Paris", level="COM", parent_id="REG:idf"),
        NodeTuple(id="COM:sté", label="Saint Étienne", level="COM", parent_id="DEP:Rhone"),
        NodeTuple(id="COM:Lyon", label="Lyon", level="COM", parent_id="DEP:metropole"),
        NodeTuple(id="COM:Villeurbane", label="Villeurbane", level="COM", parent_id="DEP:metropole"),
        NodeTuple(id="COM:Marseille", label="Marseille", level="COM", parent_id="REG:Sud"),
    ]

    cache_path = tmp_path / "foo.pickle"
    Territory.build_tree(nodes, save_tree=True, filepath=str(cache_path))

    assert cache_path.is_file()


def test_file_lock_is_exclusive(tmp_path):
    lock_path = tmp_path / "tree.lock"
    lock_path.write_bytes(b"\0")

    with open(lock_path, "r+b") as first_fd, open(lock_path, "r+b") as second_fd:
        assert _acquire_file_lock(first_fd, blocking=False)
        assert not _acquire_file_lock(second_fd, blocking=False)

        _release_file_lock(first_fd)
        assert _acquire_file_lock(second_fd, blocking=False)
        _release_file_lock(second_fd)

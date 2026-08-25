from multiprocessing import get_context
from threading import Timer

from filelock import FileLock

from territories import Territory
from territories.database import NodeTuple

NODES = [
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


def _hold_lock(lock_path, ready, release):
    with FileLock(lock_path):
        ready.set()
        release.wait(timeout=5)


def _unexpected_nodes():
    raise AssertionError("A waiting process must load the cache without consuming its data stream")
    yield


def test_build_tree(tmp_path):
    cache_path = tmp_path / "foo.pickle"
    Territory.build_tree(NODES, save_tree=True, filepath=str(cache_path))

    assert cache_path.is_file()


def test_build_tree_waits_for_another_process(tmp_path):
    cache_path = tmp_path / "foo.pickle"
    Territory.build_tree(NODES, save_tree=True, filepath=str(cache_path))

    context = get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_lock, args=(cache_path.with_suffix(".lock"), ready, release))
    process.start()

    timer = Timer(0.2, release.set)
    try:
        assert ready.wait(timeout=5)
        timer.start()
        Territory.build_tree(_unexpected_nodes(), save_tree=True, filepath=str(cache_path))
    finally:
        release.set()
        if timer.is_alive():
            timer.join()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()

    assert process.exitcode == 0

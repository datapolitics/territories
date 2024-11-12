import cProfile
import pstats
from random import sample
from territories import Territory, Partition

Territory.load_tree()

s = sample(Territory.tree.nodes(), 1000)
ter = Territory.from_names(*(ter.tu_id for ter in s))

names = [tu.tu_id for tu in ter.descendants(include_itself=True) if tu.partition_type == Partition.COM]


with cProfile.Profile() as pr:
    ter = Territory.from_names(*names)
    pr.create_stats()
    ps = pstats.Stats(pr).sort_stats("cumtime")

ps.print_stats()
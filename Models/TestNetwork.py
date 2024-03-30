#!/usr/bin/python3
from Network import Network

Net = Network(size=2, weigths = [[0, 1], [1, 0]])
assert(Net.size == 2)
assert(Net.weigths == [[0, 1], [1, 0]])


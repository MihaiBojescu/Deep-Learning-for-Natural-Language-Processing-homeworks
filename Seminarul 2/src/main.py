#!/usr/bin/env python3

from nltk import CFG, ChartParser

# Define the cfg grammar.
grammar = CFG.fromstring(
    """
S -> VRA
VRA -> VRN VBA
VRN -> VER NOU
VBA -> VRB ADV
VRB -> VER VEB

VER -> 'Flying'
VER -> 'can'
NOU -> 'planes'
VEB -> 'be'
ADV -> 'dangerous'
"""
)


sentence = "Flying planes can be dangerous".split(" ")

cp = ChartParser(grammar)

for tree in cp.parse_all(sentence):
    print(tree)

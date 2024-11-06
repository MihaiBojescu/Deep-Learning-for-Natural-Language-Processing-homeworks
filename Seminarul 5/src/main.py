#!/usr/bin/env python

from tasks.task1 import task1
from tasks.task2 import task2
from tasks.task3 import task3
from tasks.task4 import task4


def main():
    bow_matrix, terms = task1()
    task2(bow_matrix, terms)
    task3(bow_matrix, terms)
    task4(bow_matrix, terms)


if __name__ == "__main__":
    main()

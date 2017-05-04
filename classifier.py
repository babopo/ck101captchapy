# -*- coding: utf-8 -*-
# 对字符分类


def sort(argument):
    switcher = {
        1: "2",
        2: "3",
        3: "4",
        4: "6",
        5: "7",
        6: "8",
        7: "9",
        8: "B",
        9: "C",
        10: "E",
        11: "F",
        12: "G",
        13: "H",
        14: "J",
        15: "K",
        16: "M",
        17: "P",
        18: "Q",
        19: "R",
        20: "T",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
    }
    return switcher.get(argument, "")

# Utility script to generate macros from
# the triangle quadrature rules retrieved from the appendix of
# F.D. Witherden, , P.E. Vincent, "On the identification of symmetric quadrature rules for finite element methods"
# Department of Aeronautics, Imperial College London, SW7 2AZ, United Kingdom

import sys
from pathlib import PurePath
import re


def align_numeric_string(s, width):
    if s[0] == '-':
        return s.ljust(width)
    else:
        return " " + s.ljust(width - 1)


pattern = """
MAKE_QUADRATURE_RULE(STRENGTH, NUM,
     ({ X }),
     ({ Y }),
     ({ W })
)
"""


def make_macro(strength, num, X, Y, W):
    macro = pattern.replace('STRENGTH', strength)
    macro = macro.replace('NUM', num)
    macro = macro.replace('X', X)
    macro = macro.replace('Y', Y)
    macro = macro.replace('W', W)
    return macro


def main():
    macro_dictionary = {}

    for filename in sys.argv[1:]:
        X = []
        Y = []
        W = []

        stem = PurePath(filename).stem
        regex = re.fullmatch("([0-9]*)-([0-9]*)", stem)
        if not regex:
            sys.exit('Could not determine strength and number of points from filename.')

        strength = regex.group(1)
        num_points = regex.group(2)

        for line in open(filename):
            (x, y, w) = line.split()
            X.append(x)
            Y.append(y)
            W.append(w)

        digits = map(lambda x: len(x), X + Y + W)
        max_digits = max(digits)

        X_formatted = "{ %s }" % ", ".join(map(lambda s: align_numeric_string(s, max_digits), X))
        Y_formatted = "{ %s }" % ", ".join(map(lambda s: align_numeric_string(s, max_digits), Y))
        W_formatted = "{ %s }" % ", ".join(map(lambda s: align_numeric_string(s, max_digits), W))

        macro_dictionary[(int(strength), int(num_points))] = make_macro(strength, num_points, X_formatted, Y_formatted, W_formatted)

    sorted_macro_keys = sorted(macro_dictionary.keys())
    macros = map(lambda key: macro_dictionary[key], sorted_macro_keys)

    for macro in macros:
        print(macro)



if __name__ == "__main__":
    main()

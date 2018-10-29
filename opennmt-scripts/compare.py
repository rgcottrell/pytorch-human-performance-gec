#!/usr/bin/env python

import argparse

def main(opt):
    with open('../test/translate.txt', 'r') as source_file, open('../test/pred.txt', 'r') as pred_file:
        for source, target in zip(source_file, pred_file):
            print("S: {}\nP: {}\n".format(source.strip(), target.strip()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='compare.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opt = parser.parse_args()
    main(opt)

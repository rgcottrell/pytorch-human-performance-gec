#!/usr/bin/env python

import argparse
import os

def process(src_dir, out_dir, dataset):
    entry_filename = os.path.join(src_dir, 'entries.' + dataset)
    src_filename = os.path.join(out_dir, 'lang8-' + dataset + '-src.txt')
    tgt_filename = os.path.join(out_dir, 'lang8-' + dataset + '-tgt.txt')
    with open(entry_filename, 'r') as entry_file, open(src_filename, 'w') as src_out, open(tgt_filename, 'w') as tgt_out:
        for line in entry_file:
            line = line.strip()
            if (len(line) == 0):
                continue

            parts = line.split("\t")
            sentence = parts[4]
            num_corrections = int(parts[0])
            if num_corrections == 0:
                src_out.write("{}\n".format(sentence))
                tgt_out.write("{}\n".format(sentence))
            else:
                for i in range(num_corrections):
                    correction = parts[5+i]
                    src_out.write("{}\n".format(sentence))
                    tgt_out.write("{}\n".format(correction))

def main(opt):
    process(opt.src_dir, opt.out_dir, 'train')
    process(opt.src_dir, opt.out_dir, 'test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='transform-lang-8.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-src_dir', required=True, help="Path to corpus source files")
    parser.add_argument('-out_dir', required=True, help="Path for transformed data files")

    opt = parser.parse_args()
    main(opt)

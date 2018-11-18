#!/usr/bin/env python

import argparse
import os

def process(src_dir, out_dir, dataset):
    entry_filename = os.path.join(src_dir, 'entries.' + dataset)
    src_filename = os.path.join(out_dir, 'lang8-' + dataset + '.en')
    tgt_filename = os.path.join(out_dir, 'lang8-' + dataset + '.gec')
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

def split_file(file, out1, out2, percentage=0.5):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    with open(file, 'r',encoding="utf-8") as fin, \
        open(out1, 'w') as fout1, \
        open(out2, 'w') as fout2:
    
        nLines = sum(1 for line in fin)
        fin.seek(0)
    
        nTrain = int(nLines*percentage) 
        nValid = nLines - nTrain
    
        i = 0
        for line in fin:
            if (i < nTrain) or (nLines - i > nValid):
                fout1.write(line)
                i += 1
            else:
                fout2.write(line)

def main(opt):
    process(opt.src_dir, opt.out_dir, 'train')
    process(opt.src_dir, opt.out_dir, 'test')

    # rename test file to temp file so that test can be split into validation and test sets
    os.rename(os.path.join(opt.out_dir, 'lang8-' + 'test' + '.en'), os.path.join(opt.out_dir, 'lang8-' + 'temp' + '.en'))
    os.rename(os.path.join(opt.out_dir, 'lang8-' + 'test' + '.gec'), os.path.join(opt.out_dir, 'lang8-' + 'temp' + '.gec'))

    # split
    split_file(
        os.path.join(opt.out_dir, 'lang8-' + 'temp' + '.en'),
        os.path.join(opt.out_dir, 'lang8-' + 'valid' + '.en'),
        os.path.join(opt.out_dir, 'lang8-' + 'test' + '.en')
    )
    split_file(
        os.path.join(opt.out_dir, 'lang8-' + 'temp' + '.gec'),
        os.path.join(opt.out_dir, 'lang8-' + 'valid' + '.gec'),
        os.path.join(opt.out_dir, 'lang8-' + 'test' + '.gec')
    )

    # remove temp file
    os.remove(os.path.join(opt.out_dir, 'lang8-' + 'temp' + '.en'))
    os.remove(os.path.join(opt.out_dir, 'lang8-' + 'temp' + '.gec'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='transform-lang-8.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-src_dir', required=True, help="Path to corpus source files")
    parser.add_argument('-out_dir', required=True, help="Path for transformed data files")

    opt = parser.parse_args()
    main(opt)

"""
Evaluate the fluency of a trained language model.
"""

import torch

from fairseq import options
from fluency_scorer import FluencyScorer

def main(parsed_args):
    scorer = FluencyScorer(parsed_args.path, parsed_args.data)
    #scorer.score_dataset(parsed_args.gen_subset)

    lines = list()
    with open('{}/{}'.format(parsed_args.data, parsed_args.gen_subset)) as f:
        for line in f:
            lines.append(line.strip())
    scorer.score_lines(lines)

if __name__ == '__main__':
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

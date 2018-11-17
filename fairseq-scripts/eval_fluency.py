"""
Evaluate the fluency of a trained language model.
"""

import torch

from fairseq import options
from fluency_scorer import FluencyScorer

def main(parsed_args):
    print(parsed_args)
    scorer = FluencyScorer(parsed_args.path, parsed_args.data)
    scorer.run(parsed_args.gen_subset)

if __name__ == '__main__':
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

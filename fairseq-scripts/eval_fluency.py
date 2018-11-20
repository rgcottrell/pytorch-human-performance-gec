"""
Evaluate the fluency of a trained language model.
"""

import torch
import os

from fairseq import options
from fluency_scorer import FluencyScorer

def main(parsed_args):
    scorer = FluencyScorer(parsed_args.lang_model_path, parsed_args.lang_model_data)
    with open(os.path.join(parsed_args.data, parsed_args.gen_subset)) as f:
        for line in f:
            line = line.strip()
            score = scorer.score_sentence(line)
            print('[{:0.4f}] {}'.format(score, line))

if __name__ == '__main__':
    parser = options.get_eval_lm_parser()
    # fluency score arguments
    parser.add_argument('--lang-model-data', help='path to language model dictionary')
    parser.add_argument('--lang-model-path', help='path to language model file')
    args = options.parse_args_and_arch(parser)
    main(args)

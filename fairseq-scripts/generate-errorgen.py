#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Generate boosted training data from error generating model.
"""

import os
import torch

from fairseq import bleu, data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer

from fluency_scorer import FluencyScorer


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(
        args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(
        ':'), task, model_arg_overrides=eval(args.model_overrides))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        translator = SequenceScorer(models, task.target_dictionary)
    else:
        translator = SequenceGenerator(
            models, task.target_dictionary, beam_size=args.beam, minlen=args.min_len,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
            diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength,
        )

    if use_cuda:
        translator.cuda()

    # Initialize fluency scorer (and language model)
    fluency_scorer = FluencyScorer(
        args.lang_model_path, args.lang_model_data, use_cpu=False)

    en_filename = os.path.join(args.out_dir, 'errorgen.en')
    gec_filename = os.path.join(args.out_dir, 'errorgen.gec')
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t, open(en_filename, 'w') as en_file, open(gec_filename, 'w') as gec_file:
        if args.score_reference:
            translations = translator.score_batched_itr(
                t, cuda=use_cuda, timer=gen_timer)
        else:
            translations = translator.generate_batched_itr(
                t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
            )

        for sample_id, src_tokens, target_tokens, hypos in translations:
            # Process input and ground truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(
                    args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(
                    args.gen_subset).tgt.get_original_text(sample_id)
            else:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens, args.remove_bpe, escape_unk=True)

            # Only consider sentences with at least four words.
            if len(src_tokens) < 5:
                continue

            # Calculate the fluency score for the source sentence
            source_fluency = fluency_scorer.score_sentence(src_str)

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu(
                    ) if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                # Skip if this is the original sentence.
                if hypo_str == target_str:
                    continue

                # Score the hypothesis.
                hypo_fluency = fluency_scorer.score_sentence(hypo_str)

                # Save the hypothesis if it is sufficiently disfluent.
                if (source_fluency / hypo_fluency) > 1.05:
                    en_file.write('{}\n'.format(hypo_str))
                    gec_file.write('{}\n'.format(src_str))


if __name__ == '__main__':
    parser = options.get_generation_parser()
    # fluency score arguments
    parser.add_argument('--lang-model-data',
                        help='path to language model dictionary')
    parser.add_argument('--lang-model-path',
                        help='path to language model file')
    parser.add_argument('--out-dir', help='path to the disfluency corpus')

    args = options.parse_args_and_arch(parser)
    main(args)

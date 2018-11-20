#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import numpy as np
import sys

import torch

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator

from gleu import GLEU
from fluency_scorer import FluencyScorer

from flask import Flask, render_template, request
from gevent.pywsgi import WSGIServer

Batch = namedtuple('Batch', 'srcs tokens lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores gleu_scores fluency_scores alignments')


def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, task.source_dictionary, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=data.LanguagePairDataset(tokens, lengths, task.source_dictionary),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
        ), batch['id']


def main(args):
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, model_args = utils.load_ensemble_for_inference(model_paths, task, model_arg_overrides=eval(args.model_overrides))

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Initialize generator
    translator = SequenceGenerator(
        models, tgt_dict, beam_size=args.beam, minlen=args.min_len,
        stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen, unk_penalty=args.unkpen,
        sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
        diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength,
    )

    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Initialize fluency scorer (and language model)
    fluency_scorer = FluencyScorer(args.lang_model_path, args.lang_model_data)

    def make_result(src_str, hypos, tgt_str=''):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            pos_scores=[],
            gleu_scores=[],
            fluency_scores=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            # result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.hypos.append('H\t{}\t{}'.format(hypo_str, hypo['score']))
            result.pos_scores.append('P\t{}'.format(
                ' '.join(map(
                    lambda x: '{:.4f}'.format(x),
                    hypo['positional_scores'].tolist(),
                ))
            ))
            result.alignments.append(
                'A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment)))
                if args.print_alignment else None
            )

            # compute GLEU if target is provided
            if tgt_str:
                gleu_calculator = GLEU(args.n)
                gleu_calculator.load_text_sources([src_str])
                gleu_calculator.load_text_references([[tgt_str]])
                gleu_scores = gleu_calculator.run_iterations(num_iterations=args.iter,
                                                             hypothesis=[hypo_str],
                                                             per_sent=args.sent)
                gleu_score = [g for g in gleu_scores][0][0] * 100;
                result.gleu_scores.append('GLEU {:2.2f}'.format(gleu_score))
            else:
                result.gleu_scores.append('GLEU N/A (no target was provided. use format "source sentence|target setence" to provide a target/reference)')

            # compute fluency score
            fluency_scores = fluency_scorer.score_sentence(hypo_str)
            result.fluency_scores.append("Fluency Score: {:0.4f}".format(fluency_scores))

        return result

    def process_batch(batch, tgts):
        tokens = batch.tokens
        lengths = batch.lengths

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        encoder_input = {'src_tokens': tokens, 'src_lengths': lengths}
        translations = translator.generate(
            encoder_input,
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
        )

        return [make_result(batch.srcs[i], t, tgts[i]) for i, t in enumerate(translations)]

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if not args.server:
        listen_to_stdin(args, max_positions, process_batch, task)
    else:
        listen_to_web(args, max_positions, process_batch, task)


def listen_to_stdin(args, max_positions, process_batch, task):
    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    for inputs in buffered_read(args.buffer_size):
        sources = [line.split('|')[0] for line in inputs]
        targets = [line.split('|')[1] if len(line.split('|')) >= 2 else '' for line in inputs]
        indices = []
        results = []

        for batch, batch_indices in make_batches(sources, args, task, max_positions):
            indices.extend(batch_indices)
            results += process_batch(batch, targets)

        print_batch_results(indices, results)


def listen_to_web(args, max_positions, process_batch, task):
    # initialize web app
    app = Flask(__name__)

    # register route
    @app.route('/')
    def gec():
        input = request.args.get('input', '')
        inputs = [input]
        sources = [line.split('|')[0] for line in inputs]
        targets = [line.split('|')[1] if len(line.split('|')) >= 2 else '' for line in inputs]
        indices = []
        results = []

        for batch, batch_indices in make_batches(sources, args, task, max_positions):
            indices.extend(batch_indices)
            results += process_batch(batch, targets)

        outputs = print_batch_results(indices, results)

        return render_template('form.html', input=input, outputs=outputs)

    # listen with web server
    print('server running at port: {}'.format(args.port))
    http_server = WSGIServer(('', args.port), app)
    http_server.serve_forever()


def print_batch_results(indices, results):
    output = []
    outputs = [output]

    for i in np.argsort(indices):
        result = results[i]
        print(result.src_str)
        output.append(result.src_str)
        for hypo, pos_scores, gleu_scores, fluency_scores, align in zip(result.hypos, result.pos_scores,
                                                                        result.gleu_scores, result.fluency_scores,
                                                                        result.alignments):
            print(result.src_str)
            print(hypo)
            print(pos_scores)
            print(gleu_scores)
            print(fluency_scores)
            output.append(result.src_str)
            output.append(hypo)
            output.append(pos_scores)
            output.append(gleu_scores)
            output.append(fluency_scores)
            if align is not None:
                print(align)
                output.append(align)

    return outputs

if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    # GLEU arguments
    parser.add_argument('-n', default=4, type=int, help='n-gram order')
    parser.add_argument('--iter', default=500, help='number of GLEU iterations')
    parser.add_argument('--sent', default=True, action='store_true', help='sentence level scores')
    # fluency score arguments
    parser.add_argument('--lang-model-data', help='path to language model dictionary')
    parser.add_argument('--lang-model-path', help='path to language model file')
    # server arguments
    parser.add_argument('--server', default=False, action='store_true', help='listen with built-in web server')
    parser.add_argument('--port', default=5000, type=int, help='port to web interface')
    args = options.parse_args_and_arch(parser)
    # main logic and events
    main(args)

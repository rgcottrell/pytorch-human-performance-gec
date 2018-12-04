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

from flask import Flask, render_template, request, send_from_directory
from flask_restful import Resource, Api, fields, marshal_with
from gevent.pywsgi import WSGIServer
from operator import attrgetter

Batch = namedtuple('Batch', 'srcs tokens lengths')
# Translation = namedtuple('Translation', 'src_str hypos pos_scores gleu_scores fluency_scores alignments')

class Correction(object):
    iteration = 0
    src_str = hypo_str = ''
    hypo_score = pos_scores = gleu_scores = fluency_scores = alignments = 0
    hypo_score_str = pos_scores_str = gleu_scores_str = fluency_scores_str = alignments_str = ''


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

    def make_result(src_str, hypos, tgt_str='', iteration=0):
        results = []

        # compute fluency score for source string
        # the source string itself is an entry
        result0 = Correction()
        result0.iteration = iteration
        result0.src_str = result0.hypo_str = src_str
        fluency_scores = fluency_scorer.score_sentence(src_str).item()
        result0.fluency_scores = fluency_scores
        result0.fluency_scores_str = "Fluency Score: {:0.4f}".format(fluency_scores)
        results.append(result0)

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            result = Correction()
            result.iteration = iteration + 1
            result.src_str = src_str

            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            # result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.hypo_str = hypo_str
            result.hypo_score = result.hypo_score_str = hypo['score']
            result.pos_scores_str = 'P\t{}'.format(
                ' '.join(map(
                    lambda x: '{:.4f}'.format(x),
                    hypo['positional_scores'].tolist(),
                ))
            )
            result.alignments_str = (
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
                result.gleu_scores = gleu_score
                result.gleu_scores_str = 'GLEU {:2.2f}'.format(gleu_score)
            else:
                result.gleu_scores_str = 'GLEU N/A (no target was provided. use format "source sentence|target setence" to provide a target/reference)'

            # compute fluency score
            fluency_scores = fluency_scorer.score_sentence(hypo_str).item()
            result.fluency_scores = fluency_scores
            result.fluency_scores_str = "Fluency Score: {:0.4f}".format(fluency_scores)

            results.append(result)

        return results

    def process_batch(batch, tgts, iteration):
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

        return [make_result(batch.srcs[i], t, tgts[i], iteration) for i, t in enumerate(translations)]

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if not args.server:
        listen_to_stdin(args, max_positions, task, process_batch)
    else:
        listen_to_web(args, max_positions, task, process_batch)


def process_inputs(args, inputs, max_positions, task, process_batch):
    sources = [line.split('|')[0] for line in inputs]
    targets = [line.split('|')[1] if len(line.split('|')) >= 2 else '' for line in inputs]
    indices = []
    results = []
    outputs = []

    iteration = 0
    best_fluency_score = 0
    best_hypo_str = ''
    # Boost inference
    while True:
        for batch, batch_indices in make_batches(sources, args, task, max_positions):
            indices.extend(batch_indices)
            result = process_batch(batch, targets, iteration)
            results += result

        output = print_batch_results(indices, results)
        outputs += output

        if iteration == 0:
            best_fluency_score = results[0][0].fluency_scores
            best_hypo_str = sources[0]

        max_correction = max(results[iteration], key=attrgetter('fluency_scores'))
        if max_correction.fluency_scores <= best_fluency_score:
            break
        else:
            iteration = iteration + 1
            best_fluency_score = max_correction.fluency_scores
            best_hypo_str = max_correction.hypo_str

            sources = [best_hypo_str]
            print('Boost inference from \t"{}"\t({:2.2f})'.format(best_hypo_str, best_fluency_score))
            outputs[-1].append('boost inference from \t"{}"\t({:2.2f})'.format(best_hypo_str, best_fluency_score))

    print('Best inference\t"{}"\t({:2.2f})'.format(best_hypo_str, best_fluency_score))
    outputs[-1].append('Best inference H\t"{}"\t({:2.2f})'.format(best_hypo_str, best_fluency_score))

    return results, outputs


def print_batch_results(indices, results):
    output = []
    outputs = [output]

    for i in np.argsort(indices):
        for result in results[i]:
            print('Iteration\t{}'.format(result.iteration))
            print('O\t{}'.format(result.src_str))
            print('H\t{}\t{}'.format(result.hypo_str, result.hypo_score))
            # print('H\t{}'.format(result.hypo_str))
            # print(result.pos_scores_str)
            print(result.fluency_scores_str)
            output.append('Iteration\t{}'.format(result.iteration))
            output.append('O\t{}'.format(result.src_str))
            output.append('H\t{}\t{}'.format(result.hypo_str, result.hypo_score))
            # output.append('H\t{}'.format(result.hypo_str))
            # output.append(result.pos_scores_str)
            output.append(result.fluency_scores_str)
            if result.gleu_scores:
                print(result.gleu_scores_str)
                output.append(result.gleu_scores_str)
            if result.alignments_str is not None:
                print(result.alignments_str)
                output.append(result.alignments_str)

    return outputs


def listen_to_stdin(args, max_positions, task, process_batch):
    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    for inputs in buffered_read(args.buffer_size):
        process_inputs(args, inputs, max_positions, task, process_batch)


def listen_to_web(args, max_positions, task, process_batch):
    # initialize web app
    app = Flask(__name__, static_folder='')
    api = Api(app)

    # register route for web server

    # a simple form page
    @app.route('/form')
    def form():
        input = request.args.get('input', '')
        inputs = [input]
        results, outputs = process_inputs(args, inputs, max_positions, task, process_batch)
        return render_template('form.html', input=input, outputs=outputs)

    # a dynamic web app with static resource
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/static/<path:path>')
    def send_static(path):
        return send_from_directory('templates/static', path)

    # a JSON api
    resource_fields = {
        'iteration': fields.Integer,
        'src_str': fields.String, 'hypo_str': fields.String,
        'hypo_score': fields.Float, 'pos_scores': fields.Float, 'gleu_scores': fields.Float,
        'fluency_scores': fields.Float, 'alignments': fields.Float,
        'hypo_score_str': fields.String, 'pos_scores_str': fields.String, 'gleu_scores_str': fields.String,
        'fluency_scores_str': fields.String,  'alignments_str': fields.String
    }
    class API(Resource):
        @marshal_with(resource_fields)
        def get(self, input):
            inputs = [input]
            results, outputs = process_inputs(args, inputs, max_positions, task, process_batch)
            # return outputs # raw string outputs
            return results # json

    # register routes for API
    api.add_resource(API, '/api/<string:input>')

    # listen with web server
    print('server running at port: {}'.format(args.port))
    http_server = WSGIServer(('', args.port), app)
    http_server.serve_forever()


if __name__ == '__main__':
    # BLEU arguments
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

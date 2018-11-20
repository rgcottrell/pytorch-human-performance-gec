"""
Evaluate the fluency of a trained language model.
"""

import torch
import numpy as np

from fairseq import data, tasks, tokenizer, utils
from fairseq.sequence_scorer import SequenceScorer

class FluencyArgs(dict):
    """Encapsulate args to build FluencyScorer."""

    def __init__(self, path, data):
        self['path'] = path
        self['data'] = data
        # Set options to allow line-separate, raw text data files.
        self['task'] = 'language_modeling'
        self['raw_text'] = True
        self['sample_break_mode'] = 'eos'
        # Default fairseq option values.
        self['output_dictionary_size'] = -1
        self['self_target'] = False
        self['future_target'] = False
        self['past_target'] = False
        self['num_shards'] = 1
        self['shard_id'] = 0
        self['max_tokens'] = None
        self['max_sentences'] = None
        self['tokens_per_sample'] = 1024

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

class FluencyScorer(object):
    """Evaluate sentences for fluency.

    The FluencyScorer class uses an embedded language model to score candidate
    sentences for according to how likely they would be used by a native
    speaker.
    """

    def __init__(self, path, data, use_cpu=True):
        # Create the language modeling task.
        self.args = FluencyArgs(path, data)
        self.task = tasks.setup_task(self.args)
        self.use_cuda = torch.cuda.is_available and not use_cpu

        # Load language model ensemble.
        models, model_args = utils.load_ensemble_for_inference(self.args.path.split(':'), self.task)
        self.models = models
        self.model_args = model_args

        # Optimize ensemble for generation.
        for model in self.models:
            model.make_generation_fast_()
            if self.use_cuda and self.model_args.fp16:
                model.half()
        
        # Create the sequence scorer.
        self.scorer = SequenceScorer(self.models, self.task.target_dictionary)
        if self.use_cuda:
            self.scorer.cuda()
    
    def score_sentence(self, line):
        # Tokenize the input sentence into a batch of size one.
        tokens = tokenizer.Tokenizer.tokenize(line, self.task.dictionary, add_if_not_exist=False).long()
        lengths = np.array([tokens.numel()])
        ds = data.TokenBlockDataset(tokens, lengths, self.args.tokens_per_sample, pad=self.task.dictionary.pad(), eos=self.task.dictionary.eos(), break_mode=self.args.sample_break_mode, include_targets=True)

        # Create a batch iterator to wrap the data.
        add_eos_for_other_targets = self.args.sample_break_mode is not None and self.args.sample_break_mode != 'none'
        itr = self.task.get_batch_iterator(
            dataset=data.MonolingualDataset(ds, ds.sizes, self.task.dictionary, self.task.target_dictionary, add_eos_for_other_targets=add_eos_for_other_targets, shuffle=False, targets=self.task.targets),
            max_tokens=self.args.max_tokens or 3000,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in self.models 
            ]),
            num_shards=self.args.num_shards,
            shard_id=self.args.shard_id,
            ignore_invalid_inputs=True,
        ).next_epoch_itr(shuffle=False)
        
        # Evaluate the sentence and return the fluency score.
        results = self.scorer.score_batched_itr(itr, cuda=self.use_cuda)
        for _, _, _, hypos in results:
            for hypo in hypos:
                # Ignore words with infinite probability. This can happen when
                # running low-precision inference on the GPU. 
                pos_scores = hypo['positional_scores']
                word_prob = [score for score in pos_scores if score != float('-inf') and score != float('inf')]
                return self._fluency_score(word_prob)
        return 0.0

    def _fluency_score(self, word_prob):
        """Calculate fluency score.

        Given the list of log-probabilities for each token, calculate the
        fluency score of the sentence.
        """

        # If there were no tokens because they were all filtered out for
        # having infinite probabilites, then give a minimum fluency score.
        if len(word_prob) == 0:
            return 0.0

        H = 0.0
        for x in word_prob:
            H -= x
        H = H / len(word_prob)
        score = 1.0 / (1.0 + H)
        return score

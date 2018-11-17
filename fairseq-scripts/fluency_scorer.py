"""
Evaluate the fluency of a trained language model.
"""

import torch

from fairseq import data, tasks, utils
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
    
    def run(self, subset):
        # Load the dataset.
        self.task.load_dataset(subset)

        # Create batch iterator.
        itr = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.args.max_tokens or 3600,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in self.models 
            ]),
            num_shards=self.args.num_shards,
            shard_id=self.args.shard_id,
            ignore_invalid_inputs=True,
        ).next_epoch_itr(shuffle=False)

        results = self.scorer.score_batched_itr(itr, cuda=self.use_cuda)
        for _, _, _, hypos in results:
            for hypo in hypos:
                pos_scores = hypo['positional_scores']

                words = []
                word_prob = []
                for i in range(len(hypo['tokens'])):
                    w_ind = hypo['tokens'][i].item()
                    w = self.task.dictionary[w_ind]
                    words.append(w)
                    word_prob.append(pos_scores[i].item())
                
                score = self._fluency_score(word_prob)
                print("[{:0.4f}] {}".format(score, ' '.join(words)))

    def _fluency_score(self, word_prob):
        """Calculate fluency score.

        Given the list of log-probabilities for each token, calculate the
        fluency score of the sentence.
        """

        H = 0.0
        for x in word_prob:
            # Ignore words with infinite probability. This can happen when
            # running low-precision inference on the GPU. 
            #
            # FIXME: What is the right thing to do here?
            if x != float('-inf') and x != float('inf'):
                H -= x
        H = H / len(word_prob)
        score = 1.0 / (1.0 + H)
        return score

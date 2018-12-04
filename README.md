# pytorch-human-performance-gec

A PyTorch implementation of "Reaching Human-level Performance in Automatic Grammatical Error Correction: An Empirical Study"

## Initialize Submodules

After checking out the repository, be sure to initialize the included git submodules:

```sh
git submodule update --init --recursive
```

## Install Required Dependencies

This project requires the use of `PyTorch`, which can be installed by following the directions on its [project page](https://pytorch.org)

This project also uses the `fairseq` NLP library, which is included as a submodule in this repository. To prepare the library for use, make sure that it is installed along with its dependencies.

```sh
cd fairseq
pip install -r requirements.txt
python setup.py build develop
```

## fairseq Scripts

All fairseq scripts have been grouped under `fairseq-scripts` folder.

### Preparing Data

The first step is to prepare the source and target pairs of training and validation data. Extract original `lang-8-en-1.0.zip` under `corpus` folder. Then create another folder `lang-8-fairseq` under `corpus` folder to store re-formatted corpus.

To split the Lang-8 learner data training set, use the following command:

```sh
python transform-lang8.py -src_dir <dataset-src> -out_dir <corpus-dir>
```
e.g.
```sh
python transform-lang8.py -src_dir ../corpus/lang-8-en-1.0 -out_dir ../corpus/lang-8-fairseq
```

Once the data has been extracted from the dataset, use fairseq to prepare the training and validation data and create the vocabulary:

```sh
preprocess-lang8.bat
```

### Train the Model

To train the error-correcting model, run the following command:

```sh
train-lang8-cnn.bat
```

Note that this script may need to be adjusted based on the GPU and memory resources available for training.

### Testing the Model

To test the model, run the following command to try to correct a test list of sentences:

```sh
generate-lang8-cnn.bat
```

### Evaluate the model

Evaluate scripts are used to score model using text or pre-processed files in batch.

Evaluate can be done against lang-8 test dataset using

```sh
generate-lang8-cnn-rawtext.bat
```

The paper evaluates against JFLEG test dataset, which can be done using

```sh
generate-jfleg-cnn-rawtext.bat
```

Above scripts can be modified to test other test dataset easily as they use plain text.

Other scripts such as `generate-lang8.bat or generate-lang8-cnn.bat` can only deal with pre-processed data so it is less convenient.

### Interactive scripts

While evaluate scripts are good at batch processing, two interactive scripts are provided to see details of generation / correction.

Below script will run in console mode:
```sh
interactive-lang8-cnn-console.bat
```

Below script will boot a local server to provide a web GUI and RESTful API interface:
```sh
interactive-lang8-cnn-web.bat
```

### Patching fairseq Environment

If error `AttributeError: function 'bleu_zero_init' not found` occurs on Windows, modify functions to have `__declspec(dllexport)` then build again. See [Issue 292](https://github.com/pytorch/fairseq/issues/292)

If error `UnicodeDecodeError: 'charmap' codec can't decode byte` error occurs, modify `fairseq/tokenizer.py` to include `, encoding='utf8'` for all `open` functions.

When trying built-in example from `fairseq/examples/translation/prepare-[dataset].sh`, scripts may need to change .py path from `$BPEROOT/[script].py` to `$BPEROOT/subword_nmt/[script].py`.



## OpenNMT Scripts (Legacy)

All OpenNMT scripts have been grouped under `opennmt-scripts` folder.

### Preparing Data

The first step is to prepare the source and target pairs of training and validation data. Extract original `lang-8-en-1.0.zip` under `corpus` folder. Then create another folder `lang-8-opennmt` under `corpus` folder to store re-formatted corpus.

To split the Lang-8 learner data training set, use the following command:

```sh
python transform-lang8.py -src_dir <dataset-src> -out_dir <corpus-dir>
```
e.g.
```sh
python transform-lang8.py -src_dir ../corpus/lang-8-en-1.0 -out_dir ../corpus/lang-8-opennmt
```

Once the data has been extracted from the dataset, use OpenNMT to prepare the training and validation data and create the vocabulary:

```sh
preprocess-lang8.bat
```

### Train the Model

To train the error-correcting model, run the following command:

```sh
train.bat
```

Note that this script may need to be adjusted based on the GPU and memory resources available for training.

### Testing the Model

To test the model, run the following command to try to correct a test list of sentences:

```sh
translate.bat
```

After the sentences have been translated, the source and target sentence may be compared side to side using the following command:

```sh
python compare.py
```

### Patching OpenNMT-py Environment

If `preprocess.py` fails with a TypeError, then you may need to patch OpenNMT-py.

Update `OpenNMT-py\onmt\inputters\dataset_base.py` with the following code:

```python
def __reduce_ex__(self, proto):
    "This is a hack. Something is broken with torch pickle."
    return super(DatasetBase, self).__reduce_ex__(proto)
```

If `TypeError: __init__() got an unexpected keyword argument 'dtype'` occurs, `pytorch/text` installed by pip may be out of date. Update it using `pip install git+https://github.com/pytorch/text`

If `RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS` occurs during training, try install pytorch with CUDA 9.2 using conda instead of using default CUDA 9.0.


# pytorch-human-performance-gec

The goal of this project is to implement a grammatical error correction model from paper ["Reaching Human-level Performance in Automatic Grammatical Error Correction: An Empirical Study"](https://arxiv.org/abs/1807.01270) using [PyTorch](https://github.com/pytorch/pytorch) and [fairseq](https://github.com/pytorch/fairseq).

While the original paper achieves human performance, this implementation is more about an empirical study on applying deep learning in NLP as a university team project.

## Project Team

This project was completed as the final project for CS 410: Text Information Systems at the University of Illinois at Urbana-Champaign. The team members and their primary areas of responsibility were:

* Tianfei Chen: left-to-right model construction and training, GLEU scoring function, evaluation of test sets, inference, boost inference, web GUI, documentation, and project presentation.
* Robert Cottrell: framework evaluation and selection, organizing project structure and dependencies, initial prototyping and documentation, boost learning, fluency scoring, and project presentation.
* Lenny Giannachi: research datasets, acquire alternative datasets, and technology review.
* Jingwei Li: BPE tokenization, research alignment dictionary generation.
* Jonathan Montwell: research datasets, right-to-left data generation, technology review.

A [video presentation](https://www.youtube.com/watch?v=n_BHr5RyVdA) is available online and the (slides)[Project-Presentation.pdf] are included in this repository.

## What We Learned

### Empirical Study

- Write code quickly
  - Use a framework
    - Don't start from scratch. Use existing library, framework and toolkit.
    - Pre-processing, batching, checkpointing, common models are all included.
    - Tried both OpenNMT and fairseq NLP toolkits.
    - Settled with fairseq which is also what paper used.
  - It is ok to just copy toolkit and components then modify them
    - This is officially recommended way to extending and playing with fairseq.
    - When you copy many times you will naturally learn how to refactor them later.
- Run experiments
  - Run out of box samples to learn how toolkit works.
  - Run simple models to get a proof of concept and confidence.
- Keep track of what you tried
  - Keep track of what you ran
    - Source control constructed models and hyper-paramters.
    - Save the scripts so that you can repeat them again with other tweaks and datasets.
  - Keep track of what you got
    - Many frameworks and toolkits will do this for you automatically.
    - Just need to keep folders clear and watch your hard drive free space.
- Analyze model behavior
  - Feed model with some simplest tests
    - Look at generated values see if that's what you expected.
    - Look at scores and see if they make sense.
  - Evaluate models with proper evaluation metric
    - Loss function for training.
    - Fluency score (cross entropy) and GLEU score etc.

### Completed

- A left-to-right 7-layer convolutional seq2seq model has been implemented using same architecture as the paper suggested.
- The base convolutional seq2seq model has been trained using mostly same hyper parameters.
- Fluency score function, which is used for both boost training / learning and boost inference, has been implemented. For example, nature sentences get higher score.
```
[0.1977] It is a truth universally acknowledged , that a single man in possession of a good fortune must be in want of a wife . </s>
[0.1937] I am going to a party tomorrow . </s>
[0.1902] I am going to the party tomorrow . </s>
[0.1864] It was the best of times , it was the worst of times , it was the age of wisdom , it was the age of foolishness , it was the epoch of belief , it was the epoch of incredulity , it was the season of Light , it was the season of Darkness , it was the spring of hope , it was the winter of despair , we had everything before us , we had nothing before us , we were all going direct to Heaven , we were all going direct the other way - in short , the period was so far like the present period , that some of its noisiest authorities insisted on its being received , for good or for evil , in the superlative degree of comparison only . </s>
[0.1654] Yesterday I saw a car . </s>
[0.1630] Tomorrow I am going to a party . </s>
[0.1540] I saw the car yesterday . </s>
[0.1473] Tomorrow I am going to party . </s>
[0.1383] Tomorrow I go to party . </s>
[0.1296] Yesterday I see car . </s>
[0.1280] Yesterday I saw car . </s>
```
- GLEU score function, which is used to evaluate JFLEG test set, has been implemented. For example, similar sentences have higher GLEU score.
```
There are several reason|There are several reasons
O       There are several reason
H       There are several reasons       -0.029993820935487747
P       -0.0028 -0.0339 -0.0015 -0.0464 -0.0653
GLEU 100.00
For not use car|Not for use with a car
O       For not use car
H       For not use car -0.06429481506347656
P       -0.1332 -0.0239 -0.1537 -0.0102 -0.0006
GLEU 27.40
Every knowledge is connected each other|All knowledge is connected
O       Every knowledge is connected each other
H       Every knowledge is connected to each other      -0.17184138298034668
P       -0.1573 -0.0054 -0.0348 -0.0004 -0.9934 -0.0301 -0.0026 -0.1507
GLEU 18.58
```
- An error generation model has been implemented to generates more synthetic data from original training dataset, which will in turn boost training of the base model. For example.
```
S-3654  How many languages have you studied ?
H-3654  How many language have you studied ?    -0.19821381568908691
H-3654  How many languages do you study ?       -0.5254995822906494
H-3654  How much languages have you studied ?   -0.5455195903778076
H-3654  How many languages are you studied ?    -0.5917201042175293
```
- Basic inference with the base model has been implemented. For example, entered sentence is corrected in multiple ways.
```
In the world oil price very high right now .

Iteration	0
O	In the world oil price very high right now .
H	In the world oil price very high right now .	0
Fluency Score: 0.1503

Iteration	1
O	In the world oil price very high right now .
H	In the world oil prices very high right now .	-0.2768438458442688
Fluency Score: 0.1539
Iteration	1
O	In the world oil price very high right now .
H	In the world oil prices are very high right now .	-0.31139659881591797
Fluency Score: 0.1831
Iteration	1
O	In the world oil price very high right now .
H	In the world oil price is very high right now .	-0.3594667315483093
Fluency Score: 0.1731
Iteration	1
O	In the world oil price very high right now .
H	In the world oil price very expensive right now .	-0.4148099422454834
Fluency Score: 0.1434

Best inference 	"In the world oil prices are very high right now ."	(0.1831)
```
- Boost inference has been implemented to use both base model and language model. For example, entered sentence is corrected in multiple ways, the best scored one is chosen for multiple rounds of correction, until the score cannot be improved.
```
In the world oil price very high right now .

Iteration	0
O	In the world oil price very high right now .
H	In the world oil price very high right now .	0
Fluency Score: 0.1503

Iteration	1
O	In the world oil price very high right now .
H	In the world oil prices very high right now .	-0.2768438458442688
Fluency Score: 0.1539
Iteration	1
O	In the world oil price very high right now .
H	In the world oil prices are very high right now .	-0.31139659881591797
Fluency Score: 0.1831
Iteration	1
O	In the world oil price very high right now .
H	In the world oil price is very high right now .	-0.3594667315483093
Fluency Score: 0.1731
Iteration	1
O	In the world oil price very high right now .
H	In the world oil price very expensive right now .	-0.4148099422454834
Fluency Score: 0.1434

Boost inference from 	"In the world oil prices are very high right now ."	(0.1831)

Iteration	2
O	In the world oil prices are very high right now .
H	In the world oil prices are very expensive right now .	-0.3672739863395691
Fluency Score: 0.1690
Iteration	2
O	In the world oil prices are very high right now .
H	In the world oil prices are very high now .	-0.4246770739555359
Fluency Score: 0.1883
Iteration	2
O	In the world oil prices are very high right now .
H	The world oil prices are very high right now .	-0.42579686641693115
Fluency Score: 0.1770
Iteration	2
O	In the world oil prices are very high right now .
H	In the world oil prices are very high right now ,	-0.6304754018783569
Fluency Score: 0.1748

Boost inference from 	"In the world oil prices are very high now ."	(0.1883)

Iteration	3
O	In the world oil prices are very high now .
H	In the world oil prices are very expensive now .	-0.41596412658691406
Fluency Score: 0.1693
Iteration	3
O	In the world oil prices are very high now .
H	The world oil prices are very high now .	-0.45905303955078125
Fluency Score: 0.1780
Iteration	3
O	In the world oil prices are very high now .
H	In world oil prices are very high now .	-0.47978001832962036
Fluency Score: 0.1718
Iteration	3
O	In the world oil prices are very high now .
H	In the world oil prices are very high now ,	-0.6376678347587585
Fluency Score: 0.1780

Best inference	"In the world oil prices are very high now ."	(0.1883)
```
- Evaluation of JFLEG test set using GLEU score.
  - The base model has a GLEU score 48.17 on JFLEG test set when it was trained for 2 epochs.
  - The base model has a GLEU score 48.89 on JFLEG test set when it was trained for 3 epochs.
  - The introduction of boost inference increased GLEU from 48.89 to 49.39. The percentage of increase is consistent with the paper ( ≈ 1% ).
- An enhanced interactive mode with RESTful API and Web GUI.
  - RESTful API
  - ![RESTful API](raw/restful-api.png?raw=true "RESTful API")
  - Web GUI
  - ![Web GUI](raw/web-gui.png?raw=true "Web GUI")
  - Web GUI 2
  - ![Web GUI 2](raw/web-gui-2.png?raw=true "Web GUI 2")

### Not Completed

- A right-to-left convolutional seq2seq model.
- Training of the model using new dataset generated by boost learning.
- BPE tokenization and unknown token replacement.
- Stemming for raw text / interactive modes.
- Evaluation of CoNLL-10 test set using F0.5.

### Barriers

- No enough training dataset: we got only Lang-8 corpus, which is about 20% of the dataset used by the paper. We've tried to contact other organizations, but no response was received.
- Limit time: training big neural network model with large dictionaries is time consuming. We decided to complete the whole process instead of trying to reproduce the score achieved by the paper. We can probably achieve it given more time and datasets.

## Initialize Submodules

After checking out the repository, be sure to initialize the included git submodules:

```sh
git submodule update --init --recursive
```

The reasons of using them as submodules rather than Python package are:
* some scripts and functions might need be patched in order to work properly.
* a few scripts are modified based on the original scripts, which is the officially recommended way of using fairseq.

## Install Required Dependencies

The environment used for the development is Windows 10 64bit + Python 3.6 + CUDA 9.2 + pytorch 0.4.1.

`PyTorch` can be installed by following the directions on its [project page](https://pytorch.org). Conda is recommended as it will install CUDA dependencies automatically. For example,

```sh
conda install pytorch cuda92 -c pytorch
pip3 install torchvision
```

This project also uses the `fairseq` NLP toolkit, which is included as a submodule in this repository. To prepare the library for use, make sure that it is installed along with its dependencies.

```sh
cd fairseq
pip install -r requirements.txt
python setup.py build develop
```

Other project dependencies are placed under `fairseq-scripts` folder, which can be installed by running

```sh
cd fairseq-scripts
pip install -r requirements.txt
```

## Folder Structures

```
.
|-- OpenNMT-py                  The other NLP toolkit we tried early (legacy)
|-- BPEtokenization-scripts     Workbook demonstrating the use of BPE tokenization
|-- checkpoints                 Trained models and checkpoints
|   |-- errorgen-fairseq-cnn        An error generation model that takes corrected sentences as input,
                                    uncorrected sentences as output
|   |-- lang-8-fairseq              A simple single layer LSTM model for error correction
|   `-- lang-8-fairseq-cnn          A 7-layer convolutional seq2seq model for error correction
|-- corpus                      Raw and prepared corpus
|   |-- errorgen-fairseq            Corpus generated by the error generation model - the result of boost learning.
|   |-- lang-8-en-1.0               Raw Lang-8 corpus
|   |-- lang-8-fairseq              Corpus format required by fairseq
|   `-- lang-8-opennmt              Corpus format required by OpenNMT
|-- data-bin                    Pre-processed and binarized data
|   |-- errorgen-fairseq            Binarized synthetic data and dictionaries
|   |-- lang-8-fairseq              Binarized Lang-8 data and dictionaries
|   `-- wiki103                     Pre-trained WikiText-103 language model and dictionaries
|-- fairseq                     fairseq submodule
|-- fairseq-scripts             fairseq scripts used to implement the model and process proposed by the paper
|-- opennmt                     OpenNMT data and model folder (legacy)
|-- opennmt-scripts             OpenNMT scripts folder (legacy)
`-- test                        Random test text files can be thrown to here
```

## fairseq Custom Scripts / Software Usage Tutorial

All fairseq scripts have been grouped under `fairseq-scripts` folder. The whole process is:

1. Preparing data
2. Pre-process data
3. Train the model
4. Testing the model
5. Evaluate the model
6. Interactive mode
7. Boosting

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

### Pre-process Data

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

To test the model, run the following command to try to correct a list of sentences:

```sh
generate-lang8-cnn.bat
```

This command will try to correct all sentences in a file with probabilities and scores in the output. It is a convenient way to peed that the model behaves as expected against lots of test data.

### Evaluate the Model

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

### Interactive Mode

While evaluate scripts are good at batch processing, two interactive scripts are provided to see details of generation / correction.

Below script will run in console mode:
```sh
interactive-lang8-cnn-console.bat
```

Below script will boot a local server to provide a web GUI and RESTful API interface:
```sh
interactive-lang8-cnn-web.bat
```

Interactive mode allows users to enter a sentence in console, or Web GUI, to see how subtle difference in input are corrected.

### Boosting

To augment training data to provide more examples of common errors, this project builds an error-generating model that can produce additional lower quality sentences for correct sentences. This uses the same training data as the regular model, but reverses the source and target sentences.

Once the data has been extracted from the dataset, use fairseq to prepare the training and validation data and create the vocabulary:

```sh
preprocess-errorgen.bat
```

To train the error-correcting model, run the following command:

```sh
train-errorgen-cnn.bat
```

Note that this script may need to be adjusted based on the GPU and memory resources available for training.

Now the error-generating model can be use to generate additional training data. The generating script will only consider sentences longer than four words that are at least 5% less fluent (as measured by the fluency scorer) than the corrected sentences. This ensures that the new sentences are more likely to showcase interesting corrections while avoiding trivial edits. Notice that in this case we use the latest model checkpoint rather than the most generalized, because in this particular case overfitting to the training data is an advantage!

```sh
generate-errorgen-cnn.bat
```

The sentences generated in the corpus\errorgen directory can then be used as additional data to train or fine tune the error-correcting model.

### Additional Techniques

In addition to the work describe above, additional datasets and techniques for data preprocessing, model training, and other imporovements were evaluated.

BPE tokenization promises to make more effective use of a limited number of vocabulary tokens by further subdividing words into subword tokens that can be shared by many different words. A Jupyter [notebook](BPEtokenization-scripts/SubwordNMT.ipynb) showing how to install and tokenize the dataset is available in the [BPEtokenization-scripts](BPEtokenization-scripts/) directory.

An example of sentences after apply BPE tokenization can be seen below:
```
I will introduce my dog , Ti@@ ara .
She is a cheerful and plu@@ mp pretty dog , perhaps she is the cu@@ test dog in the world .
She 's an 8 year old golden re@@ tri@@ ever
Her fu@@ r is a beautiful a@@ mber colour and is soft .
She is a little stupid , but has per@@ cep@@ tive insi@@ ghts about food because she is always fami@@ shed ^ ^ She loves food !
When she has had her food , she always pr@@ ances around the living room mer@@ ri@@ ly .
And she loves ba@@ s@@ king too .
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

To test the model, run the following command to try to correct a list of sentences:

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


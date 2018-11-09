python ..\fairseq\train.py^
    ..\data-bin\lang-8-fairseq^
    --save-dir ..\checkpoints\lang-8-fairseq-cnn^
    --arch fconv^
    --encoder-embed-dim 500^
    --decoder-embed-dim 500^
    --decoder-out-embed-dim 500^
    --encoder-layers "[(1024, 3)] * 7"^
    --decoder-layers "[(1024, 3)] * 7"^
    --optimizer nag^
    --momentum 0.99^
    --lr 0.25^
    --dropout 0.2^
    --max-tokens 1000^
    --max-sentences 12^
    --clip-norm 0.1^
    --fp16

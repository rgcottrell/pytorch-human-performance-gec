python ..\fairseq\train.py^
    ..\data-bin\errorgen-fairseq^
    --save-dir ..\checkpoints\errorgen-fairseq-cnn^
    --arch fconv^
    --encoder-embed-dim 500^
    --decoder-embed-dim 500^
    --decoder-out-embed-dim 500^
    --encoder-layers "[(1024, 3)] * 7"^
    --decoder-layers "[(1024, 3)] * 7"^
    --optimizer nag^
    --momentum 0.99^
    --lr 0.25^
    --lr-scheduler reduce_lr_on_plateau^
    --lr-shrink 0.1^
    --min-lr 0.0001^
    --dropout 0.2^
    --max-tokens 1000^
    --clip-norm 0.1^
    --fp16

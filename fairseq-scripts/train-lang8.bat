python ..\fairseq\train.py^
    ..\data-bin\lang-8-fairseq^
    --save-dir ..\checkpoints\lang-8-fairseq
    --lr 0.25^
    --clip-norm 0.1^
    --dropout 0.2^
    --max-tokens 2000^
    --max-sentences 32^
    --arch lstm^
    --fp16^

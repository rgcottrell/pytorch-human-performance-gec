python ..\fairseq\generate.py^
    ..\data-bin\lang-8-fairseq^
    --path ..\checkpoints\lang-8-fairseq-cnn\checkpoint_best.pt^
    --batch-size 128^
    --beam 5
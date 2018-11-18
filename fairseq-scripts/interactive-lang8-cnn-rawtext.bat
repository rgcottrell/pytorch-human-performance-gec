:: copy lang-8's dictionary to test folder
copy ..\data-bin\lang-8-fairseq\dict.en.txt ..\test\lang-8\
copy ..\data-bin\lang-8-fairseq\dict.gec.txt ..\test\lang-8\

python .\interactive.py^
    ..\test\lang-8^
    --path ..\checkpoints\lang-8-fairseq-cnn\checkpoint_best.pt^
    --beam 5^
    --nbest 1^
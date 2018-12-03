python ..\fairseq\preprocess.py^
    --source-lang gec^
    --target-lang en^
    --trainpref ..\corpus\lang-8-fairseq\lang8-train^
    --validpref ..\corpus\lang-8-fairseq\lang8-valid^
    --testpref ..\corpus\lang-8-fairseq\lang8-test^
    --destdir ..\data-bin\errorgen-fairseq^
    --workers 4
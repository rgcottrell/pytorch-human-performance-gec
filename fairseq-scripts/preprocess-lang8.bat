python ..\fairseq\preprocess.py^
    --source-lang en^
    --target-lang gec^
    --trainpref ..\corpus\lang-8-fairseq\lang8-train^
    --validpref ..\corpus\lang-8-fairseq\lang8-valid^
    --testpref ..\corpus\lang-8-fairseq\lang8-test^
    --destdir ..\data-bin\lang-8-fairseq^
    --workers 4
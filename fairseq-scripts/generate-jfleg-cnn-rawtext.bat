:: rename lang-8's test files in test folder
:: original JFLEG source file become fairseq's source
copy ..\test\jfleg\test.src ..\test\jfleg\test.en-gec.en
:: original JFLEG (human corrected) reference file become fairseq's target
copy ..\test\jfleg\test.ref0 ..\test\jfleg\test.en-gec.gec

:: copy lang-8's dictionary to test folder
copy ..\data-bin\lang-8-fairseq\dict.en.txt ..\test\jfleg\
copy ..\data-bin\lang-8-fairseq\dict.gec.txt ..\test\jfleg\

python .\generate.py^
    ..\test\jfleg^
    --path ..\checkpoints\lang-8-fairseq-cnn\checkpoint_best.pt^
    --batch-size 128^
    --beam 5^
    --nbest 1^
    --raw-text^
    --source-lang en^
    --target-lang gec
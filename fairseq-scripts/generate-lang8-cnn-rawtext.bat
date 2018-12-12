:: copy lang-8's test files to test folder
copy ..\corpus\lang-8-fairseq\lang8-test.en ..\test\lang-8\test.en-gec.en
copy ..\corpus\lang-8-fairseq\lang8-test.gec ..\test\lang-8\test.en-gec.gec

:: copy lang-8's dictionary to test folder
copy ..\data-bin\lang-8-fairseq\dict.en.txt ..\test\lang-8\
copy ..\data-bin\lang-8-fairseq\dict.gec.txt ..\test\lang-8\

python .\generate.py^
    ..\test\lang-8^
    --path ..\checkpoints\lang-8-fairseq-cnn\checkpoint_best.pt^
    --batch-size 128^
    --beam 5^
    --nbest 12^
    --lang-model-data ..\data-bin\wiki103^
    --lang-model-path ..\data-bin\wiki103\wiki103.pt^
    --raw-text^
    --source-lang en^
    --target-lang gec
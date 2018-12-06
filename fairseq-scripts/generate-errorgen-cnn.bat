python .\generate-errorgen.py^
    ..\data-bin\errorgen-fairseq^
    --path ..\checkpoints\errorgen-fairseq-cnn\checkpoint_last.pt^
    --batch-size 128^
    --beam 5^
    --nbest 10^
    --gen-subset train^
    --lang-model-data ..\data-bin\wiki103^
    --lang-model-path ..\data-bin\wiki103\wiki103.pt^
    --out-dir ..\corpus\errorgen-fairseq^
    --fp16
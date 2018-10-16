python ..\OpenNMT-py\translate.py^
    -model model_step_60000.pt^
    -batch_size 1^
    -beam 5^
    -replace_unk^
    -src ..\test\translate.txt^
    -output ..\test\pred.txt
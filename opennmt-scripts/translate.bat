python ..\OpenNMT-py\translate.py^
    -model ..\opennmt\models\lang8_step_50000.pt^
    -batch_size 1^
    -beam 5^
    -replace_unk^
    -src ..\test\translate.txt^
    -output ..\test\pred.txt
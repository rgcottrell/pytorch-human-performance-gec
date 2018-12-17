import argparse
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize

def process(src_dir, out_dir, dataset, src_filenames, dataset_file_paths):
    src_filename = os.path.join(out_dir, 'clc_fce-' + dataset + '.en')
    tgt_filename = os.path.join(out_dir, 'clc_fce-' + dataset + '.gec')
    rtl_src_filename = os.path.join(out_dir, 'clc_fce-' + dataset + '-rtl.en')
    rtl_tgt_filename = os.path.join(out_dir, 'clc_fce-' + dataset + '-rtl.gec')

    # clean previous output
    open(src_filename, 'w').close()
    open(tgt_filename, 'w').close()
    open(rtl_src_filename, 'w').close()
    open(rtl_tgt_filename, 'w').close()

    filenames_file = os.path.join(src_dir, 'fce-error-detection', 'filenames', src_filenames)
    files = filenames(filenames_file)

    for file in files:
        process_file(file, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename, dataset_file_paths)

       
def process_file(file, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename, dataset_file_paths):
    """"Transform input xml file into tab delimited token sentences"""
    path_of_file = os.path.join(dataset_file_paths[file], file)
    print(path_of_file)
    with open(path_of_file, 'r') as xml_src:
        tree = ET.parse(xml_src)
        answer1 = tree.getroot().find('./head/text/answer1')
        answer2 = tree.getroot().find('./head/text/answer2')

        if answer1 is not None:
            process_answer(answer1, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename)

        if answer2 is not None:
            process_answer(answer2, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename)

def process_answer(answer_xml, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename):
    answer_partitions = list(answer_xml.find('.//coded_answer'))

    for elm_p in answer_partitions:
        sentence_permutations = list()
        remaining_text = ""

        ns_elements = elm_p.findall('./NS')

        if elm_p.text is not None:
            get_sentence_permutations(elm_p.text, sentence_permutations, ns_elements, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename)

def get_sentence_permutations(remaining_text, sentence_permutations, ns_elements, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename):
    should_continue = True

    while should_continue:

        end_of_sentence_idx = remaining_text.find('.')

        # write sentences until we can't find the end of a whole and assumed correct sentence
        while end_of_sentence_idx > 0:
            sentence = remaining_text[:end_of_sentence_idx + 1]
            remaining_text = remaining_text[end_of_sentence_idx + 1:]
            sentence_permutations.append(sentence)
            write_sentences(sentence_permutations, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename)
            end_of_sentence_idx = remaining_text.find('.')
            sentence_permutations = list()

        if len(sentence_permutations) == 0:
            sentence_permutations.append(remaining_text)

        # abort there is no sentence
        if len(ns_elements) == 0:
            should_continue = False
            continue

        elm_ns = ns_elements.pop(0)
        process_correction(sentence_permutations, elm_ns)

        #if text is following correction, check for end of sentence
        if elm_ns.tail is not None:
            tail_eos_idx = elm_ns.tail.find('.')

            if tail_eos_idx > -1:
                remaining_text = elm_ns.tail[tail_eos_idx + 1:]
                for i in range(0, len(sentence_permutations)):
                    sentence_permutations[i] = sentence_permutations[i] + elm_ns.tail[:tail_eos_idx + 1]

                write_sentences(sentence_permutations, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename)
                sentence_permutations = list()
                corrections = 0
                if len(remaining_text) == 0 and len(ns_elements) == 0:
                    should_continue = False
                elif len(remaining_text) > 0:
                    sentence_permutations.append(remaining_text)
                else:
                    sentence_permutations.append("")

            elif len(ns_elements) == 0:
                should_continue = False
            else:
                for i in range(0, len(sentence_permutations)):
                    sentence_permutations[i] = sentence_permutations[i] + elm_ns.tail

        elif len(ns_elements) == 0:
           should_continue = False

def process_correction(sentence_permutations, elm_ns):
    elm_i = elm_ns.find('./i')
    elm_c = elm_ns.find('./c')

    if elm_c is not None:
        incorrect_text = ""
        if elm_i is not None and elm_i.text is not None:
            incorrect_text = elm_i.text

        nested_correction = elm_c.find('./NS')
        if nested_correction is not None:
            preceding_text = ""
            if elm_c.text is not None:
                preceding_text = elm_c.text

            tailing_text = ""
            if nested_correction.tail is not None:
                tailing_text = nested_correction.tail

            traverse_corrections(sentence_permutations, incorrect_text, nested_correction, preceding_text, tailing_text)

        else:
            add_correction_to_permutations(sentence_permutations, incorrect_text, elm_c.text)

    else:
        if elm_i is not None and elm_i.text is not None:
            add_correction_to_permutations(sentence_permutations, elm_i.text, "")

def traverse_corrections(sentence_permutations, incorrect_text, elm_correction, preceding_text, tailing_text):
    elm_i = elm_correction.find('./i')
    elm_c = elm_correction.find('./c')

    if elm_i is not None:
        nested_correction = elm_i.find('./NS')
    else:
        nested_correction = None

    #base case
    if nested_correction is None:
        if elm_c is not None:
            add_correction_to_permutations_skip_previous(sentence_permutations, incorrect_text, preceding_text + elm_c.text + tailing_text)
        else:
            add_correction_to_permutations_skip_previous(sentence_permutations, incorrect_text, preceding_text + tailing_text)
    else:
        if elm_i.text is not None:
            preceding_text = preceding_text + elm_i.text

        if nested_correction.tail is not None:
            tailing_text = nested_correction.tail + tailing_text

        #add_correction_to_permutations(sentence_permutations, "", preceding_text + elm_c.text + tailing_text)
        add_nested_correction_to_permutations(sentence_permutations, preceding_text + elm_c.text + tailing_text)
        traverse_corrections(sentence_permutations, incorrect_text, nested_correction, preceding_text, tailing_text)

def add_correction_to_permutations(sentence_permutations, incorrect_text, correct_text):
    sentence_permutations.append(sentence_permutations[0] + correct_text)
    for i in range(0, len(sentence_permutations) - 1):
        sentence_permutations[i] = sentence_permutations[i] + incorrect_text

def add_correction_to_permutations_skip_previous(sentence_permutations, incorrect_text, correct_text):
    sentence_permutations.append(sentence_permutations[0] + correct_text)
    for i in range(0, len(sentence_permutations) - 2):
        sentence_permutations[i] = sentence_permutations[i] + incorrect_text

def add_nested_correction_to_permutations(sentence_permutations, correct_text):
    sentence_permutations.append(sentence_permutations[0] + correct_text)

def write_sentences(sentence_permutations, src_filename, tgt_filename, rtl_src_filename, rtl_tgt_filename):
    with open(src_filename, 'a') as src_out, open(tgt_filename, 'a') as tgt_out, open(rtl_src_filename, 'a') as rtl_src_out, open(rtl_tgt_filename, 'a') as rtl_tgt_out:
        permutations = len(sentence_permutations)
        first_sentence = strip_and_tokenize(sentence_permutations[0])
        first_rtl = " ".join(reversed(first_sentence.split()))
        if permutations == 1:
            src_out.write("{}\n".format(first_sentence))
            rtl_src_out.write("{}\n".format(first_rtl))
            tgt_out.write("{}\n".format(first_sentence))
            rtl_tgt_out.write("{}\n".format(first_rtl))

        else:
            for i in range(1, permutations):
                src_out.write("{}\n".format(first_sentence))
                rtl_src_out.write("{}\n".format(first_rtl))
                ith_sentence = strip_and_tokenize(sentence_permutations[i])
                ith_rtl = " ".join(reversed(ith_sentence.split()))
                tgt_out.write("{}\n".format(ith_sentence))
                rtl_tgt_out.write("{}\n".format(ith_rtl))

def strip_and_tokenize(sentence):
    tokens = word_tokenize(sentence)
    new_sentence = ""
    for token in tokens:
        new_sentence = new_sentence + " " + token

    return new_sentence.strip()

def filenames(filenames_file):
   """Returns list of files from filenames file"""
   with open(filenames_file, 'r') as file:
       files = []
       for line in file:
           line = line.strip()
           if(len(line) == 0):
               continue
           
           files.append(line)

   return files
            

def split_file(file, out1, out2, percentage=0.5):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    with open(file, 'r',encoding="utf-8") as fin, \
        open(out1, 'w') as fout1, \
        open(out2, 'w') as fout2:
    
        nLines = sum(1 for line in fin)
        fin.seek(0)
    
        nTrain = int(nLines*percentage) 
        nValid = nLines - nTrain
    
        i = 0
        for line in fin:
            if (i < nTrain) or (nLines - i > nValid):
                fout1.write(line)
                i += 1
            else:
                fout2.write(line)

def get_paths_of_xml_files(src_dir):
    """Returns dictionary with xml filenames as keys and paths as their values"""
    dataset_file_paths = dict()
    dataset = os.path.join(src_dir, 'fce-released-dataset', 'dataset')

    for subdir, dirs, files in os.walk(dataset):
        for file in files:
            dataset_file_paths[file] = subdir

    return dataset_file_paths


def main(opt):
    dataset_file_paths = get_paths_of_xml_files(opt.src_dir)
    process(opt.src_dir, opt.out_dir, 'train', 'fce-public.train.filenames.txt', dataset_file_paths)
    process(opt.src_dir, opt.out_dir, 'test', 'fce-public.test.filenames.txt', dataset_file_paths)
    
    # rename test file to temp file so that test can be split into validation and test sets
    os.rename(os.path.join(opt.out_dir, 'clc_fce-' + 'test' + '.en'), os.path.join(opt.out_dir, 'clc_fce-' + 'temp' + '.en'))
    os.rename(os.path.join(opt.out_dir, 'clc_fce-' + 'test' + '.gec'), os.path.join(opt.out_dir, 'clc_fce-' + 'temp' + '.gec'))
    os.rename(os.path.join(opt.out_dir, 'clc_fce-' + 'test-rtl' + '.en'), os.path.join(opt.out_dir, 'clc_fce-' + 'temp-rtl' + '.en'))
    os.rename(os.path.join(opt.out_dir, 'clc_fce-' + 'test-rtl' + '.gec'), os.path.join(opt.out_dir, 'clc_fce-' + 'temp-rtl' + '.gec'))

    # split
    split_file(
        os.path.join(opt.out_dir, 'clc_fce-' + 'temp' + '.en'),
        os.path.join(opt.out_dir, 'clc_fce-' + 'valid' + '.en'),
        os.path.join(opt.out_dir, 'clc_fce-' + 'test' + '.en')
    )
    split_file(
        os.path.join(opt.out_dir, 'clc_fce-' + 'temp' + '.gec'),
        os.path.join(opt.out_dir, 'clc_fce-' + 'valid' + '.gec'),
        os.path.join(opt.out_dir, 'clc_fce-' + 'test' + '.gec')
    )
    split_file(
        os.path.join(opt.out_dir, 'clc_fce-' + 'temp-rtl' + '.en'),
        os.path.join(opt.out_dir, 'clc_fce-' + 'valid-rtl' + '.en'),
        os.path.join(opt.out_dir, 'clc_fce-' + 'test-rtl' + '.en')
    )
    split_file(
        os.path.join(opt.out_dir, 'clc_fce-' + 'temp-rtl' + '.gec'),
        os.path.join(opt.out_dir, 'clc_fce-' + 'valid-rtl' + '.gec'),
        os.path.join(opt.out_dir, 'clc_fce-' + 'test-rtl' + '.gec')
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         description='transform-CLC_FCE.py',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-src_dir', required=True, help="Path to corpus source files")
    parser.add_argument('-out_dir', required=True, help="Path for transformed data files")

    opt = parser.parse_args()
    main(opt)

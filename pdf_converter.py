import sys
from pdf2image import convert_from_path
import csv
import os
from glob import glob
from PIL import Image
import numpy as np
import imgaug as ia

csv.field_size_limit(sys.maxsize)

start_line = 0
try:
    start_line = int(sys.argv[1])
    print('Start line is ' + start_line)
except Exception:
    print('Start line is 0')


try:
	os.mkdir('is_ocr_images')
except Exception:
	print('Already exists')


def replace_folder_name(path):
    path = path.replace('ocr_pdfs/', '')
    return path

paths = glob('ocr_pdfs/*.pdf')
print(len(paths))
paths_dict = {}
for path in paths:
    new_path = replace_folder_name(path)
    splitted_name = new_path.split('_')
    process_id = splitted_name[1]
    document_id = splitted_name[2]
    key = process_id + '_' + document_id
    try:
        print(paths_dict[key])
    except Exception as e:
        paths_dict[key] = []
    paths_dict[key].append(new_path)


keys = ['process_id', 'process_class', 'process_processing_date', 'process_process_time', 'process_is_complete', 'document_id', 'document_processing_date', 'document_processing_time', 'page_is_ocr', 'page_body', 'page_text_extract', 'page_number', 'page_image', 'page_piece']
with open('is_ocr.csv') as csv_file:
    file = csv.reader(csv_file)
    next(file, None)
    limit = start_line + 10000
    index = -1
    for row in file:
        index += 1
        if index < start_line:
            continue
        if index == limit:
            break
        process_id = row[0]
        doc_id = row[5]
        key = process_id + '_' + doc_id
        pg_number = int(row[11])
        print('[index: {}] Getting {}_{} document. Current page: {}'.format(index, process_id, doc_id, pg_number))
        path = 'ocr_pdfs/' + paths_dict[key][0]
        pages = convert_from_path(path, thread_count=4, grayscale=True, size=(800), fmt='jpg', first_page=pg_number, last_page=pg_number)
        pages[0].save('is_ocr_images/' + key + '_' + str(pg_number) + '.jpg', 'JPEG')

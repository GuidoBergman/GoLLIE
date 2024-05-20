#!/bin/bash

SEMEVAL_DATA_PATH="data/original/data"
OUTPUT_PATH="data"

mkdir ${OUTPUT_PATH}/train/
mkdir ${OUTPUT_PATH}/dev/
mkdir ${OUTPUT_PATH}/train/articles
mkdir ${OUTPUT_PATH}/dev/articles
mkdir ${OUTPUT_PATH}/train/labels
mkdir ${OUTPUT_PATH}/dev/labels


cp -r ${SEMEVAL_DATA_PATH}/*/train-articles-subtask-3/*  ${OUTPUT_PATH}/train/articles
cp -r ${SEMEVAL_DATA_PATH}/*/train-labels-subtask-3-spans/* ${OUTPUT_PATH}/train/labels

cp -r ${SEMEVAL_DATA_PATH}/*/dev-articles-subtask-3/* ${OUTPUT_PATH}/dev/articles
cp -r ${SEMEVAL_DATA_PATH}/*/dev-labels-subtask-3-spans/* ${OUTPUT_PATH}/dev/labels

# Delete duplicated articles
find ${OUTPUT_PATH}/train/articles -type f -name '*(*).txt'  -delete
find ${OUTPUT_PATH}/dev/articles -type f -name '*(*).txt'  -delete
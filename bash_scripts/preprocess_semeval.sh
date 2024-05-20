#!/bin/bash

SEMEVAL_DATA_PATH=""

cp -r ${SEMEVAL_DATA_PATH}/*/train-articles-subtask-3/* train/articles
cp -r ${SEMEVAL_DATA_PATH}/*/train-labels-subtask-3-spans/* train/labels

cp -r ${SEMEVAL_DATA_PATH}/*/dev-articles-subtask-3/* dev/articles
cp -r ${SEMEVAL_DATA_PATH}/*/dev-labels-subtask-3-spans/* dev/labels

# Delete duplicated articles
find train/articles -type f -name '*(*).txt'  -delete
find dev/articles -type f -name '*(*).txt'  -delete
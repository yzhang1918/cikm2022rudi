#!/usr/bin/env bash

set -e  # exit on errors

folder=$1
device=$2
teacher=$3

echo '==================================================='
echo "{{ Automatic Statistics Generation: $teacher }}"
python extract_stats.py --prefix $teacher --folder $folder

echo '==================================================='
echo "{{ RuDi: output=$folder/$teacher-rudi_rules }}"
python extract_rules.py --prefix $teacher --folder $folder --device $device 

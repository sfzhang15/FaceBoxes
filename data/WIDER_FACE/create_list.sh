#!/bin/bash

root_dir=$HOME/data/WIDER_FACE_VOC/
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in train
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  echo "Create list for $dataset..."
  dataset_file=$root_dir/$sub_dir/$dataset.txt

  img_file=$bash_dir/$dataset"_img.txt"
  cp $dataset_file $img_file
  sed -i "s/^/JPEGImages\//g" $img_file
  sed -i "s/$/.jpg/g" $img_file

  label_file=$bash_dir/$dataset"_label.txt"
  cp $dataset_file $label_file
  sed -i "s/^/Annotations\//g" $label_file
  sed -i "s/$/.xml/g" $label_file

  paste -d' ' $img_file $label_file >> $dst_file

  rm -f $label_file
  rm -f $img_file

  # Shuffle train file.
  rand_file=$dst_file.random
  cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
  mv $rand_file $dst_file
done

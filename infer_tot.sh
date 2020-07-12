#!/bin/bash
workdir=$(cd $(dirname $0); pwd)

echo $workdir

OrgImagepath=$workdir"/OrgImage"

Outputpath=$workdir"/Output/"

simrdwnpath=$workdir"/simrdwn/simrdwn/core/simrdwn_unosat.py"

simrdwntestimgpath=$workdir"/simrdwn/data/test_images/building/"

unosatinfergeopath=$workdir"/DetectoRS-master-UNOSAT/tools/infer_json.py"

configpath=$workdir"/DetectoRS-master-UNOSAT/configs/DetectoRS/DetectoRS_mstrain_400_1200_r50_40e_ok.py"

ckptpath=$workdir"/DetectoRS-master-UNOSAT/work_dirs/DetectoRS_mstrain_400_1200_r50_40e.py/epoch_19.pth"

test_unosat_building_path=$workdir"/simrdwn/results/test_unosat_building"

# score_thr=0.6

echo $test_unosat_building_path

echo $OrgImagepath


for f in $(find $OrgImagepath -iname "*.*"); do



  #make test path
  if [ ! -d "$test_unosat_building_path"  ];then
    mkdir $test_unosat_building_path
    mkdir $test_unosat_building_path'/logs'
  else
    echo dir exist
  fi

  echo $f

  filename=$(basename $f)
  echo $filename

  #.tif or .TIF
  extension=".${filename##*.}"  
  echo $extension

  cp $f $simrdwntestimgpath

  echo $simrdwntestimgpath

  # split TIF
  python $simrdwnpath --framework unosat --mode test --outname building --label_map_path class_labels_building.pbtxt --testims_dir building --keep_test_slices 0 --test_slice_sep __ --edge_buffer_test 1 --test_box_rescale_frac 1 --slice_sizes_str 416 --slice_overlap 0.2 --alpha_scaling 1 --show_labels 1 --only_pre 1
  



  # find the split dir
  PATH_TO_TEST_DIR="$(find $test_unosat_building_path -maxdepth 1 -name "*split")" 
  echo $PATH_TO_TEST_DIR

  #infer patches
  python $unosatinfergeopath $configpath $ckptpath --image_dir $PATH_TO_TEST_DIR --output_dir $test_unosat_building_path --score_thr 0.1




  # add geo refference
  python $simrdwnpath --framework unosat --mode test --outname building --label_map_path class_labels_building.pbtxt --testims_dir building --keep_test_slices 0 --test_slice_sep __ --edge_buffer_test 1 --test_box_rescale_frac 1 --slice_sizes_str 416 --slice_overlap 0.2 --alpha_scaling 1 --show_labels 1 --only_pre 0




  #move the geojson to the ouput dir
  echo $(find $test_unosat_building_path -name "*.GeoJSON")
  mv $(find $test_unosat_building_path -name "*.GeoJSON")  $test_unosat_building_path"/`basename $f $extension`.GeoJSON"
  cp $(find $test_unosat_building_path -name "*.GeoJSON")  $Outputpath



  # echo $test_unosat_building_path"_$(basename $f $extension)"
  mv $test_unosat_building_path $test_unosat_building_path"_$(basename $f $extension)"

  rm $simrdwntestimgpath$filename

done

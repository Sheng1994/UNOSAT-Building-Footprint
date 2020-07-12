# Modification
Mainly add the part of passing configs for building the pipeline

```
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                        help="root path to the orignal data")
    parser.add_argument('--datasetname', type=str, default='buildingfootprint',
                        help="dataset name")
    parser.add_argument('--clip_size', type=int, default=512,
                        help="TIF image patch clip size")    
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help="dataset train part ratio")      

    args = parser.parse_args()


 % root path for saving the tif and shp file.
    ROOT = args.dataroot
    img_path = 'img'
    shp_path = 'shp'
% root path for saving the mask.
    IMAGE_DIR = os.path.join(ROOT, args.datasetname)
    ANNOTATION_DIR = os.path.join(ROOT, "annotations")
    
    clip_from_file(args.clip_size, ROOT, img_path, shp_path,datasetname =args.datasetname,ann_path = 'annotations')
    slice(ROOT, train=args.train_ratio, eval=1-args.train_ratio, test=0.0,datasetname =args.datasetname,ann_path = 'annotations')
    from_mask_to_coco(ROOT, 'train', args.datasetname, "annotations")
    from_mask_to_coco(ROOT, 'eval', args.datasetname, "annotations")
    from_mask_to_coco(ROOT, 'test', args.datasetname, "annotations")
```

# shp2coco
shp2coco is a tool to help create `COCO` datasets from `.shp` file (ArcGIS format). <br>

It includes:<br>
1:mask tif with shape file.<br>
2:crop tif and mask.<br>
3:slice the dataset into training, eval and test subset.<br>
4:generate annotations in uncompressed RLE ("crowd") and polygons in the format COCO requires.<br>

This project is based on [geotool](https://github.com/Kindron/geotool) and [pycococreator](https://github.com/waspinator/pycococreator)

## Usage:
If you need to generate annotations in the COCO format, try the following:<br>
`python shape_to_coco.py`<br>
If you need to visualize annotations, try the following:<br>
`python visualize_coco.py`<br>

## Example:
![example](https://github.com/DuncanChen2018/shp2coco/blob/master/example_data/example.png)

## Thanks to the Third Party Libs
[geotool](https://github.com/Kindron/geotool)<br>
[pycococreator](https://github.com/waspinator/pycococreator)<br>

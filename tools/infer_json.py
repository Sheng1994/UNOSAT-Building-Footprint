from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector

import pycocotools.mask as maskUtils
import pandas as pd
import numpy as np
import mmcv
import os
import cv2
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--score_thr', type=float, default=0.3, help='bbox score threshold')

   # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    imglist = []
    for f in os.listdir(args.image_dir):
        imglist.append(os.path.join(args.image_dir, f))
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    buildings = pd.DataFrame(columns=('Loc_Tmp','Prob','Xmin','Ymin','Xmax','Ymax','Seg'))
    k=0

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for i in tqdm(imglist):
        # test a single image
        result = inference_detector(model, i)
        # # show the results
        # show_result_pyplot(model, i, result, score_thr=args.score_thr,outfile=args.outfile)

        #modified from def show_result

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        bboxes = np.vstack(bbox_result)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > args.score_thr)[0]

            if len(inds)<1:
                continue

            for j in inds:
                j = int(j)

                # segm need to be  'uint8' to findContours
                segm = maskUtils.decode(segms[j]).astype(np.uint8)


                #sometime segm are all zero
                # segm_xy = [i[0] for i in contour[0].tolist()]
                # IndexError: list index out of range
                if np.max(segm) == 0:
                    continue

                contour, hier = cv2.findContours(
                    segm.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


                segm_xy = [i[0] for i in contour[0].tolist()]
                #ValueError: A LinearRing must have at least 3 coordinate tuples
                if len(segm_xy) < 3:
                    continue

                bbox = bboxes[j, :4]
                score = bboxes[j, -1]

                buildings.loc[k] = [i, score, bbox[0], bbox[1], bbox[2], bbox[3], segm_xy]
                k = k + 1

    buildings.to_json(os.path.join(args.output_dir, 'building'+".json"))

if __name__ == '__main__':
    main()

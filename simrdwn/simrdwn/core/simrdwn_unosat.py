# coding=utf-8
from __future__ import print_function
import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import argparse
import shutil
import copy
# import logging
# import tensorflow as tf

import utils
import post_process_unosat
import add_geo_coords
#import parse_tfrecord
import preprocess_tfrecords
import slice_im

######################
from osgeo import gdal
from gdalconst import *
#######################


sys.stdout.flush()

# def tifproextract(img_path):
#     dataset = gdal.Open(img_path, GA_ReadOnly)
#     im_proj = dataset.GetProjection()  # 获取投影信息
#     return im_proj.split(",")[-1].split("\"")[1]


def tifproextract(im_root_with_ext, testims_dir_tot):
    """
    1. get image path (args.test_image_tmp) from image root name
            (args.test_image_tmp)
    2. get image projection
    """
    img_path = os.path.join(testims_dir_tot, im_root_with_ext)
    dataset = gdal.Open(img_path, GA_ReadOnly)
    im_proj = dataset.GetProjection()  
    pro = "epsg:"+im_proj.split(",")[-1].split("\"")[1]

    return pro


def update_args(args):

    args.res_name = args.mode + '_' + args.framework + '_' + args.outname  #
    args.core_dir = os.path.dirname(os.path.realpath(__file__))
    args.this_file = os.path.abspath(__file__)
    args.simrdwn_dir = os.path.dirname(os.path.dirname(args.core_dir))
    args.results_topdir = os.path.join(args.simrdwn_dir, 'results')
    args.results_dir = os.path.join(args.results_topdir, args.res_name)
    args.log_dir = os.path.join(args.results_dir, 'logs')
    args.log_file = os.path.join(args.log_dir, args.res_name + '.log')
    args.extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG',
                           '.jpg', '.JPEG', '.jpeg']
    # if train_data_dir is not a full directory, set it as within simrdwn
    if args.train_data_dir.startswith('/'):
        pass
    else:
        args.train_data_dir = os.path.join(args.simrdwn_dir, 'data/train_data')


    # keep raw testims dir if it starts with a '/'
    if args.testims_dir.startswith('/'):
        args.testims_dir_tot = args.testims_dir

    else:
        args.testims_dir_tot = os.path.join(args.simrdwn_dir,
                                            'data/test_images',
                                            args.testims_dir)

    args.test_ims_list = [f for f in os.listdir(args.testims_dir_tot)
                          if f.endswith(tuple(args.extension_list))][:1]

    # set total location of test image file list
    args.test_presliced_list_tot = os.path.join(
        args.results_topdir, args.test_presliced_list)

    if len(args.test_presliced_list) > 0:
        args.test_splitims_locs_file = args.test_presliced_list_tot
    else:
        args.test_splitims_locs_file = os.path.join(
            args.results_dir, args.test_splitims_locs_file_root)

    # plot thresh and slice sizes
    args.plot_thresh = np.array(
        args.plot_thresh_str.split(args.str_delim)).astype(float)
    args.slice_sizes = np.array(
        args.slice_sizes_str.split(args.str_delim)).astype(int)
    args.unosat_test_classes_files = [os.path.join(args.results_dir, 'building.json')]


    # update label_map_path, if needed
    if (args.label_map_path.startswith('/')) or (len(args.label_map_path) == 0):
        pass
    else:
        args.label_map_path = os.path.join(args.train_data_dir,
                                           args.label_map_path)


   

    # make label_map_dic (key=int, value=str), and reverse
    if len(args.label_map_path) > 0:
        args.label_map_dict = preprocess_tfrecords.load_pbtxt(
            args.label_map_path, verbose=False)
        # ensure dict is 1-indexed
        if min(list(args.label_map_dict.keys())) != 1:
            print("Error: label_map_dict (", args.label_map_path, ") must"
                  " be 1-indexed")
            return
    else:
        args.label_map_dict = {}

    args.rotate_boxes = bool(args.rotate_boxes)
    args.test_add_geo_coords = bool(args.test_add_geo_coords)
    args.label_map_dict_tot = copy.deepcopy(args.label_map_dict)

    args.unosat_object_labels = args.unosat_object_labels_str.split(',')
    # also set label_map_dict, if it's empty
    # if len(args.label_map_path) == 0:
    #     for itmp, val in enumerate(args.yolt_object_labels):
    #         args.label_map_dict[itmp] = val
    #     args.label_map_dict_rev = {v: k for k,
    #                                v in args.label_map_dict.items()}
    args.unosat_classnum = len(args.unosat_object_labels)
    args.labels_log_file = os.path.join(args.log_dir, 'labels_list.txt')
    args.val_df_path_aug = os.path.join(args.results_dir, args.val_df_root_aug)

    im_root_with_ext = args.test_ims_list[0]

    args.proj_str = tifproextract(im_root_with_ext,args.testims_dir_tot)

    args.only_pre = bool(args.only_pre)

    if len(args.building_csv_file) > 0:
        args.building_csv_file = os.path.join(
            args.results_dir, args.building_csv_file)

    return args

def split_test_im(im_root_with_ext, testims_dir_tot, results_dir,
                  log_file,
                  slice_sizes=[416],
                  slice_overlap=0.2,
                  test_slice_sep='__',
                  zero_frac_thresh=0.5,
                  ):
    """
    Split files for test step
    Assume input string has no path, but does have extension (e.g:, 'pic.png')

    1. get image path (args.test_image_tmp) from image root name
            (args.test_image_tmp)
    2. slice test image and move to results dir
    """

    # get image root, make sure there is no extension
    im_root = im_root_with_ext.split('.')[0]
    im_path = os.path.join(testims_dir_tot, im_root_with_ext)

    # slice test plot into manageable chunks

    # slice (if needed)
    if slice_sizes[0] > 0:
        # if len(args.slice_sizes) > 0:
        # create test_splitims_locs_file
        # set test_dir as in results_dir
        test_split_dir = os.path.join(results_dir,  im_root + '_split' + '/')
        test_dir_str = '"test_split_dir: ' + test_split_dir + '\n"'
        print("test_dir:", test_dir_str[1:-2])
        os.system('echo ' + test_dir_str + ' >> ' + log_file)
        # print "test_split_dir:", test_split_dir

        # clean out dir, and make anew
        if os.path.exists(test_split_dir):
            if (not test_split_dir.startswith(results_dir)) \
                    or len(test_split_dir) < len(results_dir) \
                    or len(test_split_dir) < 10:
                print("test_split_dir too short!!!!:", test_split_dir)
                return
            shutil.rmtree(test_split_dir, ignore_errors=True)
        os.mkdir(test_split_dir)

        # slice
        for s in slice_sizes:
            slice_im.slice_im(im_path, im_root,
                              test_split_dir, s, s,
                              zero_frac_thresh=zero_frac_thresh,
                              overlap=slice_overlap,
                              slice_sep=test_slice_sep)
            test_files = [os.path.join(test_split_dir, f) for
                          f in os.listdir(test_split_dir)]
        n_files_str = '"Num files: ' + str(len(test_files)) + '\n"'
        print(n_files_str[1:-2])
        os.system('echo ' + n_files_str + ' >> ' + log_file)

    else:
        test_files = [im_path]
        test_split_dir = os.path.join(results_dir, 'nonsense')

    return test_files, test_split_dir

def run_test(args,framework='YOLT2',
             infer_cmd='',
             results_dir='',
             log_file='',
             n_files=0,
             # test_tfrecord_out='',
             slice_sizes=[416],
             testims_dir_tot='',
             unosat_test_classes_files='',
             label_map_dict={},
             # val_df_path_init='',
             test_slice_sep='__',
             edge_buffer_test=1,
             max_edge_aspect_ratio=4,
             test_box_rescale_frac=1.0,
             rotate_boxes=False,
             # min_retain_prob=0.05,
             test_add_geo_coords=True,
             verbose=False
             ):
    """Evaluate multiple large images"""

    t0 = time.time()
    # run command
    print("Running", infer_cmd)
    os.system('echo ' + infer_cmd + ' >> ' + log_file)

    ###infer
    # os.system(infer_cmd)  # run_cmd(outcmd)
    ###


    t1 = time.time()
    cmd_time_str = '"\nLength of time to run command: ' + infer_cmd \
        + ' for ' + str(n_files) + ' cutouts: ' \
        + str(t1 - t0) + ' seconds\n"'
    print(cmd_time_str[1:-1])
    os.system('echo ' + cmd_time_str + ' >> ' + log_file)

    if framework.upper()  in ['YOLT2', 'YOLT3']:
      pass

    elif framework.upper()  in ['UNOSAT']:
        # post-process
        # df_tot = post_process_yolt_test_create_df(args)

        #   Create dataframe from yolt output text files.
        #   yolt_test_classes_files  car.txt object detection loc

        df_tot = post_process_unosat.post_process_unosat_test_create_df(
            unosat_test_classes_files,log_file,
            testims_dir_tot=testims_dir_tot,
            slice_sizes=slice_sizes,
            slice_sep=test_slice_sep,
            edge_buffer_test=edge_buffer_test,
            max_edge_aspect_ratio=max_edge_aspect_ratio,
            test_box_rescale_frac=test_box_rescale_frac,
            rotate_boxes=rotate_boxes)

    ###########################################
    # plot

    # add geo coords to eall boxes?
    if test_add_geo_coords and len(df_tot) > 0:
        ###########################################
        # !!!!! Skip?
        # json = None
        ###########################################
        df_tot, json = add_geo_coords.add_geo_coords_to_df(
            df_tot, create_geojson=True, inProj_str=args.proj_str,
            outProj_str=args.proj_str, verbose=verbose)
    else:
        json = None

    return df_tot, json

def prep_test_files(results_dir, log_file, test_ims_list,
                    testims_dir_tot, test_splitims_locs_file,
                    slice_sizes=[416],
                    slice_overlap=0.2,
                    test_slice_sep='__',
                    zero_frac_thresh=0.5,
                    ):
    """Split images and save split image locations to txt file"""
    # split test images, store locations
    t0 = time.time()
    test_split_str = '"Splitting test files...\n"'
    print(test_split_str[1:-2])
    os.system('echo ' + test_split_str + ' >> ' + log_file)
    print("test_ims_list:", test_ims_list)

    test_files_locs_list = []
    test_split_dir_list = []
    # !! Should make a tfrecord when we split files, instead of doing it later
    for i, test_base_tmp in enumerate(test_ims_list):
        iter_string = '"\n' + str(i+1) + ' / ' + \
            str(len(test_ims_list)) + '\n"'
        print(iter_string[1:-2])
        os.system('echo ' + iter_string + ' >> ' + log_file)
        # print "\n", i+1, "/", len(args.test_ims_list)

        test_base_string = '"test_file: ' + str(test_base_tmp) + '\n"'
        print(test_base_string[1:-2])
        os.system('echo ' + test_base_string + ' >> ' + log_file)

        # split data
        test_files_list_tmp, test_split_dir_tmp = \
            split_test_im(test_base_tmp, testims_dir_tot,
                          results_dir, log_file,
                          slice_sizes=slice_sizes,
                          slice_overlap=slice_overlap,
                          test_slice_sep=test_slice_sep,
                          zero_frac_thresh=zero_frac_thresh)
        # add test_files to list
        test_files_locs_list.extend(test_files_list_tmp)
        test_split_dir_list.append(test_split_dir_tmp)

    # swrite test_files_locs_list to file (file = test_splitims_locs_file)
    print("Total len test files:", len(test_files_locs_list))
    print("test_splitims_locs_file:", test_splitims_locs_file)
    # write list of files to test_splitims_locs_file
    with open(test_splitims_locs_file, "w") as fp:
        for line in test_files_locs_list:
            if not line.endswith('.DS_Store'):
                fp.write(line + "\n")

    t1 = time.time()
    cmd_time_str = '"\nLength of time to split test files: ' \
        + str(t1 - t0) + ' seconds\n"'
    print(cmd_time_str[1:-2])
    os.system('echo ' + cmd_time_str + ' >> ' + log_file)

    return test_files_locs_list, test_split_dir_list


def prep(args):
    """Prep data for test

    Arguments
    ---------
    args : Namespace
        input arguments

    Returns
    -------
    test_cmd_tot : str
        Testing command

    """
    print("\nSIMRDWN now...\n")
    os.chdir(args.simrdwn_dir)
    print("cwd:", os.getcwd())
    # t0 = time.time()

    # make dirs
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
        os.mkdir(args.log_dir)

    # create log file
    # print("Date string:", args.date_string)
    # os.system('echo ' + str(args.date_string) + ' > ' + args.log_file)
    # init to the contents in this file?
    # os.system('cat ' + args.this_file + ' >> ' + args.log_file)
    args_str = '"\nArgs: ' + str(args) + '\n"'
    print(args_str[1:-1])
    os.system('echo ' + args_str + ' >> ' + args.log_file)

    # copy this file (yolt_run.py) as well as config, plot file to results_dir
    shutil.copy2(args.this_file, args.log_dir)
    # shutil.copy2(args.yolt_plot_file, args.log_dir)
    # shutil.copy2(args.tf_plot_file, args.log_dir)
    print("log_dir:", args.log_dir)

    # print ("\nlabel_map_dict:", args.label_map_dict)
    print("\nlabel_map_dict_tot:", args.label_map_dict_tot)
    # print ("object_labels:", args.object_labels)
    print("unosat_object_labels:", args.unosat_object_labels)
    print("unosat_classnum:", args.unosat_classnum)

    # save labels to log_dir
    # pickle.dump(args.object_labels, open(args.log_dir \
    #                                + 'labels_list.pkl', 'wb'), protocol=2)
    with open(args.labels_log_file, "w") as fp:
        for ob in args.unosat_object_labels:
            fp.write(str(ob) + "\n")

    test_cmd_tot = ""

    return test_cmd_tot

def execute(args,test_cmd_tot,only_pre):
    t3 = time.time()
    # load presliced data, if desired
    if only_pre==True:
        print("Prepping test files")
        test_files_locs_list, test_split_dir_list = \
            prep_test_files(args.results_dir, args.log_file,
                            args.test_ims_list,
                            args.testims_dir_tot,
                            args.test_splitims_locs_file,
                            slice_sizes=args.slice_sizes,
                            slice_overlap=args.slice_overlap,
                            test_slice_sep=args.test_slice_sep,
                            zero_frac_thresh=args.zero_frac_thresh,)
        print("Done prepping test files, ending")

    else:

        df_tot, json = run_test(args=args,infer_cmd=test_cmd_tot,
                                framework=args.framework,
                                results_dir=args.results_dir,
                                log_file=args.log_file,
                                # test_tfrecord_out=args.test_tfrecord_out,
                                slice_sizes=args.slice_sizes,
                                testims_dir_tot=args.testims_dir_tot,
                                unosat_test_classes_files=args.unosat_test_classes_files,
                                label_map_dict=args.label_map_dict,
                                # val_df_path_init=args.val_df_path_init,
                                # val_df_path_aug=args.val_df_path_aug,
                                # min_retain_prob=args.min_retain_prob,
                                test_slice_sep=args.test_slice_sep,
                                edge_buffer_test=args.edge_buffer_test,
                                max_edge_aspect_ratio=args.max_edge_aspect_ratio,
                                test_box_rescale_frac=args.test_box_rescale_frac,
                                rotate_boxes=args.rotate_boxes,
                                test_add_geo_coords=args.test_add_geo_coords,
                                )

        if len(df_tot) == 0:
            print("No detections found!")
        else:
            # save to csv
            df_tot.to_csv(args.val_df_path_aug, index=False)
            # get number of files
            n_files = len(np.unique(df_tot['Loc_Tmp'].values))
            # n_files = str(len(test_files_locs_list)
            t4 = time.time()
            cmd_time_str = '"Length of time to run test for ' \
                           + str(n_files) + ' files = ' \
                           + str(t4 - t3) + ' seconds\n"'
            print(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

        # run again, if desired

        # refine and plot
        t8 = time.time()
        if len(args.slice_sizes) > 0:
            sliced = True
        else:
            sliced = False
        print("test data sliced?", sliced)
        # import pdb
        # pdb.set_trace()

        # refine for each plot_thresh (if we have detections)
        if len(df_tot) > 0:
            for plot_thresh_tmp in args.plot_thresh:
                print("Plotting at:", plot_thresh_tmp)
                groupby = 'Image_Path'
                groupby_cat = 'Category'
                # Remove elements below detection threshold, and apply non-max suppression.
                df_refine = post_process_unosat.refine_df(df_tot,
                                                   groupby=groupby,
                                                   groupby_cat=groupby_cat,
                                                   nms_overlap_thresh=args.nms_overlap_thresh,
                                                   plot_thresh=plot_thresh_tmp,
                                                   verbose=False)
                # make some output plots, if desired
                # if len(args.building_csv_file) > 0:
                #     building_csv_file_tmp = args.building_csv_file.split('.')[0] \
                #                             + '_plot_thresh_' + str(plot_thresh_tmp).replace('.', 'p') \
                #                             + '.csv'
                # else:
                #     building_csv_file_tmp = ''
                #
                # if args.n_test_output_plots > 0:
                #     post_process_unosat.plot_refined_df(df_refine, groupby=groupby,
                #                                  label_map_dict=args.label_map_dict_tot,
                #                                  outdir=args.results_dir,
                #                                  plot_thresh=plot_thresh_tmp,
                #                                  show_labels=bool(
                #                                      args.show_labels),
                #                                  alpha_scaling=bool(
                #                                      args.alpha_scaling),
                #                                  plot_line_thickness=args.plot_line_thickness,
                #                                  print_iter=5,
                #                                  n_plots=args.n_test_output_plots,
                #                                  building_csv_file=building_csv_file_tmp,
                #                                  shuffle_ims=bool(
                #                                      args.shuffle_val_output_plot_ims),
                #                                  verbose=False)

                # geo coords?
                if bool(args.test_add_geo_coords):
                    df_refine, json = add_geo_coords.add_geo_coords_to_df(
                        df_refine,
                        create_geojson=bool(args.save_json),
                        # inProj_str='epsg:32637', outProj_str='epsg:32637',
                        inProj_str=args.proj_str, outProj_str=args.proj_str,
                        verbose=False)

                # save df_refine
                outpath_tmp = os.path.join(args.results_dir,
                                           args.val_prediction_df_refine_tot_root_part +
                                           '_thresh=' + str(plot_thresh_tmp) + '.csv')
                # df_refine.to_csv(args.val_prediction_df_refine_tot)
                df_refine.to_csv(outpath_tmp)
                print("Num objects at thresh:", plot_thresh_tmp, "=",
                      len(df_refine))
                # save json
                # import pdb
                # pdb.set_trace()
                if bool(args.save_json) and (len(json) > 0):
                    output_json_path = os.path.join(args.results_dir,
                                                    args.val_prediction_df_refine_tot_root_part +
                                                    '_thresh=' + str(plot_thresh_tmp) + '.GeoJSON')
                    json.to_file(output_json_path, driver="GeoJSON")

            cmd_time_str = '"Length of time to run refine_test()' + ' ' \
                           + str(time.time() - t8) + ' seconds"'
            print(cmd_time_str[1:-1])
            os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

        # remove or zip test_split_dirs to save space
        # if len(test_split_dir_list) > 0:
        #     for test_split_dir_tmp in test_split_dir_list:
        #         if os.path.exists(test_split_dir_tmp):
        #             # compress image chip dirs if desired
        #             if args.keep_test_slices:
        #                 print("Compressing image chips...")
        #                 shutil.make_archive(test_split_dir_tmp, 'zip',
        #                                     test_split_dir_tmp)
        #             # remove unzipped folder
        #             print("Removing test_split_dir_tmp:", test_split_dir_tmp)
        #             # make sure that test_split_dir_tmp hasn't somehow been shortened
        #             #  (don't want to remove "/")
        #             if len(test_split_dir_tmp) < len(args.results_dir):
        #                 print("test_split_dir_tmp too short!!!!:",
        #                       test_split_dir_tmp)
        #                 return
        #             else:
        #                 print("Removing image chips...")
        #
        #                 shutil.rmtree(test_split_dir_tmp, ignore_errors=True)

        cmd_time_str = '"Total Length of time to run test' + ' ' \
                       + str(time.time() - t3) + ' seconds\n"'
        print(cmd_time_str[1:-1])
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)


    # print ("\nNo honeymoon. This is business.")
    print("\n\n\nWell, I'm glad we got that out of the way.\n\n\n\n")
    return

def main():

    # Construct argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--framework', type=str, default='unosat',
                        help="object detection framework [unosat,yolt2, 'yolt3', ssd, faster_rcnn]")
    parser.add_argument('--mode', type=str, default='test',
                        help="[train, test]")
    parser.add_argument('--outname', type=str, default='tmp',
                        help="unique name of output")
    parser.add_argument('--testims_dir', type=str, default='test_images',
                        help="Location of test images (look within simrdwn_dir unless begins with /)")
    parser.add_argument('--test_presliced_list', type=str, default='',
                        help="Location of presliced training data list "
                        + " if empty, use tfrecord")
    parser.add_argument('--test_splitims_locs_file_root', type=str, default='test_splitims_input_files.txt',
                        help="Root of test_splitims_locs_file")
    parser.add_argument('--slice_sizes_str', type=str, default='416',
                        help="Proposed pixel slice sizes for test, will be split"
                        + " into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])"
                        + "(Set to < 0 to not slice")
    parser.add_argument('--str_delim', type=str, default=',',
                        help="Delimiter for string lists")
    parser.add_argument('--slice_overlap', type=float, default=0.35,
                        help="Overlap fraction for sliding window in test")
    parser.add_argument('--test_slice_sep', type=str, default='__',
                        help="Character(s) to split test image file names")
    parser.add_argument('--zero_frac_thresh', type=float, default=0.5,
                        help="If less than this value of an image chip is "
                        + "blank, skip it")
    parser.add_argument('--label_map_path', type=str,
                        default='',
                        help="Object classes, if not starts with '/', "
                        "assume it's in train_data_dir")
    parser.add_argument('--edge_buffer_test', type=int, default=-1000,
                        help="Buffer around slices to ignore boxes (helps with"
                        + " truncated boxes and stitching) set <0 to turn off"
                        + " if not slicing test ims")
    parser.add_argument('--max_edge_aspect_ratio', type=float, default=12,
                        help="Max aspect ratio of any item within the above "
                        + " buffer")
    parser.add_argument('--test_box_rescale_frac', type=float, default=1.0,
                        help="Defaults to 1, rescale output boxes if training"
                        + " boxes are the wrong size")
    parser.add_argument('--rotate_boxes', type=int, default=0,
                        help="Attempt to rotate output boxes using hough lines")
    parser.add_argument('--test_add_geo_coords', type=int, default=1,
                        help="switch to add geo coords to test outputs")
    parser.add_argument('--unosat_object_labels_str', type=str, default='building',
                        help="unosat labels str: building,car,boat,giraffe")
    parser.add_argument('--keep_test_slices', type=int, default=0,
                        help="Switch to retain sliced test files")
    parser.add_argument('--alpha_scaling', type=int, default=0,
                        help="Switch to scale box alpha with probability")
    parser.add_argument('--show_labels', type=int, default=0,
                        help="Switch to use show object labels")
    parser.add_argument('--train_data_dir', type=str, default='',
                        help="folder holding training image names, if empty "
                        "simrdwn_dir/data/")
    parser.add_argument('--val_df_root_aug', type=str, default='test_predictions_aug.csv',
                        help="Results in dataframe format")
    parser.add_argument('--plot_thresh_str', type=str, default='0.1',
                        help="Proposed thresholds to try for test, will be split"
                        + " into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--nms_overlap_thresh', type=float, default=0.5,
                        help="Overlap threshold for non-max-suppresion in python"
                        + " (set to <0 to turn off)")
    # if evaluating spacenet data
    parser.add_argument('--building_csv_file', type=str, default='',
                        help="csv file for spacenet outputs")
    parser.add_argument('--n_test_output_plots', type=int, default=10,
                        help="Switch to save test pngs")
    parser.add_argument('--plot_line_thickness', type=int, default=2,
                        help="Thickness for test output bounding box lines")
    parser.add_argument('--shuffle_val_output_plot_ims', type=int, default=0,
                        help="Switch to shuffle images for plotting, if 0, images are sorted")
    parser.add_argument('--save_json', type=int, default=1,
                        help="Switch to save a json in test")
    parser.add_argument('--val_prediction_df_refine_tot_root_part', type=str,
                        default='test_predictions_refine',
                        help="Refined results in dataframe format")
    parser.add_argument('--only_pre', type=int,
                        default=1,
                        help="Sice TIF")
    args = parser.parse_args()
    args = update_args(args)
    test_cmd_tot = prep(args)
    # import pdb
    # pdb.set_trace()
    execute(args, test_cmd_tot,only_pre=args.only_pre)

if __name__ == "__main__":

    print("\n\n\nPermit me to introduce myself...\n")
    main()

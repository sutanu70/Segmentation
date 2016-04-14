clear all; close all;

DATASETS = '' % SPECIFY YOUR PATH

bbox_orig_folder = fullfile(DATASETS, 'VOC_aug/dataset/SegClassBboxAug_RGB');
bbox_save_folder = fullfile(DATASETS, 'VOC_aug/dataset/SegClassBboxAug_1D');

bboxcrf_orig_folder = fullfile(DATASETS, 'VOC_aug/dataset/SegClassBboxErode20CRFAug_RGB');
bboxcrf_save_folder = fullfile(DATASETS, 'VOC_aug/dataset/SegClassBboxErode20CRFAug_1D');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bbox_imgs_dir = dir(fullfile(bbox_orig_folder, '*.png'));
bboxcrf_imgs_dir = dir(fullfile(bboxcrf_orig_folder, '*.png'));

if ~exist(bbox_save_folder, 'dir')
  mkdir(bbox_save_folder)
end

if ~exist(bboxcrf_save_folder, 'dir')
  mkdir(bboxcrf_save_folder)
end

for i = 1 : numel(bbox_imgs_dir)
  fprintf(1, 'processing %d (%d) ...\n', i, numel(bbox_imgs_dir));
  img = imread(fullfile(bbox_orig_folder, bbox_imgs_dir(i).name));
  img2 = imread(fullfile(bboxcrf_orig_folder, bboxcrf_imgs_dir(i).name));

  imwrite(img, fullfile(bbox_save_folder, bbox_imgs_dir(i).name));
  imwrite(img2, fullfile(bboxcrf_save_folder, bboxcrf_imgs_dir(i).name));
end

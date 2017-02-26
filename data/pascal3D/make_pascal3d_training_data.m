pascal_root_dir = '/media/Scratchspace/Pascal3D+/PASCAL3D+_release1.1';

voc2012_options = struct('flip',1,'aug_n',8,'jitter_IoU',0.6,'difficult',1,'truncated',1,'occluded',1);
prepare_voc12_imgs(pascal_root_dir, 'train', 'pascal3d_voc2012_train', 'car', voc2012_options)

% imagenet_options = struct('flip',1,'aug_n',3,'jitter_IoU',0.6,'difficult',1,'truncated',1,'occluded',1);
% prepare_imagenet_imgs(pascal_root_dir, 'train', 'pascal3d_imagenet_train', 'car', imagenet_options);
% 
% imagenet_options = struct('flip',1,'aug_n',3,'jitter_IoU',0.6,'difficult',1,'truncated',1,'occluded',1);
% prepare_imagenet_imgs(pascal_root_dir, 'val', 'pascal3d_imagenet_val', 'car', imagenet_options)
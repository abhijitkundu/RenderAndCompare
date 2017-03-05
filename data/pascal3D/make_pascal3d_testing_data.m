pascal_root_dir = '/media/Scratchspace/Pascal3D+/PASCAL3D+_release1.1';

% voc2012_val_options = struct('flip',1,'aug_n',8,'jitter_IoU',0.6,'difficult',1,'truncated',1,'occluded',1, 'min_size', 1);
% prepare_voc12_imgs(pascal_root_dir, 'val', 'pascal3d_voc2012_val', 'car', voc2012_val_options);

voc2012_val_easy_options = struct('flip',0,'aug_n',1,'jitter_IoU',1,'difficult',0,'truncated',0,'occluded',0, 'min_size', 1);
prepare_voc12_imgs(pascal_root_dir, 'val', 'pascal3d_voc2012_val_easy', 'car', voc2012_val_easy_options);
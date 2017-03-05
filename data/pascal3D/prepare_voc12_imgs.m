function prepare_voc12_imgs(pascal_root_dir, img_set, output_img_dir, cls_name, opts)
% PREPARE_VOC12_IMGS
%   prepare voc12 images (cropped from ground truth bboxes with jittering)
% input:
%   pascal_root_dir: pascal3d+ root dir (See http://cvgl.stanford.edu/projects/pascal3d.html)
%   img_set: 'train' or 'val'
%   output_img_dir: output folder names, final structure is <output_img_dir>/<category>/<imgs>
%   cls_names: cell array of category names
%   opts: matlab struct with flip, aug_n, jitter_IoU, difficult, truncated,
%   occluded, and min_size fields
%       if flip is 1, images will be flipped.
%       if ang_n>1, images will be augmented by jittering bbox. jitter_IoU==1 means normal crop
%       difficult, truncated, occluded \in {0,1}, where 0 indicated that we do not
%           want images with that property (e.g. 0,0,0 means we want easy images only)
% output:
%   cropped images according to ground-truth bounding boxes (with jittering) and image filelists
%

% paths
annotation_path = fullfile(pascal_root_dir, 'Annotations');
image_path = fullfile(pascal_root_dir, 'Images');
addpath(fullfile(pascal_root_dir, 'VDPM'));
addpath(fullfile(pascal_root_dir, 'PASCAL/VOCdevkit/VOCcode'));

% read ids of train/val set images
VOCinit;
ids = textread(sprintf(VOCopts.imgsetpath, img_set), '%s');
M = numel(ids);

if ~exist(output_img_dir, 'dir')
    mkdir(output_img_dir);
end


dirname = fullfile(output_img_dir, cls_name);
if ~exist(dirname, 'dir')
    mkdir(dirname)
end

labelfile = fopen(fullfile(output_img_dir, sprintf('%s.txt',cls_name)),'w');
CADPath = fullfile(pascal_root_dir, 'CAD', sprintf('%s.mat', cls_name));
cad = load(CADPath);
cad = cad.(cls_name);

for i = 1:M
    anno_filename = fullfile(annotation_path, sprintf('%s_pascal/%s.mat', cls_name, ids{i}));
    if ~exist(anno_filename,'file')
        continue;
    end
    anno = load(anno_filename);
    objects = anno.record.objects;
    
    for k = 1:length(objects)
        obj = objects(k);
        
        if ~isempty(obj.viewpoint) && strcmp(obj.class, cls_name)
            % skip of viewpoint annotation is not continuous
            if obj.viewpoint.distance == 0
                %                     fprintf('skip %s..\n.', ids{i});
                continue;
            end
            % write view annotation
            azimuth = mod(obj.viewpoint.azimuth, 360);
            elevation = mod(obj.viewpoint.elevation, 360);
            tilt = mod(obj.viewpoint.theta, 360);
            distance = obj.viewpoint.distance / 2.86; % 2.86 was determined empiracally
            principal_offset = [obj.viewpoint.px obj.viewpoint.py]; % Center them at origin
            
            
            truncated = obj.truncated;
            occluded = obj.occluded;
            difficult = obj.difficult;
            % skip un-annotated image
            if azimuth == 0 && elevation == 0 && tilt == 0
                fprintf('skip weird viewpoints %s..\n.', ids{i});
                continue;
            end
            % skip unwanted image
            if (difficult==1 && opts.difficult==0) || (truncated==1 && opts.truncated==0) || (occluded==1 && opts.occluded==0)
                fprintf('fliter skip %s...\n', ids{i});
                continue;
            end
            
            img_filename = fullfile(image_path, sprintf('%s_pascal/%s.jpg', cls_name, ids{i}));
            im = imread(img_filename);
            
            [im_height, im_width, ~] = size(im);
            
            bbx_extremes = obj.bbox;
            bbx_extremes(1:2) = max(bbx_extremes(1:2), 1);
            bbx_extremes(3:4) = min(bbx_extremes(3:4), [im_width, im_height]);
            assert(bbx_extremes(3)>=bbx_extremes(1));
            assert(bbx_extremes(4)>=bbx_extremes(2));

            gt_crop_bbx = [ bbx_extremes(1), bbx_extremes(2), bbx_extremes(3)-bbx_extremes(1), bbx_extremes(4)-bbx_extremes(2)];
            
             % too small
            w = gt_crop_bbx(3) + 1;
            h = gt_crop_bbx(4) + 1;
            if w < opts.min_size || h < opts.min_size
                fprintf('Tiny Image skip %s...\n', ids{i});
                continue;
            end
            
            
            vertex = cad(obj.cad_index).vertices;
            x2d = project_3d(vertex, obj);
            assert(~isempty(x2d), 'x2d should not be empty');
            
            cad_min_max = [min(x2d(:,1)),min(x2d(:,2)),max(x2d(:,1)),max(x2d(:,2))];
            amodal_bbx = [cad_min_max(1), cad_min_max(2), cad_min_max(3)-cad_min_max(1), cad_min_max(4)-cad_min_max(2)];
            
            for aug_i = 1:opts.aug_n
                if aug_i == 1
                    cropped_im = imcrop(im, gt_crop_bbx);
                    crop_bbx = gt_crop_bbx;
                else
                    [cropped_im, ~, crop_bbx] = jitter_imcrop(im, gt_crop_bbx, opts.jitter_IoU);
                end
                cropped_im_filename = sprintf('%s_%s_%s_%s.jpg', cls_name, ids{i}, num2str(k), num2str(aug_i));
                imwrite(cropped_im, fullfile(output_img_dir, cls_name, cropped_im_filename));

                adj_amodal_bbx = amodal_bbx;
                adj_amodal_bbx(1:2) = adj_amodal_bbx(1:2) - principal_offset;

                adj_crop_bbx = crop_bbx;
                adj_crop_bbx(3:4) = adj_crop_bbx(3:4) +  [1, 1];
                adj_crop_bbx(1:2) = adj_crop_bbx(1:2) - principal_offset - [0.5, 0.5];
                
                [cropped_im_height, cropped_im_width, ~] = size(cropped_im);
                assert(cropped_im_width==adj_crop_bbx(3), '%d != %f',cropped_im_width, adj_crop_bbx(3));
                assert(cropped_im_height==adj_crop_bbx(4), '%d != %f',cropped_im_height, adj_crop_bbx(3));

                assert(azimuth < 360);
                assert(elevation < 360);
                assert(tilt < 360);

                
                fprintf(labelfile, '%s %f %f %f %f ', cropped_im_filename, azimuth, elevation, tilt, distance);
                fprintf(labelfile, '%f %f %f %f ', adj_amodal_bbx(1), adj_amodal_bbx(2), adj_amodal_bbx(3), adj_amodal_bbx(4));
                fprintf(labelfile, '%f %f %f %f\n', adj_crop_bbx(1), adj_crop_bbx(2), adj_crop_bbx(3), adj_crop_bbx(4));
                
                
                if opts.flip
                    cropped_im_flip = fliplr(cropped_im); % flip the image horizontally
                    cropped_im_flip_filename = sprintf('%s_%s_%s_%s_%s.jpg', cls_name, ids{i}, num2str(k), num2str(aug_i), 'flip');
                    imwrite(cropped_im_flip, fullfile(output_img_dir, cls_name ,cropped_im_flip_filename));
                    
                    adj_amodal_bbx(1) = - (adj_amodal_bbx(1)  + adj_amodal_bbx(3));
                    adj_crop_bbx(1) =  - (adj_crop_bbx(1)  + adj_crop_bbx(3));

                    flipped_azimuth = mod(360.0-azimuth,360);
                    flipped_tilt = mod(-1.0*tilt,360);

                    assert(flipped_azimuth < 360);
                    assert(flipped_tilt < 360);
                    
                    fprintf(labelfile, '%s %f %f %f %f ', cropped_im_flip_filename, flipped_azimuth, elevation, flipped_tilt, distance);
                    fprintf(labelfile, '%f %f %f %f ', adj_amodal_bbx(1), adj_amodal_bbx(2), adj_amodal_bbx(3), adj_amodal_bbx(4));
                    fprintf(labelfile, '%f %f %f %f\n', adj_crop_bbx(1), adj_crop_bbx(2), adj_crop_bbx(3), adj_crop_bbx(4));
                end
            end
        end
    end
end
fclose(labelfile);

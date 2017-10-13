clear;
pascal_root_dir = '/media/Scratchspace/Pascal3D+/PASCAL3D+_release1.1';
dataset = 'pascal';
class = 'car';

annotationPath = fullfile(pascal_root_dir, 'AnnotationsFixed', sprintf('%s_%s/', class, dataset));
imagePath = fullfile(pascal_root_dir, 'Images', sprintf('%s_%s/', class, dataset));

% load cad model
CADPath = fullfile(pascal_root_dir, 'CAD', sprintf('%s.mat', class));
object = load(CADPath);
cad = object.(class);

listing = dir(annotationPath);
recordSet = {listing.name};

f = figure;
for recordElement = recordSet
    [~, ~, ext] = fileparts(recordElement{1});
    if ~strcmp(ext, '.mat')
        continue;
    end
    
%     if ~strcmp(recordElement{1}, 'n03770679_16071.mat')
%         continue;
%     end
    
    anno_file = [annotationPath recordElement{1}];
    record = load(anno_file,'record');
    record = record.record;
    
    num_of_objects = length(record.objects(:));
    fprintf('%s\n', recordElement{1})
    im = imread([imagePath, record.filename]);
    imshow(im);
    hold on;
    for ob_id = 1:num_of_objects
        object = record.objects(ob_id);
        if ~strcmp(object.class, class)
            continue;
        end
        
        vbbx = object.bbox;
        
        if object.truncated == 0 && object.occluded == 0 && object.difficult == 0
            record.objects(ob_id).abbx = vbbx;
            fprintf('\t%d easy abbx added\n', ob_id);
            continue;
        end
        
        if isfield(object, 'abbx')
            if length(object.abbx) == 4
                continue;
            end
        end
        
        if object.viewpoint.distance == 0
            fprintf('No continuous viewpoint\n');
            continue;
        end
        
        
        
        fprintf('\t%d truncated=%d, occluded=%d, difficult=%d\n', ob_id, object.truncated, object.occluded, object.difficult);
        
        w = vbbx(3)-vbbx(1)+1;
        h = vbbx(4)-vbbx(2)+1;
        bbox_draw = [vbbx(1),vbbx(2), vbbx(3)-vbbx(1), vbbx(4)-vbbx(2)];
        rectangle('Position', bbox_draw, 'EdgeColor', 'g');
        
        vertex = cad(object.cad_index).vertices;
        face = cad(object.cad_index).faces;
        x2d = project_3d(vertex, object);
        patch('vertices', x2d, 'faces', face, ...
            'FaceColor', 'blue', 'FaceAlpha', 0.08, 'EdgeColor', 'none');
        bbox_cad = [min(x2d(:,1)),min(x2d(:,2)),max(x2d(:,1)),max(x2d(:,2))];
        bbox_cad_draw = [bbox_cad(1), bbox_cad(2), bbox_cad(3)-bbox_cad(1), bbox_cad(4)-bbox_cad(2)];
        rectangle('Position', bbox_cad_draw, 'EdgeColor', 'r');
        
        ax = gca;
        ax.Clipping = 'off';
        
        pause;
        k = get(f,'CurrentCharacter');
        if k== 'v'
            record.objects(ob_id).abbx = vbbx;
            fprintf('\t%d abbx set to vbbx\n', ob_id);
        elseif k == 'c'
            record.objects(ob_id).abbx = bbox_cad;
            fprintf('\t%d abbx set to cad_bbx\n', ob_id);  
        end
    end
    hold off;
    fprintf('Saving record at %s\n', anno_file);
    save(anno_file, 'record');
end
    
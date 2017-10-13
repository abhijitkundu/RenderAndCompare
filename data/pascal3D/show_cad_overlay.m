% show the overlay of the CAD model to the image
% cls: class name, eg., 'car', 'bicycle', etc.
% example: show_cad_overlay('car');
function show_cad_overlay(cls, dataset)

if nargin < 2
    dataset = 'imagenet';
end

pascal_root_dir = '/media/Scratchspace/Pascal3D+/PASCAL3D+_release1.1';


% projection test

annotationPath = fullfile(pascal_root_dir, 'AnnotationsFixed', sprintf('%s_%s/', cls, dataset));
imagePath = fullfile(pascal_root_dir, 'Images', sprintf('%s_%s/', cls, dataset));

% load cad model
CADPath = fullfile(pascal_root_dir, 'CAD', sprintf('%s.mat', cls));
object = load(CADPath);
cad = object.(cls);

listing = dir(annotationPath);
recordSet = {listing.name};

f = figure;
for recordElement = recordSet
    if ~strcmp(recordElement{1}, 'n03770679_4602.mat')
        continue;
    end
    
    [~, ~, ext] = fileparts(recordElement{1});
    if ~strcmp(ext, '.mat')
        continue;
    end
    record = load([annotationPath recordElement{1}],'record');
    record = record.record;
    
    im = imread([imagePath, record.filename]);
    imshow(im);
    
    carIdxSet = find(ismember({record.objects(:).class}, cls));
    
    hold on;
    for carIdx = carIdxSet
        object = record.objects(carIdx);
        
        box = object.bbox;
        w = box(3)-box(1)+1;
        h = box(4)-box(2)+1;
        bbox_draw = [box(1),box(2), box(3)-box(1), box(4)-box(2)];
        rectangle('Position', bbox_draw, 'EdgeColor', 'g');
        
        if object.viewpoint.distance == 0
            fprintf('No continuous viewpoint\n');
            continue;
        end
        
        
        object.viewpoint.distance = 5.5;
        record.objects(carIdx).viewpoint.distance = 5.5;
        
        vertex = cad(object.cad_index).vertices;
        face = cad(object.cad_index).faces;
        x2d = project_3d(vertex, object);
        patch('vertices', x2d, 'faces', face, ...
            'FaceColor', 'blue', 'FaceAlpha', 0.05, 'EdgeColor', 'none');
        
        
        % Find bbox for the CAD model
        bbox_cad = [min(x2d(:,1)),min(x2d(:,2)),max(x2d(:,1)),max(x2d(:,2))];
        bbox_cad_draw = [bbox_cad(1), bbox_cad(2), bbox_cad(3)-bbox_cad(1), bbox_cad(4)-bbox_cad(2)];
        rectangle('Position', bbox_cad_draw, 'EdgeColor', 'r');
        
        fprintf('azimuth= %f, elevation = %f, tilt= %f,  distance= %f\n', object.viewpoint.azimuth, object.viewpoint.elevation, object.viewpoint.theta, object.viewpoint.distance )
        fprintf('cad_index=%d focal= %f, principal = [%f,%f] viewport= %f\n', object.cad_index, object.viewpoint.focal, object.viewpoint.px, object.viewpoint.py, object.viewpoint.viewport )
        plot(object.viewpoint.px, object.viewpoint.py,'r+', 'MarkerSize', 50);
    end
    fprintf('------------------------------------------------------------------------------------------------\n')
    axis off;
    hold off;
    pause;
    clf;
%     save(recordElement{1}, 'record');
end
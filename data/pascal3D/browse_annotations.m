pascal_root_dir = '/media/Scratchspace/Pascal3D+/PASCAL3D+_release1.1';
dataset = 'imagenet';
cls = 'car';

annotationPath = fullfile(pascal_root_dir, 'Annotations', sprintf('%s_%s/', cls, dataset));
imagePath = fullfile(pascal_root_dir, 'Images', sprintf('%s_%s/', cls, dataset));

listing = dir(annotationPath);
recordSet = {listing.name};

for recordElement = recordSet
    [~, ~, ext] = fileparts(recordElement{1});
    if ~strcmp(ext, '.mat')
        continue;
    end
    record = load([annotationPath recordElement{1}],'record');
    record = record.record;
    
    num_of_objects = length(record.objects(:));
    fprintf('%s\n', recordElement{1})
    im = imread([imagePath, record.filename]);
    imshow(im);
    hold on;
    for ob_id = 1:num_of_objects
        object = record.objects(ob_id);
        box = object.bbox;
        w = box(3)-box(1)+1;
        h = box(4)-box(2)+1;
        bbox_draw = [box(1),box(2), box(3)-box(1), box(4)-box(2)];
        rectangle('Position', bbox_draw, 'EdgeColor', 'g');
    end
    hold off;
    pause;
end
    
clear;

pascal_root_dir = '/media/Scratchspace/Pascal3D+/PASCAL3D+_release1.1';

cls = 'car';

% load cad model
CADPath = fullfile(pascal_root_dir, 'CAD', sprintf('%s.mat', cls));
object = load(CADPath);
cad = object.(cls);


image=zeros(540,960,3); %initialize


figure(1), imshow(image)

vertex = cad(2).vertices;
face = cad(2).faces;


viewpoint = struct('azimuth',90,'elevation',0,'tilt',0,'distance',6,'px',480.0,'py',270.0);

x2d = render3Dpoints(vertex, viewpoint);
patch('vertices', x2d, 'faces', face,'FaceColor', 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

bbx_min = min(x2d);
bbx_max = max(x2d);

bbox_w = bbx_max(1)-bbx_min(1) + 1;
bbox_h = bbx_max(2)-bbx_min(2) + 1;


bbox_draw = [bbx_min(1),bbx_min(2), bbx_max(1)-bbx_min(1), bbx_max(2)-bbx_min(2)];
rectangle('Position', bbox_draw, 'EdgeColor', 'g');


fprintf('bbox_w= %f, bbox_h = %f, aspect = %f\n', bbox_w, bbox_h, bbox_w/bbox_h );
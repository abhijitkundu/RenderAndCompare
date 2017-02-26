function [ points2d ] = render3Dpoints( points3d, viewpoint )
%RENDER3DPOINTS Summary of this function goes here
%   Detailed explanation goes here

% project the 3D points
a = viewpoint.azimuth*pi/180;
e = viewpoint.elevation*pi/180;
d = viewpoint.distance;
f = 1.0;
theta = viewpoint.tilt*pi/180;
principal = [viewpoint.px viewpoint.py];
viewport =3000.000000;

% camera center
C = zeros(3,1);
C(1) = d*cos(e)*sin(a);
C(2) = -d*cos(e)*cos(a);
C(3) = d*sin(e);

% Rotate coordinate system by theta is equal to rotating the model by -theta.
a = -a;
e = -(pi/2-e);

% rotation matrix
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
R = Rx*Rz;

% perspective project matrix
% however, we set the viewport to 3000, which makes the camera similar to
% an affine-camera. Exploring a real perspective camera can be a future work.
M = viewport;
P = [M*f 0 0; 0 M*f 0; 0 0 -1] * [R -R*C];

% project
points2d = P*[points3d ones(size(points3d,1), 1)]';
points2d(1,:) = points2d(1,:) ./ points2d(3,:);
points2d(2,:) = points2d(2,:) ./ points2d(3,:);
points2d = points2d(1:2,:);

% rotation matrix 2D
R2d = [cos(theta) -sin(theta); sin(theta) cos(theta)];
points2d = (R2d * points2d)';
% x = x';

% transform to image coordinates
points2d(:,2) = -1 * points2d(:,2);
points2d = points2d + repmat(principal, size(points2d,1), 1);

end


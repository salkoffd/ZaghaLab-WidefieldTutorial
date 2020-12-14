function [vidCropped, rectout] = vidcrop(video)
% [vidCropped, rectout] = vidcrop(video)
% Crops video according to user input. Based on imcrop.
%
% When you run the function, the first frame of the movie will be
% displayed. Drag a rectangle over the region you want cropped. Then right
% click and select "Crop Image".
%
% Outputs:
%   vidCropped is the cropped video
%   rectout is the rectangle position and height/width. See Matlab
%   documentation on imcrop for more info.
%
% vidCropped(:,:,1) *should* be equal to cropped image produced
% by Icropped = imcrop(video(:,:,1),rect).
%
% last updated 12/8/2020

% figure out appropriate color scale
myPercent = 99;
colorMax = prctile(reshape(video(:,:,1),[numel(video(:,:,1)),1]),myPercent);
% plot figure
h = figure;
imshow(video(:,:,1),[0 colorMax])
[~,rectout] = imcrop;

% coordinates
xmin = ceil(rectout(1));
xmax = floor(rectout(1)+rectout(3));
ymin = ceil(rectout(2));
ymax = floor(rectout(2)+rectout(4));

% crop video
vidCropped = video(ymin:ymax, xmin:xmax, :);

close(h)
disp('video cropped')

end


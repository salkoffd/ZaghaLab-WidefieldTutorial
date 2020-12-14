function [video,timestamps] = LoadVideoAVI(filename,maxFrames,bCrop)
% [video,timestamps] = LoadVideoAVI(filename,maxFrames,bCrop)
% Loads AVI video into MATLAB variable
% Only supports black & white videos. Loads video into uint8 array.
% last updated 12/10/2020
%
%   inputs:
%     filename: path to avi file e.g. '201118_0.avi'
%     maxFrames: maximum number of frames to be loaded. Setting to [] will
%                load all frames in the video
%     bCrop: if true, you'll be prompted to crop the video before loading
%
%   outputs:
%     video: height-by-width-by-frame uint8 array of loaded video
%     timestamps: list of frame timestamps


% create video object to read AVI frames
vidObj = VideoReader(filename);
% calculate number of frames
% internet says this is an estimation (some movies don't have a constant framerate)
nFramesEstimate = round(vidObj.Duration*vidObj.FrameRate);

% how many frames to load?
if nargin==1 || isempty(maxFrames)
    maxFrames = nFramesEstimate;
end

% crop video
if nargin<3 || bCrop == false
    ymin = 1;
    ymax = vidObj.height;
    xmin = 1;
    xmax = vidObj.width;
elseif bCrop == true
    % read one frame
    testframe = readFrame(vidObj);
    h = figure;
    imshow(testframe)
    disp('Select region to crop by dragging a rectangle. Right click and select "Crop Image"')
    % get user input to crop
    [~,rectout] = imcrop;
    close(h)
    % coordinates
    xmin = ceil(rectout(1));
    xmax = floor(rectout(1)+rectout(3));
    ymin = ceil(rectout(2));
    ymax = floor(rectout(2)+rectout(4));
end

% preallocate video matrix
% video = zeros(vidObj.height,vidObj.width,nFramesEstimate,'uint8');
video = zeros(ymax-ymin+1, xmax-xmin+1, maxFrames, 'uint8');
timestamps = zeros(1,maxFrames);

% reset time to 0
vidObj = VideoReader(filename,'CurrentTime',0);
disp('Loading AVI file...')
iFrame = 0;
while iFrame < maxFrames    %hasFrame(vidObj)
    iFrame = iFrame + 1;
    % read the frame
    img = readFrame(vidObj);
    % store img into video matrix
    % only taking red channel (black & white)
    video(:,:,iFrame) = img(ymin:ymax, xmin:xmax, 1);
    % store timestamp
    timestamps(iFrame) = vidObj.CurrentTime;
end
disp('Done')
end


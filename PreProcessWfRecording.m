function [ videoWF_dff ] = PreProcessWfRecording(videoWF,videoWF_TS,hwMean_sec,bSmooth)
% [ videoWF_dff ] = PreProcessWfRecording(videoWF,videoWF_TS,hwMean_sec,bSmooth)
% Pre-processes widefield video of neural activity.
% Finds dF/F using local mean fluorescence as baseline.
%
% Output: 
%      videoWF_dff: neural activity readout (fluorescence normalized to local mean)
%
% Inputs: 
%      videoWF: raw fluorescence movie (height x width x frames)
%      videoWF_TS: array of video timestamps 
%      hwMean_sec: fluorescence will be normalized by the
%          mean fluorescence around each point. Local means have a
%          half-width of hwMean_sec seconds. Default = 200s. Robust to
%          non-continuous (i.e. trial-by-trial) recordings. Local means
%          at the endpoints will only include data from one side.
%      bSmooth: if true, applies gaussian smoothing. Default = false.
%          Useful for removing salt-and-pepper noise from camera.
%
% Written by David Salkoff 12/10/20, dovsalkoff@gmail.com

% Local mean half-width
if ~exist('localMean_hw','var') || isempty(hwMean_sec)
    hwMean_sec = 200; %half-width of normalizing window in seconds
end

%apply spatial smoothing?
if ~exist('bSmooth','var') || isempty(bSmooth)
    bSmooth = false;
end
        
%in case where timestamps were not fully recorded, truncate video
if length(videoWF_TS)<size(videoWF,3)
    videoWF(:,:,length(videoWF_TS)+1:end) = [];
    warning('Fewer timestamps than video frames. Truncating video.');
end
%in case where video stopped recording, truncate timestamps
if length(videoWF_TS)>size(videoWF,3)
    videoWF_TS(size(videoWF,3)+1:end) = [];
    warning('Fewer video frames than timestamps. Truncating timestamps');
end

disp('Caculating local mean dF/F')

% Calculate local mean image and normalize every x seconds in movie .
% Local mean normalization
localMean_jump = 0.5; %compute mean every x seconds (default 2)
%list of times where local mean is computed (centers)
lm_times = videoWF_TS(1):localMean_jump:videoWF_TS(end);
lm_times = horzcat(lm_times,lm_times(end)+localMean_jump); %apend one more time for good measure
%save local means in big matrix
localMeans = zeros(size(videoWF,1),size(videoWF,2),length(lm_times),'single');
% for % complete display
thresh = .1; 
for iMean = 1:length(lm_times)
    tFrameCenter = lm_times(iMean);
    %find first frame within local mean time range
    frameFirst = find(videoWF_TS - (tFrameCenter-hwMean_sec) > 0,1,'first');
    %find last frame within local mean time range
    frameLast = find(videoWF_TS - (tFrameCenter+hwMean_sec) < 0,1,'last');
    %compute mean and store
    localMeans(:,:,iMean) = single(mean(videoWF(:,:,frameFirst:frameLast),3));
    if iMean/length(lm_times) >= thresh
        disp([num2str(thresh*100) '%'])
        thresh = thresh + .1;
    end
end

%subtract and normalize wide-field movie by stored local means
videoWF_dff = single(videoWF); %convert to single for dF/F measurement
for iFrame = 1:size(videoWF,3)
    %find local mean closest in time to this frame
    [~,lm_closest] = min(abs(videoWF_TS(iFrame)-lm_times));
    videoWF_dff(:,:,iFrame) = (videoWF(:,:,iFrame)-localMeans(:,:,lm_closest))./localMeans(:,:,lm_closest); %dF/F
end


if bSmooth == true
    disp('Applying gaussian smoothing');
    %gaussian low-pass filter parameters
    hsize = [20,20];
    sigma = 1; % standard deviation for gaussian filter
    filterLP = fspecial('gaussian', hsize, sigma);
    % for % complete display
    thresh = .1; 
    %apply gaussian filter to each frame of movie
    for iFrame = 1:size(videoWF_dff,3)
        videoWF_dff(:,:,iFrame) = imfilter(videoWF_dff(:,:,iFrame),filterLP);
        % for % complete display
        if iFrame/size(videoWF_dff,3) >= thresh
            disp([num2str(thresh*100) '%'])
            thresh = thresh + .1;
        end
    end
end

% %heads up - this is old code
% %apply temporal high-pass filter to movie
% disp('High-pass filtering movie')
% %acquisition rate
% %design hp filter
% fc= (1/60);% cut off frequency
% fn=(1/ar)/2; %nyquivst frequency = sample frequency/2;
% order = 4; %4th order filter, high pass
% [b, a]=butter(order,(fc/fn),'high');
% %fvtool(b,a);
% %zero-phase filtering to preserve features in time
% for i=1:size(videoWF_dff,1)
%     for ii=1:size(videoWF_dff,2)
%         signal = squeeze(videoWF_dff(i,ii,:));
%         videoWF_dff(i,ii,:) = filtfilt(b,a,double(signal-mean(signal)));
%     end
% end

% %apply temporal low-pass filter to movie
% disp('Low-pass filtering movie')
% %design lp filter
% fc= 5;% cut off frequency
% fn=(1/ar)/2; %nyquivst frequency = sample frequency/2;
% order = 4; %4th order filter, low pass
% [b, a]=butter(order,(fc/fn),'low');
% %fvtool(b,a);
% %zero-phase filtering to preserve features in time
% for i=1:size(videoWF_dff,1)
%     for ii=1:size(videoWF_dff,2)
%         signal = squeeze(videoWF_dff(i,ii,:));
%         videoWF_dff(i,ii,:) = filtfilt(b,a,double(signal-mean(signal)));
%     end
% end


end
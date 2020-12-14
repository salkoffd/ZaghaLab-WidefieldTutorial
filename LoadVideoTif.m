function [videoWF] = LoadVideoTif(filePath)
% [videoWF] = LoadVideoTif(filePath) Import tif file into MATLAB workspace
% videoWF is the widefield video [height x width x timeframe]
% example file path. Include file name.
% filePath = 'C:\Users\Dov\Desktop\Zagha Lab\Imaging Selective Detection Task\GSS+03\190204_SPOT.tif';
% Last updated 12/10/20

% numerical type of videoWF. 
numType = 'single';

disp('Loading tif file...')
file_info = imfinfo(filePath);
nFramesWF = length(file_info);
videoWF_height = file_info(1).Height;
videoWF_width = file_info(1).Width;
videoWF = zeros(videoWF_height,videoWF_width,nFramesWF,numType);
thresh = .1; % for % complete display
for i=1:nFramesWF
    videoWF(:,:,i) = imread(filePath,i);
    % for % complete display
    if i/nFramesWF >= thresh
        disp([num2str(thresh*100) '%'])
        thresh = thresh + .1;
    end
end
disp('done')

end


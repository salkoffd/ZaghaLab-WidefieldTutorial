function sumEnergy = MotionEnergy(video,mask)
% sumEnergy = MotionEnergy(video,mask)
% Computes the motion energy in a video
%
%   Motion energy is defined as the absolute temporal derivative of each
%   pixel, summed across the region of interest. If no mask is provided,
%   the function will calculate motion energy of the entire video.
%
% inputs:
%   video: a height by width by time frame matrix
%   mask: (optional) a logical height by width matrix with true values
%   for the area of interest
%
% outputs:
%   sumEnergy: the motion energy of the region of interest at each time
%   frame.
%
% last updated 12/10/20

% set this to true to normalize and de-mean the energy signal 
bZtransform = true;

dimMeasurements = size(video);

% make mask if none provided
if nargin ==1
    mask = true(dimMeasurements(1:2));
end

% calcualte motion energy for each pixel
videoMasked = reshape(video(repmat(mask,[1,1,dimMeasurements(3)])), nnz(mask), []);
motionEnergy = abs(diff(videoMasked,1,2));
% sum pixels
sumEnergy = squeeze(sum(motionEnergy));

% compute z-score
if bZtransform==true
    sumEnergy = (sumEnergy - mean(sumEnergy)) ./ std(sumEnergy);
end

end


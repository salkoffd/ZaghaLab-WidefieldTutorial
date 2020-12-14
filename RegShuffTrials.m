function [regFull_shuff] = RegShuffTrials(regFull, cols, framesPerTrial)
% [regFull_shuff] = RegShuffTrials(regFull, cols, framesPerTrial)
% Shuffle trials in the regressor matrix
%
%   Rows of the regressor matrix will be shuffled so that rows
%   corresponding to the same trial stay together. Only shuffle rows of
%   columns specified by cols. The rows of each column are shuffled in the
%   same order. Therefore you should call this function for each variable
%   you want to shuffle.
%
% Inputs
%    regFull = Full regression matrix
%    cols = logical matrix with true for each column in regFull to be shuffled
%    framesPerTrial = you guessed it, frames per trial
%
% Outputs
%    reFull_shuff = shuffled regression matrix
% 
% Written by David Salkoff, 12/10/20, dovsalkoff@gmail.com

nTrials = round(size(regFull,1)/framesPerTrial);
newTrialOrder = randperm(nTrials);
regFull_shuff = regFull;
for iTrial = 1:nTrials
    % indices of new shuffled matrix
    idxStart = (iTrial-1)*framesPerTrial + 1;
    idxEnd = idxStart + framesPerTrial - 1;
    % indices of trial we're taking out of order
    idxStart_shuff = (newTrialOrder(iTrial)-1)*framesPerTrial + 1;
    idxEnd_shuff = idxStart_shuff + framesPerTrial - 1;
    % paste out of order trial into new regressor matrix
    regFull_shuff(idxStart:idxEnd,cols) = regFull(idxStart_shuff:idxEnd_shuff,cols);
end

end


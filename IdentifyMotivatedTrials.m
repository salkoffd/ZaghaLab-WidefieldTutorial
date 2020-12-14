function [bMotivatedTrials] = IdentifyMotivatedTrials(tLicks,tTrials, durNoLicksLimit)
% [bMotivatedTrials] = IdentifyMotivatedTrials(tLicks,tTrials, durNoLicksLimit)
%   Classify trials as motivated or non-motivated
%   Motivated epochs are defined as periods of sustained licking
%   (no further than durNoLicksLimit seconds apart).
%
% Output
%   bMotivatedTrials - boolean vector with true for each trial that occured
%    during a motivated epoch
%
% Inputs
%   tLicks - list of time (in sec) of licks
%   tTrials - list of time (in sec) of trials
%   durNoLicksLimit - if mouse stops licking for this many seconds, call this
%       epoch unmotivated. Default = 60s.
%
% Written by David Salkoff, 12/10/20, dovsalkoff@gmail.com

% Parameters for finding motivated epochs
if ~exist('durNoLicksLimit','var') || isempty(durNoLicksLimit)
    durNoLicksLimit = 60;
end
% don't analyze behavior until x seconds have passed and mouse is licking
durCutStart = 0;           %default = 0;
%motivated epochs can't be less than x seconds
blockDurMin = 30;       %default = 30 seconds

%Begin analysis to find motivated epochs based on licking
%session duration in seconds
durSession = max([tLicks,tTrials]);
%which lick should be considered the first to analyze?
firstLickInd = find(tLicks>=durCutStart,1,'first');
%find blocks where mouse was engaged.  Blocks are separated by at least
%durNoLicksLimit seconds
nBlocksEngaged = 1;  %initialize at 1
tBlockStart=[];
tBlockEnd = [];
tBlockStart(1) = tLicks(firstLickInd); %first block starts with first lick
for iLick = firstLickInd:length(tLicks)-1
    timeNextLick = tLicks(iLick+1) - tLicks(iLick);
    if timeNextLick > durNoLicksLimit
        tBlockEnd(nBlocksEngaged) = tLicks(iLick); %#ok<AGROW>
        nBlocksEngaged = nBlocksEngaged + 1;
        tBlockStart(nBlocksEngaged) = tLicks(iLick+1); %#ok<AGROW>
    end
end
%does last block end with a lick or is it cut short?
if durSession - tLicks(end) > durNoLicksLimit/2
    tBlockEnd(end+1) = tLicks(end); %ends with lick
else
    tBlockEnd(end+1) = durSession; %cut short
end
%find duration of each block
blockDurs = tBlockEnd - tBlockStart;
%delete blocks shorter than blockDurMin
deleteBlocks = blockDurs<blockDurMin;
tBlockStart(deleteBlocks)=[];
tBlockEnd(deleteBlocks)=[];
blockDurs = tBlockEnd - tBlockStart;
nBlocksEngaged = length(blockDurs);
%Is each stimulus during a motivational epoch or not?
bMotivatedTrials = false(1,length(tTrials)); %1 if motivated, 0 if not
for iBlock = 1:nBlocksEngaged
    engagedTrials = and(tTrials>tBlockStart(iBlock),tTrials<tBlockEnd(iBlock));
    bMotivatedTrials(engagedTrials) = true;
end

end


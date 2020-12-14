function [designMatrix] = DesignFromEventTimes(tEvents,tFrames,lagDur_pre,lagDur_post)
% [designMatrix] = DesignFromEventTimes(tEvents,tFrames,lagDur_pre,lagDur_post)
% Creates a design (regressor) matrix from a list of events. Creates lagged
% versions of regressor according to lagDur_pre and lagDur_post. Compatible
% with continuous or non-continuous recording. Lagged versions of regressor
% may be included even if the event occurred before/after the camera turned
% on/off.
%
% Inputs
%   tEvents: list of events that you want to convert to a regressor matrix
%   tFrames: list of camera frame time stamps
%   lagDur_pre: duration (in seconds) for lagged regressors to extend before the event
%   lagDur_post: duration (in seconds) for lagged regressors to extend after the event
%
% Output
%   designMatrix - nxf design/regressor matrix where n is number of camera
%   frames and f is (lagDur_pre+lagDur_post)/frameInterval + 1. This matrix
%   can subsequently be used for multiple regression.
%
% Written by David Salkoff, 12/10/20, dovsalkoff@gmail.com

plot_results = false;

% time between frames (robust to non-continuous recordings)
frameInterval = mode(round(diff(tFrames),3)); % nearest millisecond
% frames before/after event for lagged versions
lagFrames_pre = round(lagDur_pre / frameInterval);
lagFrames_post = round(lagDur_post / frameInterval);

% pre-allocate regression matrix
designMatrix = false(length(tFrames),lagFrames_pre + lagFrames_post + 1);

% find first event near first camera frame
first_recorded_event = find(tEvents >= (tFrames(1) - lagDur_pre - frameInterval/2),1);
% for each event, search for nearby frames and enter true
for i = first_recorded_event:length(tEvents)
    % search for frames occurring during event (+/- lag)
    framesOverlap = find(tFrames > tEvents(i) - lagDur_pre - frameInterval/2 & ...
        tFrames < tEvents(i)+ lagDur_post + frameInterval/2);
    if length(framesOverlap) > lagFrames_pre + lagFrames_post + 1
        disp('Warning: frame rate may be highly variable');
    end
    % if event occurred near camera frames (otherwise skip)
    if ~isempty(framesOverlap)
        % find 0 lag framesOverlap index of event
        if length(framesOverlap) == lagFrames_pre + lagFrames_post + 1 % dealing with infrequent cases where two lag0 frames are found
            lag0 = lagFrames_pre + 1;
        else
            lag0 = find(tFrames(framesOverlap) > tEvents(i) - frameInterval*.51 & ...
                tFrames(framesOverlap) < tEvents(i) + frameInterval*.51, 1, 'last'); % allow wiggle room for fluctuating camera interval (hence frameInterval*.51 instead of frameInterval*.5)
        end
        % if event occurred during the trial
        if ~isempty(lag0)
            % regEvent index of this 0-lagged event
            idx = sub2ind(size(designMatrix), framesOverlap(lag0), lagFrames_pre + 1);
            % construct index list for regEvent (this event, all lags)
            idxLags = framesOverlap - framesOverlap(lag0);
            idxLags = idxLags .* (length(tFrames) + 1);
            idxLags = idxLags + idx;
            designMatrix(idxLags) = true;
        % if event occurred after the trial (can still use lagged regressors)
        elseif tEvents(i) > tFrames(framesOverlap(end))
            % if the camera had kept recording, how many frames after was the event?
            framesAfter = round((tEvents(i) - tFrames(framesOverlap(end))) / frameInterval);
            % regEvent index of the last peri-event camera frame
            idx = sub2ind(size(designMatrix), framesOverlap(end), lagFrames_pre + 1);
            % construct index list for regEvent (this event, all lags)
            idxLags = framesOverlap - framesOverlap(end) - framesAfter;
            idxLags = idxLags .* (length(tFrames) + 1);
            idxLags = idxLags + idx + framesAfter;
            designMatrix(idxLags) = true;
        % if event occurred before the camera started for this trial (can still use lagged regressors)
        elseif tEvents(i) < tFrames(framesOverlap(1))
            % if the camera were recording earlier, how many frames before was the event?
            framesBefore = round((tFrames(framesOverlap(1)) - tEvents(i)) / frameInterval);
            % regEvent index of the first peri-event camera frame
            idx = sub2ind(size(designMatrix), framesOverlap(1), lagFrames_pre + 1);
            % construct index list for regEvent (this event, all lags)
            idxLags = framesOverlap - framesOverlap(1) + framesBefore;
            idxLags = idxLags .* (length(tFrames) + 1);
            idxLags = idxLags + idx - framesBefore;
            designMatrix(idxLags) = true;
        end
    end
end

if any(sum(designMatrix) < 1)
    disp('Warning: some regressors are empty')
    plot_results = true;
end

if plot_results == true
    figure
    plot(-lagFrames_pre:1:lagFrames_post,sum(designMatrix),'Linewidth',1.5)
    ylabel('Count')
    xlabel('Frames Since Event')
    ylim([0 round(max(sum(designMatrix))*1.1)]);
    xlim([-lagFrames_pre lagFrames_post]);
    title('Lagged Regressor Count')
end

end


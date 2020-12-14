%% Widefield Analysis Tutorial
% This script showcases the tools used for pre-processing widefield videos,
% projecting into low dimension, creating a design matrix, and running
% multiple regression on neural/behavioral data. Much of this code (e.g.
% ridge regression functions) comes from Musall, Kaufman et al., 2019.
% Code can be found at:
%       https://github.com/churchlandlab/ridgeModel
%
% Required functions: 
%   Image Processing Toolbox (for PlayMovie (implay), type ver into command window to see which toolboxes you have)
%   Bioinformatics Toolbox (for suptitle)
%   ridgeMML.m (from Churchland Lab, see repository URL above)
%   crossValModel.m (from Churchland Lab)
%   modelCorr.m (from Churchland Lab)
%   LoadVideoTif.m
%   LoadVideoAVI.m
%   PreProcessWfRecording.m
%   vidcrop.m
%   PlayMovie.m
%   DesignFromEventTimes.m
%   MotionEnergy.m
%   CatRegressors.m
%   MakeBetaMaps.m
%   PlotBetaMaps.m
%   IdentifyMotivatedTrials.m
%   RegShuffTrials.m
%
% % Source: "Single-trial neural dynamics are dominated by richly varied
% movements" by Simon Musall, Matthew T. Kaufman, Ashley L. Juavinett,
% Steven Gluf and Anne K. Churchland, Nature Neuroscience, 2019. Page 1677.
% https://www.nature.com/articles/s41593-019-0502-4
%
% Written by David Salkoff, 12/10/20, dovsalkoff@gmail.com
% 

%% Load widefield video

% folder where all data is stored
dataFolder = 'C:\Users\Dov\Desktop\Zagha Lab\Imaging Selective Detection Task';

mouse_to_analyze = 1;

% mouse and date of experiment
if mouse_to_analyze == 1
    subject_label = 'GSS+03';
    experiment_date = '190204';
elseif mouse_to_analyze == 2
    subject_label = 'KA_I11';
    experiment_date = '201117';
    % crop widefield video?
    % posterior section has flourescence that dominates variance
    flagCrop = questdlg('The posterior area of this recording dominates the fluorescence variance. Would you like to crop the widefield video?');
    if strcmp(flagCrop,'Yes')
        crop_widefield_video = true;
    else
        crop_widefield_video = false;
    end
end

% change folder
cd([dataFolder '\' subject_label '\' experiment_date])
% load task & behavioral data
load([experiment_date '_event_history.mat'])
load([experiment_date '_lick_history'])
load([experiment_date '_parameters'])
load([experiment_date '_camera_history'])
% load([experiment_date '_time_stamp']) % not needed?

% load widefield video
if exist('videoWF.mat','file')
    load('videoWF.mat')
    if exist('crop_widefield_video','var') && crop_widefield_video
        videoWF = vidcrop(videoWF);
    end
else
    % find largest .tif file and load
    listing = dir('**/*.tif');
    [~,idx] = max([listing.bytes]);
    videoWF = LoadVideoTif(listing(idx).name);
    % crop widefield video?
    if exist('crop_widefield_video','var') && crop_widefield_video
        videoWF = vidcrop(videoWF);
    end
    % pre-process widefield video
    videoWF = PreProcessWfRecording(videoWF,camera_history);
    % save('videoWF.mat','videoWF') % optional, but may want to change pre-processing settings
end
vHeight = size(videoWF,1);
vWidth = size(videoWF,2);

% Experiment configuration
% time between frames (robust to non-continuous recordings)
frameInterval = mode(round(diff(camera_history),3)); % nearest millisecond
% camera frequency
framePerSec = round(1/frameInterval);
% frames per trial
framesPerTrial = framePerSec * (parameters.Pre_Stimulus_Imaging +...
    parameters.Total_Lockout + parameters.Lick_Window);
% what frame does the stimulus occur on?
frameStim = round(parameters.Pre_Stimulus_Imaging / frameInterval) + 1;

% target and distractor id
idTargetLong = 1;
idTargetShort = 2;
idDistractorLong = 3;
idDistractorShort = 4;

% % Response Types
response = 11;
no_response = 22;
premature = 33;  % lick before 200 ms after stimulus
spont = 44;      % lick before stimulus

%% Project widefield video into lower dimension
% using principle component analysis

% reshape widefield video for dimensionality reduction
nFrames_to_use = size(videoWF,3); % if you want to use only part of videoWF
X = double(videoWF(:,:,1:nFrames_to_use));
X = reshape(X,[vHeight*vWidth,nFrames_to_use]);
% check we can reshape correctly
% PlayMovie(reshape(X,[vHeight,vWidth,nFrames_to_use]))

% Reduce dimensionality with MATLAB's PCA function
tic
[coeff,score,~,~,explained,mu] = pca(X'); % note X is transposed
toc

% how much variance should we explain?
need_to_explain = 98.977;
sum_explained = 0;
nPCs = 0;
while sum_explained < need_to_explain
    nPCs = nPCs + 1;
    sum_explained = sum_explained + explained(nPCs);
end
disp([num2str(nPCs) ' dimensions needed to explain ' num2str(need_to_explain) '% variance']);
% set number of PCs used to 200 - somewhat arbitrary
nPCs = 200;
disp(['Number of principle components to use = ' num2str(nPCs)]);

% plot cumulative variance explained by principle components
figure
plot(cumsum(explained),'Linewidth',2)
hold on
histogram('BinCounts', explained, 'BinEdges', 0:length(explained));
title('Cumulative Variance explained by top principle components')
xlim([0 20]);
ylim([0 100]);
xlabel('Principle Component')
ylabel('Percent Variance Explained')
hold off

% reduce dimensionality
% make coeff approximately equal to U and score approximately equal to Vc
% (sometimes scores/coeffs are flipped)
coeff = coeff(:,1:nPCs); % reduce dimensionality - take top nPCs components
score = score(:,1:nPCs)'; % reduce dimensionality & transpose

% %% Reduce dimensionality with MATLAB's svd function
% % for demonstrative purposes only
% % relationship between PCA and SVD
% % https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
% 
% % X = U * S * V
% tic
% [U,S,V] = svd(X,'econ');
% toc
% 
% % "temporal components"
% Vc = S * V';
% Vc = Vc(1:nPCs,:); % reduce dimensionality - take top 200 components
% % "spatial components"
% U = U(:,1:nPCs);   % ditto
% 
% % % Reconstruct the video - because we can
% % Xreconstruct = U*Vc;
% % PlayMovie(reshape(Xreconstruct,[vHeight,vWidth,nFrames_to_use]))
% 
% figure
% subplot(2,1,1)
% plot(Vc(1,:),'Linewidth',4); hold on; plot(score(1,:),'Linewidth',1.5)
% legend({"Adjusted temporal components Vc"; "PC score"})
% title('First component score from svd / pca analysis')
% ylabel('Score')
% xlabel('Frame')
% subplot(2,1,2)
% plot(Vc(2,:),'Linewidth',1.5); hold on; plot(score(2,:),'Linewidth',1.5)
% legend({"Adjusted temporal components Vc"; "PC score"})
% title('Second component score from svd / pca analysis')
% ylabel('Score')
% xlabel('Frame')
% 
% % clear vars so we don't accidentally use them later
% clear U
% clear Vc

%% Plot the loadings of first few PCs

% how many components to plot?
plotPCs = 6;
% number of rows and columns for subplot
figRows = floor(plotPCs^.5);
figCols = ceil(plotPCs/figRows);
% plot loadings of each PC
figure
try
    suptitle({'Loadings of top Principle Components', ''})
catch
    warning('Cannot display title: Bioinformatics Toolbox missing')
end
for iPlot = 1:plotPCs
    img = coeff(:,iPlot);
    img = reshape(img,[vHeight, vWidth]);
    subplot(figRows,figCols,iPlot)
    imshow(img, [-.01 .02], 'Colormap', jet)
    title(['Principle Component ' num2str(iPlot)])
end

%% Reconstruct the video using first n PCs

Xreconstruct = score(1:nPCs,:)'*coeff(:,1:nPCs)';
Xreconstruct = Xreconstruct + mu;
Xreconstruct = Xreconstruct';
Xreconstruct = reshape(Xreconstruct,[vHeight,vWidth,nFrames_to_use]);

PlayMovie(Xreconstruct)

%% Make arrays of task / behavioral variable indices and timing

tLicks = lick_history; % for nomenclature consistency
% targets
idxTargets = find(event_history(1,:)==idTargetLong);
tTargets = event_history(5,idxTargets) + parameters.Pre_Stimulus_Imaging; % stimulus occurrs x sec after camera turns on
% distractors
idxDistractors = find(event_history(1,:)==idDistractorLong);
tDistractors = event_history(5,idxDistractors) + parameters.Pre_Stimulus_Imaging; % stimulus occurrs x sec after camera turns on
% hits / rewards
% assuming that reward is delivered immediately after hit lick
idxHits = find((event_history(1,:)==idTargetLong & event_history(2,:)==response));
% successful/hit trial
tTrialHit = event_history(5,idxHits);
tRewards = event_history(5,idxHits) + parameters.Pre_Stimulus_Imaging + event_history(3,idxHits); % stimulus occurrs x sec after camera turns on. Response time is time since stimulus
% miss trial
idxMisses = find((event_history(1,:)==idTargetLong & event_history(2,:)==no_response));
tTrialMiss = event_history(5,idxMisses);
% for time regressor - time of start of each trial (when camera turned on)
tTrials = camera_history(1:framesPerTrial:size(videoWF,3));

% plot timing of events
figure
scatter(tTrialHit,ones(1,length(tTrialHit)),'filled'); hold on;
scatter(tTrialMiss,ones(1,length(tTrialMiss)),'filled');
scatter(tTargets,ones(1,length(tTargets)),'filled');
scatter(tDistractors,ones(1,length(tDistractors)),'filled');
scatter(tLicks,ones(1,length(tLicks)),'filled','MarkerFaceColor','k');
legend({"Hit Trials"; "Miss Trials"; "Targets"; "Distractors"; "Licks"});
hold off;
title('Event Times')
xlabel('Time (sec)')

% Identify motivated trials (optional)
% consider only miss trials when mouse is engaged in task
[bMotivatedTrials] = IdentifyMotivatedTrials(tLicks,tTrials);
disp(['Found ' num2str(nnz(bMotivatedTrials)) ' motivated trials and ' ...
    num2str(length(tTrials) - nnz(bMotivatedTrials)) ' unmotivated trials'])

% response times & histograms
rtHits = event_history(3,idxHits);
[rtHistCountsHits,edgesHits] = histcounts(rtHits,0:frameInterval:1);

% response time histogram
figure
bar(edgesHits(1:end-1)+frameInterval/2,rtHistCountsHits)
xlabel('Response Time (sec)')
ylabel('Count')
title('Response Times on Hit Trials')

%% Plot activity maps for hit trials (use non-projected data)

% camera frames for start of each hit trial
frames = (idxHits-1)*framesPerTrial + 1;
% preallocate
vidHits = zeros(vHeight,vWidth,framesPerTrial, length(frames),'single');
for iTrial = 1:length(frames)
    vidHits(:,:,:,iTrial) = videoWF(:,:,frames(iTrial):frames(iTrial)+framesPerTrial-1);
end
% activity on each trial
actiHits = squeeze(mean(mean(vidHits,2),1));
% mean video
vidHits = mean(vidHits,4);
% average activity
signalHits = squeeze(mean(mean(vidHits,2),1));

% plot single-trials and mean
plot(actiHits)
hold on
plot(signalHits,'Linewidth',2,'Color','k')
hold off
ylabel('dF/F')
xlabel('Frame')
title('Cortex-wide activity on hit trials')
graphMin = floor(100*min(actiHits(:)))/100;
graphMax = ceil(100*max(actiHits(:)))/100;
ylim([graphMin graphMax])

% plot activity maps before and after target
nFramesBeforeTarget = 1;
nFramesAfterTarget = 5;
frames = (-nFramesBeforeTarget:1:nFramesAfterTarget) + frameStim;
h = figure;
for iPlot = 1:length(frames)
    g = subplot(1,nFramesBeforeTarget+nFramesAfterTarget+1,iPlot);
    colorMin = round(min(vidHits(:)),2);
    colorMax = round(max(vidHits(:)),2);
    imshow(vidHits(:,:,frames(iPlot)),[colorMin, colorMax])
    colormap jet
    myTitle = ['t = ' num2str((frames(iPlot)-frameStim)*frameInterval) ' s'];
    title(myTitle)
end
colorbar
try
    suptitle('Average activity on hit trials, relative to target presentation (dF/F)')
catch
end
% save
h.WindowState = 'maximized';
saveas(h,[experiment_date ' average activity near time of target.png'])

%% Play movie of average hit trials
% indicate stimulus time (left) and cumulative lick responses (right)

% annotate at bottom (video legend)
valSquare = max(vidHits(:));
sizeSquare = round(vHeight/10);
vidLeg = nan(sizeSquare, vWidth,framesPerTrial);
% color video legend red at time of stimulus
vidLeg(:,1:sizeSquare,frameStim) = valSquare;
% add response histogram
flagLick = false;
for iBin = 1:length(rtHistCountsHits)
    if rtHistCountsHits(iBin) > 0
        flagLick = true;
    end
    if flagLick == true
        % color according to cumulative lick histogram
        val = sum(rtHistCountsHits(1:iBin))/sum(rtHistCountsHits)*valSquare;
        % color right-hand square according to RT hist count
        vidLeg(:,end-sizeSquare+1:end,iBin+frameStim) = val;
    end
end
vidLeg(:,end-sizeSquare+1:end,iBin+frameStim:end) = val; % paint squares red until end of video
vidHits_withLegend = cat(1,vidHits,vidLeg);

PlayMovie(vidHits_withLegend,[colorMin, colorMax], 10);

%% Plot data in new, low-dimensional space
% tips on colors:
% https://www.mathworks.com/matlabcentral/answers/5042-how-do-i-vary-color-along-a-2d-line

% plot hit trials
fraction_trials_plot = .1;

h = figure;
for iTrial = 1:round(length(idxHits)*fraction_trials_plot)
    frameStart = (idxHits(iTrial)-1) * framesPerTrial + 1;
    frameEnd = frameStart + framesPerTrial - 1;
    % define data
    newData = X(:,frameStart:frameEnd)';
    % project onto principle components
    newScore = newData/coeff(:,1:3)';
    % line for this trial
    x = newScore(:,1)';
    y = newScore(:,2)';
    z = newScore(:,3)';
    % This is the color
    col = [ones(1,frameStim-1), linspace(5,10,framesPerTrial-frameStim+1)];  % dark blue before stimulus
    surface([x;x],[y;y],[z;z],[col;col],...
        'edgecolor','interp',...
        'linewidth',2);
    colormap jet
end
xlabel('PC 1 score')
ylabel('PC 2 score')
zlabel('PC 3 score')
title('Hit trial activity projected onto principle axes')
legend({"Cooler color is earlier"})
view(3)
grid on
h.WindowState = 'maximized';

%% Load AVI, select ROIs, Compute Motion Energy
try
    %% Load AVI and save
    if ~isfile('videoAVI.mat')
        tic
        filename = [experiment_date, '_0.avi'];
        [video,ts] = LoadVideoAVI(filename, [], true);
        toc
        disp('Saving AVI MATLAB file')
        save('videoAVI.mat','video', 'ts')
    else
        load('videoAVI.mat')
    end
    
    %% when was the first target trial?
    % index of video timestamp (have to observe full video)
    % "I wave when Im about to start MATLAB. Please match the first paddle
    % movement after waving (or induced paddle movement) to the first target
    % trial." - Krithiga
    
%     PlayMovie(video, [0 255], 20, 'gray');
%     
%     imshow(video(:,:,1),[0, 255])
%     maskPaddle = roipoly;
%     signalPaddle = MotionEnergy(video,maskPaddle);
%     plot(signalPaddle)
    
    %%
    if ~isfile('idxTS_tTargetFirst.mat')
        if strcmp(experiment_date,'201117')
            idxTS_tTargetFirst = 923;  %201117
        end
        if strcmp(experiment_date,'201118')
            idxTS_tTargetFirst = 567; %201118
        end
        
        save('idxTS_tTargetFirst.mat','idxTS_tTargetFirst')
    else
        load('idxTS_tTargetFirst.mat')
    end
    
    % adjust video timestamps so that its on the same clock as widefield camera_history
    tAdjust = ts(idxTS_tTargetFirst) - tTargets(1);
    ts = ts - tAdjust;
    
    %% make whisker mask
    
    if ~isfile('maskWhisker.mat')
        disp('Please select the whisker area of interest (ROI)')
        imshow(video(:,:,1),[0, 255])
        maskWhisker = roipoly;
        imshow(maskWhisker)
        save('maskWhisker','maskWhisker')
    else
        load('maskWhisker.mat')
    end
    
    %% make body movement mask
    
    if ~isfile('maskBody.mat')
        disp('Please select the body area of interest (ROI)')
        imshow(video(:,:,1),[0, 255])
        maskBody = roipoly;
        imshow(maskBody)
        save('maskBody','maskBody')
    else
        load('maskBody.mat')
    end
    
    %% Compute motion energy
    % absolute temporal derivative of video only compute motion energy in ROI
    % motion energy for whiskers
    signalWhisk = MotionEnergy(video,maskWhisker);
    %  interpolate motion energy so that indices match with widefield video
    signalWhisk_interp = interp1(ts(1:end-1), signalWhisk, camera_history)';
    % figure
    % plot(camera_history, signalWhisk_interp)
    % motion energy for body
    signalBody = MotionEnergy(video,maskBody);
    %  interpolate motion energy so that indices match with widefield video
    signalBody_interp = interp1(ts(1:end-1), signalBody, camera_history)';
    % figure
    % plot(camera_history, signalBody_interp)
catch
    disp('No behavioral video found')
end

%% Create lagged regressors and concatenate

if exist('signalWhisk_interp','var')
    regFullNames = ["Licking"; "Target"; "Distractor"; "Reward"; "Whisking"; "Fidgeting"];
else
    regFullNames = ["Licking"; "Target"; "Distractor"; "Reward"];
end

% lags for regressors
% first column is pre-stimulus lag. second column is post-stimulus lag
% rows correspond to regressors in regFullNames
lagsAll = [1 1.6; 0 1.1; 0 1.1; 0 .8];

% make design matrices (will concatenate later)
regLick = DesignFromEventTimes(tLicks,camera_history,...
    lagsAll(regFullNames=='Licking',1),...
    lagsAll(regFullNames=='Licking',2));
regTarget = DesignFromEventTimes(tTargets,camera_history,...
    lagsAll(regFullNames=='Target',1),...
    lagsAll(regFullNames=='Target',2));
regDistractor = DesignFromEventTimes(tDistractors,camera_history,...
    lagsAll(regFullNames=='Distractor',1),...
    lagsAll(regFullNames=='Distractor',2));
regReward = DesignFromEventTimes(tRewards,camera_history,...
    lagsAll(regFullNames=='Reward',1),...
    lagsAll(regFullNames=='Reward',2));

% Concatenate regressors
if exist('signalWhisk_interp', 'var')
    [regFull, regVarLabelsID] = CatRegressors(regLick, regTarget, regDistractor, regReward, signalWhisk_interp, signalBody_interp);
else
    [regFull, regVarLabelsID] = CatRegressors(regLick, regTarget, regDistractor, regReward);
end

%% Quality control and removing NaNs

% For each frame in camera_history, does it correspond with a trial to delete?
idxUnmotivated = false(size(regFull,1),1);
for iTrial = 1:size(event_history,2)
    if ~bMotivatedTrials(iTrial)
        idxStart = (iTrial-1)*framesPerTrial + 1;
        idxEnd = idxStart + framesPerTrial - 1;
        idxUnmotivated(idxStart:idxEnd) = true;
    end
end

% Delete rows of regFull if they contain nan(s)
% signalWhisk_interp will contain NaNs if video ended before widefield video stopped
idxNaNs = any(isnan(regFull),2);
idxRemove = idxUnmotivated | idxNaNs;
% delete trials
if sum(idxRemove) > 0
    disp([num2str(sum(idxRemove)) ' rows from regFull will be excluded in analysis'])
else
    disp('No rows from regFull excluded from analysis')
end

%% run QR and check for rank-defficiency. This will show whether a given regressor is highly collinear with other regressors in the design matrix.
% The resulting plot ranges from 0 to 1 for each regressor, with 1 being
% fully orthogonal to all preceeding regressors in the matrix and 0 being
% fully redundant. Having fully redundant regressors in the matrix will
% break the model, so in this example those regressors are removed. In
% practice, you should understand where the redundancy is coming from and
% change your model design to avoid it in the first place!
% (from Churchland / Musall demo code: tutorial_linearModel

% Dov edit - code will now only indicate that redundancy exists, won't
% remove regressors automatically.

[~, qrrFull] = qr(regFull ./ sqrt(sum(regFull.^2)),0); %orthogonalize normalized design matrix
%this shows how orthogonal individual regressors are to the rest of the matrix
figure
plot(abs(diag(qrrFull)),'linewidth',2);
ylim([0 1.1]);
title('Regressor orthogonality'); 
axis square; 
ylabel('Norm. vector angle'); 
xlabel('Regressors');
hold on
plot(single(regVarLabelsID)/(max(single(regVarLabelsID))/.4));
hold off
legend({"Regressor orthogonality"; "Variable ID (all lags)"},'Location','Southeast');

if sum(abs(diag(qrrFull)) > max(size(regFull)) * eps(qrrFull(1))) < size(regFull,2) %check if design matrix is full rank
    rejIdx = false(1,size(regFull,2));
    temp = ~(abs(diag(qrrFull)) > max(size(regFull)) * eps(qrrFull(1)));
    fprintf('Design matrix is rank-defficient. Need to remove %d/%d additional regressors.\n', sum(temp), sum(~rejIdx));
    rejIdx(~rejIdx) = temp; %reject regressors that cause rank-defficint matrix
else
    disp('Design matrix is full-rank')
end

%% Run full regression (ordinary least squares) on high-dimensional data
% fits regression model to each pixel in movie
% Warning - this could take a while if videoWF is full resolution

flagGO = questdlg('Fit model to high-dimensional data? This could take a while...');
if strcmp(flagGO,'Yes')
    tic
    [mapsLM] = MakeBetaMaps(regFull(~idxRemove,:), videoWF(:,:,~idxRemove));
    beep
    toc
    
    % add some other info to structure and save
    mapsLM.regVarLabelsID = regVarLabelsID;
    mapsLM.regFullNames = regFullNames;
    mapsLM.lagsAll = lagsAll;
    % save('mapsLM.mat', 'mapsLM')
    
    % Plot R-squared map and all Beta Maps
    nFramesAroundEvent = 3;
    PlotBetaMaps(mapsLM,nFramesAroundEvent)
    
    % Play Movie of beta values for a regressor
    myVar = "Licking";
    iReg = find(regFullNames==myVar);
    
    disp(['Playing beta movie for the ' char(regFullNames(iReg)) ' regressor'])
    % PlayMovie(mapsLM.B(:,:,(regVarLabelsID==iReg)),[-.1 .1])
    PlayMovie(mapsLM.B(:,:,(regVarLabelsID==iReg)),[-.1 .1], 10, 'jet')
else
    
end

%% Run full regression (ordinary least squares) on low-dimensional data

% remove nans/unmotivated trials, normalize regression matrix
regFull_norm = regFull(~idxRemove,:);
regFull_norm = (regFull_norm - mean(regFull_norm,1)) ./ std(regFull_norm,0,1);
% pre-allocate beta weights for each PC
betasPCs = zeros(size(regFull_norm,2),nPCs);

% find beta values for each dimension
tic
for iPC = 1:nPCs
    scoresPC = score(iPC,~idxRemove)';
    % fit multiple regression model to low-D data
    lm = fitlm(regFull_norm,scoresPC,'RobustOpts','ols');
    betasPCs(:,iPC) = lm.Coefficients.Estimate(2:end);
end

% project beta weights back up to orignal space
mapsLM_B = betasPCs*coeff(:,1:nPCs)';
mapsLM.B = reshape(mapsLM_B',[vHeight,vWidth,size(regFull_norm,2)]);

% Calculate R-squared
% calculate model estimates in low-dimensional space (y ~= Xb)
scoreModel = regFull(~idxRemove,:) * betasPCs;
% project results back up to original space
XreconstructModel = scoreModel*coeff(:,1:nPCs)';
% Xreconstruct = Xreconstruct + mu;
XreconstructModel = XreconstructModel';
XreconstructModel = reshape(XreconstructModel,[vHeight,vWidth,sum(~idxRemove)]);
% PlayMovie(Xreconstruct)

% find R-squared for each pixel
mapsLM_Rsquared = zeros(vHeight,vWidth);
mapsLM_Rsquared_p = zeros(vHeight,vWidth);
for i = 1:vHeight
    for ii = 1:vWidth
        % get correlation
        [R,P] = corrcoef(squeeze(videoWF(i,ii,~idxRemove)),squeeze(XreconstructModel(i,ii,:)));
        mapsLM_Rsquared(i,ii) = R(2)^2; % R-squared
        mapsLM_Rsquared_p(i,ii) = P(2);
    end
end
toc

% store in structure
mapsLM.Rsquared = mapsLM_Rsquared;
mapsLM.Rsquared_p = mapsLM_Rsquared_p;
% add some other info to structure and save
mapsLM.regVarLabelsID = regVarLabelsID;
mapsLM.regFullNames = regFullNames;
mapsLM.lagsAll = lagsAll;
mapsLM.nPCs = nPCs;
% save('mapsLM.mat', 'mapsLM')

% Plot R-squared map and all Beta Maps
nFramesAroundEvent = 3;
PlotBetaMaps(mapsLM,nFramesAroundEvent)

%% Play Movie of beta values for a regressor
myVar = "Licking";
iReg = find(regFullNames==myVar);

disp(['Playing beta movie for the ' char(regFullNames(iReg)) ' regressor'])
PlayMovie(mapsLM.B(:,:,(regVarLabelsID==iReg)),[-.01 .01], 10, 'jet')

%% Run full regression (ridge regression)
% From Musall, Kaufman et al. 2019 code
% https://github.com/churchlandlab/ridgeModel

[ridgeVals, betasPCs_ridge] = ridgeMML(score', regFull, true); %get ridge penalties and beta weights.
% compute model predictions by multiplying regression matrix with beta weights
scoreModel_ridge = (regFull * betasPCs_ridge)';
% compute explained variance
corrMat = modelCorr(score,scoreModel_ridge,coeff) .^2;
corrMat = reshape(corrMat,vHeight,vWidth);
% store in structure
mapsRR.Rsquared = corrMat;
% project beta weights to original space
mapsRR_B = coeff * betasPCs_ridge';
mapsRR.B = reshape(mapsRR_B, vHeight, vWidth, []);
% add some other info to structure and save
mapsRR.regVarLabelsID = regVarLabelsID;
mapsRR.regFullNames = regFullNames;
mapsRR.lagsAll = lagsAll;
mapsRR.nPCs = nPCs;
% save('mapsLM.mat', 'mapsLM')

% plot explained variance
h = figure;
imshow(corrMat,[0,.6],'Colormap',jet, 'InitialMagnification', 400)
title('Explained Variance')
colorbar
h.WindowState = 'maximized';

%% Run cross-validated ridge regression

% how many iterations?
folds = 10;

% run cross-validated model
[scoreModel_ridgeCV, betasPCs_ridgeCV] = ...
    crossValModel(regFull, score, regFullNames, regVarLabelsID, regFullNames, folds);

% compute explained variance
fullMat = modelCorr(score,scoreModel_ridgeCV,coeff) .^2; 
mapsRRCV.Rsquared = reshape(fullMat,vHeight,vWidth);
% average beta weights across folds
betasPCs_ridgeCV = cell2mat(betasPCs_ridgeCV);
betasPCs_ridgeCV = reshape(betasPCs_ridgeCV,size(regFull,2),size(score,1),folds);
betasPCs_ridgeCV = mean(betasPCs_ridgeCV,3);
% project beta weights to original space
mapsRRCV_Beta = coeff * betasPCs_ridgeCV';
mapsRRCV.B = reshape(mapsRRCV_Beta, vHeight, vWidth, []);
% store other variables in this structure
mapsRRCV.regVarLabelsID = regVarLabelsID;
mapsRRCV.regFullNames = regFullNames;
mapsRRCV.lagsAll = lagsAll;
save('mapsRR.mat', 'mapsRR')

% Plot R-squared map and all Beta Maps (cross-validated ridge regression)
nFramesAroundEvent = 3;
PlotBetaMaps(mapsRRCV,nFramesAroundEvent)

%% Inspect difference between OLS and RR model
% plot R-squared and beta maps side-by-side

% which beta map to plot?
myVar = "Licking";
myLag = 2; % post-event frame

h = figure;
% plot R-squared map, OLS
subplot(2,3,1)
imshow(mapsLM.Rsquared, [0,0.6], 'Colormap', jet, 'InitialMagnification', 400);
myTitle = 'R-squared map, OLS model';
title(myTitle);
colorbar

% plot R-squared map, RR
subplot(2,3,2)
imshow(mapsRRCV.Rsquared, [0,0.6], 'Colormap', jet, 'InitialMagnification', 400);
myTitle = 'R-squared map, RR model';
title(myTitle);
colorbar

% plot difference in R-squared maps
subplot(2,3,3)
imshow(mapsRRCV.Rsquared - mapsLM.Rsquared, [0,0.2], 'Colormap', jet, 'InitialMagnification', 400);
myTitle = 'Difference of R-squared maps (RR-OLS)';
title(myTitle);
colorbar

subplot(2,3,4)
iReg = find(regFullNames==myVar);
myBetas1 = mapsLM.B(:,:,(regVarLabelsID==iReg));
myFrame = lagsAll(iReg,1)/frameInterval+myLag+1;
imshow(myBetas1(:,:,myFrame),[-.01 .01], 'Colormap', jet, 'InitialMagnification', 400);
myTitle = [char(regFullNames(iReg)) ' regressor betas, lag = ' num2str(myLag*frameInterval) ' sec (OLS)'];
title(myTitle);
colorbar

subplot(2,3,5)
iReg = find(regFullNames==myVar);
myBetas2 = mapsRRCV.B(:,:,(regVarLabelsID==iReg));
myFrame = lagsAll(iReg,1)/frameInterval+myLag+1;
imshow(myBetas2(:,:,myFrame),[-.01 .01], 'Colormap', jet, 'InitialMagnification', 400);
myTitle = [char(regFullNames(iReg)) ' regressor betas, lag = ' num2str(myLag*frameInterval) ' sec (RR)'];
title(myTitle);
colorbar

subplot(2,3,6)
iReg = find(regFullNames==myVar);
imshow(myBetas2(:,:,myFrame) - myBetas1(:,:,myFrame),[0 .007], 'Colormap', jet, 'InitialMagnification', 400);
myTitle = 'Difference of beta maps (RR-OLS)';
title(myTitle);
colorbar


h.WindowState = 'maximized';

%% Shuffle trials for a regressor
% this will shuffle trials. If you want to shuffle frames instead, make the
% third argument in RegShuffTrials equal 1. This can be used to compute
% variance explained by chance (without changing the number of predictors)

% Shuffle one variable (e.g. Target)
% which columns in regFull will have their rows shuffled?
varCols = regVarLabelsID == find(regFullNames=='Target');
regFull_shuffTarget = RegShuffTrials(regFull, varCols, framesPerTrial);

% plot difference between normal and shuffled regressor matrix
img1 = regFull(1:framesPerTrial*20,:);
img2 = regFull_shuffTarget(1:framesPerTrial*20,:);
imgDiff = img2 - img1;
figure
imagesc(imgDiff,[-1 1])
title({strjoin(['Regression matrix shuffling,' regFullNames(regVarLabelsID(find(varCols==1,1,'first'))) 'variable only']);...
    'Change in regression matrix for first 20 trials'});
ylabel('Camera Frame')
xlabel('Regressor')

% %% Shuffle all variables iteratively
% % this will shuffle trials. If you want to shuffle frames instead, make the
% % third argument in RegShuffTrials equal 1.
% 
% regFull_shuffAll = regFull;
% for iVar = 1:max(regVarLabelsID)
%     varCols = regVarLabelsID == iVar;
%     regFull_shuffAll = RegShuffTrials(regFull_shuffAll, varCols, framesPerTrial);
% end
% 
% % plot difference between normal and shuffled regressor matrix
% img1 = regFull(1:framesPerTrial*20,:);
% img2 = regFull_shuffAll(1:framesPerTrial*20,:);
% imgDiff = img2 - img1;
% figure
% imagesc(imgDiff,[-1 1])
% title({'Regression matrix shuffling, all variables';...
%     'Change in regression matrix for first 20 trials'});ylabel('Camera Frame')
% xlabel('Regressor')

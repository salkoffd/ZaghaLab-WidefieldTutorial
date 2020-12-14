function [mapsLM] = MakeBetaMaps(X, movie, predictorLabels)
% [mapsLM] = MakeBetaMaps(X, movie, predictorLabels)
% Fits a linear regression model to each pixel in a movie
% uses the fitlm function on each pixel independently. Use this function to
% calculate beta maps for each predictor.
%
% Inputs:
%   X: the regressor matrix and is identical to the X variable described in
%       the documentation of fitlm. n x p where n is observations and p is
%       the number of predictors.
%   movie: a matrix of pixel activity over time (height x width x frame)
%   predictorLabels: a string of the predictor labels. (For plotting - optional)
%
% output:
%   mapsLM: structure containing:
%       Rsquared: map of total variance explained by the model (height x width)
%       Rsquared_p: p-value associated with the full model. Tests the null
%           hypothesis that all coefficients are equal to 0.
%       B: matrix of all beta values across all pixels and predictors. (heigh x width x predictor)
%       P: p-value associated with the Beta values in B (heigh x width x predictor)
%       sig: logical matrix describing significance at alpha level for each Beta value in B.
%
% last updated 12/8/20

plot_results = false;

% scale y values before regressing?
bScaleY = false;

% alpha level for significance
alpha = .05;

% movie dimensions
movieHeight = size(movie,1);
movieWidth = size(movie,2);

% how many regressors?
nRegressors = size(X,2);

% center and scale regressors
X = (X - mean(X,1)) ./ std(X,0,1);

% preallocate maps
% total variance explained by model for each pixel
mapsLM.Rsquared = zeros(movieHeight,movieWidth);
% maps for beta coefficients and associated p-values
mapsLM.B = zeros(movieHeight,movieWidth,nRegressors); 
mapsLM.P = zeros(movieHeight,movieWidth,nRegressors,'single');
mapsLM.sig = false(movieHeight,movieWidth,nRegressors);

completeThresh = .1;
% generate model for each pixel iteratively
for i = 1:movieHeight
    if i/movieHeight >= completeThresh
        disp([num2str(completeThresh*100) '%'])
        completeThresh = completeThresh + .1;
    end
    for ii = 1:movieWidth
        pixActivity = squeeze(movie(i,ii,:));
        % center and scale y
        if bScaleY
            pixActivity = (pixActivity - mean(pixActivity)) / std(pixActivity);
        end
        % fit model
        lm = fitlm(X,pixActivity,'linear');
        % test if model is significant (are any coefficients significantly
        % different than 0?)
        [p] = coefTest(lm);
        % store model variables
        mapsLM.Rsquared(i,ii) = lm.Rsquared.Ordinary;
        mapsLM.Rsquared_p(i,ii) = p;
        mapsLM.B(i,ii,:) = lm.Coefficients.Estimate(2:end);
        mapsLM.P(i,ii,:) = lm.Coefficients.pValue(2:end);
        mapsLM.sig(i,ii,:) = lm.Coefficients.pValue(2:end)<alpha;
    end
end

if plot_results == true
    
    if nargin == 2
        predictorLabels = string("Predictor " + [1:nRegressors]);
    elseif length(predictorLabels) ~= nRegressors
        disp('Warning: Make sure your predictor labels are correct')
        predictorLabels = string("Predictor " + [1:nRegressors]);
    end
    
    % plot R squared map
    figure
    valMax = max(max(mapsLM.Rsquared));
    colorMax = round(valMax,1);
    imshow(mapsLM.Rsquared, [0,colorMax], 'Colormap', jet, 'InitialMagnification', 1000);
    colorbar
    title('R squared map');
    
    % number of rows and columns for subplot
    figRows = floor(nRegressors^.5);
    figCols = ceil(nRegressors/figRows);
    
    % plot beta maps
    valMax = max(max(max(mapsLM.B)));
    figure;
    suptitle('Multiple Regression Beta Maps')
    for iPlot = 1:nRegressors
        subplot(figRows,figCols,iPlot)
        colorMax = round(valMax,1);
        j = imshow(mapsLM.B(:,:,iPlot), [0,colorMax], 'Colormap', jet, 'InitialMagnification', 200);
        set(j,'AlphaData',mapsLM.sig(:,:,iPlot)+.5);
        colorbar
        title(predictorLabels(iPlot));
    end
end

end


function PlotBetaMaps(mapsLM,nFramesAroundEvent,title_prefix)
% PlotBetaMaps(mapsLM,nFramesAroundEvent,title_beginning)
%   Plots R-squared map and beta maps using mapsLM structure
%   For use with MakeBetaMaps function and MultipleRegressionTutorial script
%
% last updated 12/9/20

% time between camera frames
frameInterval = 0.1;
% set opacity to significance (p<0.05)?
use_transparency = false;
% save figures?
save_figures = false;

if ~exist('title_beginning','var')
    title_prefix = '';
else
    title_prefix = [title_prefix '_'];
end

try
    % plot R-squared map
    colorMax = min([max(round(mapsLM.Rsquared(:),1))+.1 1]);
    h = figure;
    g = imshow(mapsLM.Rsquared, [0,colorMax], 'Colormap', jet, 'InitialMagnification', 400);
    colorbar
    if use_transparency
        ad = mapsLM.Rsquared_p < 0.05;
        ad = ad + 0.25;
        set(g, 'AlphaData', ad)
    end
    myTitle = 'R-squared map, full model';
    title(myTitle);
    % save
    h.WindowState = 'maximized';
    if save_figures
        saveas(h,[title_prefix myTitle '.png'])
    end
catch
    disp('No R-squared map found')
end

for iVar = 1:length(mapsLM.regFullNames)
    h = figure;
    % analog variable?
    if sum(mapsLM.regVarLabelsID==iVar) == 1
        % analog variable
        % find beta map for this variable
        img = mapsLM.B(:,:,mapsLM.regVarLabelsID==iVar);
        % transparency based on p-value
        colorMax = round(max(img(:)),2);
        g = imshow(img, [-colorMax, colorMax], 'Colormap', jet, 'InitialMagnification', 400);
        colorbar
        myTitle = ['Beta map, regressor = ' char(mapsLM.regFullNames(iVar))];
        title(myTitle);
        if use_transparency
            ad = mapsLM.P(:,:,mapsLM.regVarLabelsID==iVar) < 0.05;
            ad = ad + 0.25; % non-significant pixels will be transparent
            set(g, 'AlphaData', ad)
        end
        % save
        h.WindowState = 'maximized';
        if save_figures
            saveas(h,[title_prefix myTitle '.png'])
        end
    else
        % not analog (event-based variable)
        idx = find(mapsLM.regVarLabelsID==iVar);
        imgs = mapsLM.B(:,:,idx);
        if use_transparency
            % set transparency
            ad = mapsLM.P(:,:,idx) < 0.05;
            ad = ad + .25;
        end
        colorMax = round(max(imgs(:)), 3);
        % index of beta map at event lag 0
        idxT0 = round(mapsLM.lagsAll(iVar,1)/frameInterval) + 1;
        if idxT0 == 1
            % task variable - display maps after event
            for iPlot = 1:1+nFramesAroundEvent
                subplot(1, 1+nFramesAroundEvent, iPlot)
                g = imshow(imgs(:,:,iPlot), [-colorMax,colorMax], 'Colormap', jet, 'InitialMagnification', 400);
                title(['t = ' num2str((iPlot-1)*frameInterval) ' s'])
                if use_transparency
                    set(g, 'AlphaData', ad(:,:,iPlot));
                end
            end
            colorbar
            myTitle = ['Beta map, regressor = ' char(mapsLM.regFullNames(iVar))];
            try
                suptitle(myTitle);
            catch
                disp('Cannot display suptitle')
            end
            % save
            h.WindowState = 'maximized';
            if save_figures
                saveas(h,[title_prefix myTitle '.png'])
            end
        else
            % behavioral variable - display maps before and after event
            for iPlot = 1:1+2*nFramesAroundEvent
                idx = idxT0 - nFramesAroundEvent + iPlot - 1;
                subplot(1, 1+2*nFramesAroundEvent, iPlot)
                g = imshow(imgs(:,:,idx), [-colorMax,colorMax], 'Colormap', jet, 'InitialMagnification', 400);
                title(['t = ' num2str((idx - idxT0)*frameInterval) ' s'])
                if use_transparency
                    set(g, 'AlphaData', ad(:,:,idx));
                end
            end
            colorbar
            myTitle = ['Beta map, regressor = ' char(mapsLM.regFullNames(iVar))];
            try
                suptitle(myTitle);
            catch
                disp('Cannot display suptitle')
            end
            % save
            h.WindowState = 'maximized';
            if save_figures
                saveas(h,[title_prefix myTitle '.png'])
            end
        end
    end
end



end


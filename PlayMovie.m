function PlayMovie(movie, gain, fps, cm, winSize)
%   PlayMovie(movie, gain, fps, cm, winSize)
%
%   Inputs:
%     movie: height-by-width-by-frame array
%     gain: make the color scale more dramatic by increasing gain (optional
%           argument, range 0-1, default = .4). Alternatively, specify the
%           color min and max with a vector [min max].
%     fps: frames per second
%     cm: colormap e.g. 'gray' or 'jet'
%     winSize: window size. Set from 0 to 1 with 1 being full-screen
%
%   Written by David Salkoff dovsalkoff@gmail.com
%   Updated 12/9/20

if ~exist('winSize', 'var') || isempty(winSize)
    winSize = .6; % range 0-1
end

% default gain
if ~exist('gain', 'var') || isempty(gain)
    gain = .4; % range 0-1
end

if ~exist('fps', 'var') || isempty(fps)
    fps = 10;
end

if ~exist('cm', 'var') || isempty(cm)
    cm = 'jet';
end

% min and max values for color scale
if length(gain) == 1
    value_min = min(movie(:)) * (1-gain+.001);
    value_max = max(movie(:)) * (1-gain+.001);
elseif length(gain) == 2
    value_min = gain(1);
    value_max = gain(2);
end

% play movie with implay
handle1 = implay(movie, fps);
handle1.Visual.ColorMap.UserRangeMin = value_min;
handle1.Visual.ColorMap.UserRangeMax = value_max;
handle1.Visual.ColorMap.UserRange = 1;
handle1.Visual.ColorMap.MapExpression = cm;

% enlarge window and zoom
myScreenSize = get(0,'ScreenSize');
set(findall(0,'tag','spcui_scope_framework'),'position',...
    [50 50 myScreenSize(3)*winSize myScreenSize(4)*winSize]); %window size
set(0,'showHiddenHandles','on')
fig_handle = gcf ;  
fig_handle.findobj; % to view all the linked objects with the vision.VideoPlayer
ftw = fig_handle.findobj ('TooltipString', 'Maintain fit to window');   % this will search the object in the figure which has the respective 'TooltipString' parameter.
ftw.ClickedCallback();  % execute the callback linked with this object

end


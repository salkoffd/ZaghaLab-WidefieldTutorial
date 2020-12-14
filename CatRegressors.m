function [regFull, regVarLabelsID] = CatRegressors(varargin)
% [regFull, regVarLabelsID] = CatRegressors(r1,r2,r3,...rN)
% Concatenates regressors into full regressor matrix
% 
% Input(s):
%   [r1,r2,r3,...rN] regressor matrices for single variables with lagged
%   copies, separated by commas. Each matrix is frames-by-copies where
%   frames is the same for all matrices.
%
% Outputs:
%   regFull: full regression matrix (frames-by-copies*variables)
%   regVarLabels: for each regressor copy, specifies the ID of the
%       variable/predictor (where ID is order of inputs).
%
% Written by David Salkoff 12/10/20

regFull = single(cat(2,varargin{:}));

regVarLabelsID = zeros(1,size(regFull,2),'int16');
idxStart = 1;
for iReg = 1:length(varargin)
    thisVarLength = size(varargin{iReg},2);
    idxEnd = idxStart+thisVarLength-1;
    regVarLabelsID(idxStart:idxEnd) = iReg;
    idxStart = idxEnd + 1;
end

end


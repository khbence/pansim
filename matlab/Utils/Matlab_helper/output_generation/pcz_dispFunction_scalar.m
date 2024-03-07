function [ret] = pcz_dispFunction_scalar(varargin)
%% pcz_dispFunction_scalar
%  
%  File: pcz_dispFunction_scalar.m
%  Directory: utilities/output_generation
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2020. March 19. (2019b)
%  Major review on 2020. May 01. (2019b)
%

for i = 1:nargin
        
    seq = strjoin( cellfun(@(n) {num2str(n)},num2cell(varargin{i}(:)')) , ',' );
    
    if ~isscalar(varargin{i})
        seq = sprintf('[%s]', seq);
    end

    pcz_dispFunction2('%s = %s', inputname(i), seq);
end

end

classdef PStatus
    %%
    %
    %  file:   PStatus.m
    %  author: Polcz Péter <ppolcz@gmail.com>
    %
    %  Created on 2016.01.19. Tuesday, 13:48:44
    %
    
    %%
    properties (Constant = true)
    end
    
    properties (GetAccess = private, SetAccess = private)
        all_iter, all_progress, prog;
    end
    
    properties (GetAccess = public, SetAccess = public)
    end
    
    methods (Access = public)
        %% public
        
        function o = PStatus(all_iter, varargin)
            o.all_iter = all_iter;

            if isempty(varargin) || ischar(varargin{1})
                o.all_progress = 100;
            else
                o.all_progress = varargin{1};
                varargin = varargin(2:end);
            end
            
            if ~isempty(varargin)
                msg = sprintf(varargin{:});
            else
                msg = 'Progress';
            end
            
            pcz_dispFunction('%s..........', msg);
            
            o = o.init();
        end
        
        function o = progress(o,iter)
            new_prog = round(iter / o.all_iter * o.all_progress);
            if (new_prog > o.prog)
                o.prog = new_prog;
                
                fprintf('\b\b\b\b\b\b\b%6.2f%%',100*o.prog/o.all_progress); 
                
                if new_prog == o.all_progress
                    fprintf('\n')
                end
                
                % Old and deprecated:
                % pcz_debug(o.depth, o.msg, o.prog, '/', o.all_progress)

            end
            
        end
        
        function o = init(o)
            o.prog = 0;
        end
        
    end
    
    methods (Access = private)
        %% private
    end
    
end

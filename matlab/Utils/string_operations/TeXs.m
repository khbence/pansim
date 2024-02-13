function [ret] = TeXs(s,args)
arguments
    s
    args.EscapeFor string {mustBeMember(args.EscapeFor,["disp","sprintf"])} = "sprintf"
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2023. August 22. (2023a)
%

ret = pcz_latexify_accents(s,"EscapeFor",args.EscapeFor);

end
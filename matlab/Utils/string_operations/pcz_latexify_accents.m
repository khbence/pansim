function s = pcz_latexify_accents(s,args)
arguments
    s
    args.EscapeFor string {mustBeMember(args.EscapeFor,["disp","sprintf"])} = "disp"
end
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2023. August 22. (2023a)
%

% s = 'árvíztűrő tükörfúrógép ÁRVÍZTŰRŐ TÜKÖRFÚRÓGÉP';
% Rekonstruált járványgörbék a tényleges lezárások alkalmazásával';

Replace_Accent = {
    "á"  "\'{a}"
    "é"  "\'{e}"
    "í"  "\'{i}"
    "ó"  "\'{o}"
    "ú"  "\'{u}"
    "ö" "\""{o}"
    "ü" "\""{u}"
    "ő"  "\H{o}"
    "ű"  "\H{u}"
    "Á"  "\'{A}"
    "É"  "\'{E}"
    "Í"  "\'{I}"
    "Ó"  "\'{O}"
    "Ú"  "\'{U}"
    "Ö" "\""{O}"
    "Ü" "\""{U}"
    "Ű"  "\H{O}"
    "Ő"  "\H{U}"
    };


for rule = Replace_Accent.'
    s = strrep(s,rule{1},rule{2});
end

switch args.EscapeFor
    case "sprintf"
        s = strrep(s,'\','\\\\');

end
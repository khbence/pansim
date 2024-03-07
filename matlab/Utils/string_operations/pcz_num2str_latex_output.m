function [ret] = pcz_num2str_latex_output(A,varargin)
%% Script pcz_num2str_latex_output
%  
%  File: pcz_num2str_latex_output.m
%  Directory: projects/3_outsel/2018_01_10_LPV_inversion
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  
%  Created on 2018. January 14.
%
%%

global LATEX_EQNR
LATEX_EQNR = LATEX_EQNR + 1;

o.label = '';
o.format = '%-8.3g';
o.del1 = ' & ';
o.del2 = ' \\\\\n';
o.pref = '        ';
o.round = 4;
o.esc = 1;
o.vline = [];
o.hline = [];
o.postlabel = '';
o = parsepropval(o,varargin{:});


if isempty(LATEX_EQNR)
    fprintf('\n\\begin{align} \n')
else
    fprintf('\n\\begin{align} {\\LARGE(%d) \\quad}\n', LATEX_EQNR)
end


if ~iscell(A)

    tol = 10^(-o.round);
    A( A < tol & A > -tol ) = 0;
    o.A = A;

    if isempty(o.label) && ~isempty(inputname(1))
        o.label = [ inputname(1) ' = ' ];
    elseif ~isempty(o.label) && ~o.esc
        o.label = strrep(o.label,'\','\\');
    end

    coldef = repmat('c',[1 size(A,2)]);
    if ~isempty(o.vline)
        bar = repmat(' ',[1 size(A,2)+1]);
        bar(o.vline+1) = '|';
        coldef = [ bar(1) reshape([coldef ; bar(2:end)], [1 size(A,2)*2]) ];
        coldef = strrep(coldef, ' ', '');
    end 

    b = ['    ' o.label '\\left(\\begin{array}{' coldef '}\n'];

    A_latex = pcz_num2str_latex(o.A, 'format', o.format, ...
        'beginning', b, ...
        'ending', ' \\\\\n    \\end{array}\\right)', ...
        'del1', o.del1, 'del2', o.del2, 'pref', o.pref, 'round', o.round);

    nl_index = strfind(A_latex, newline);
    for linenr = sort(o.hline,'desc')
        i = nl_index(linenr+1);
        A_latex = [A_latex(1:i-1) ' \hline ' A_latex(i:end) ];
    end

    disp(A_latex)

else
    assert(isempty(o.label) || ( iscell(o.label) && all(size(o.label) == size(A)) ), ...
        'Label should be empty or a cell with the same size')
    
    if isempty(o.label)
        o.label = repmat({o.label}, size(A));
    elseif isempty(o.postlabel)
        o.postlabel = ' = ';
    end
        
    A_cell = A;
    for i = 1:numel(A_cell)
        
        A = A_cell{i};
        
        tol = 10^(-o.round);
        A( A < tol & A > -tol ) = 0;
        o.A = A;
        
        coldef = repmat('c',[1 size(A,2)]);
        b = ['    ' o.label{i} o.postlabel '\\left(\\begin{array}{' coldef '}\n'];

        A_latex = pcz_num2str_latex(o.A, 'format', o.format, ...
            'beginning', b, ...
            'ending', ' \\\\\n    \\end{array}\\right)', ...
            'del1', o.del1, 'del2', o.del2, 'pref', o.pref, 'round', o.round);
        
        disp(A_latex)
        
        if i < numel(A_cell)
            disp ',\quad'
        end
    end
end

disp '\end{align}'


end
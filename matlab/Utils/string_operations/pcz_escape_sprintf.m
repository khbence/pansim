function text = pcz_escape_sprintf(text)

% escape newline
text = strrep(text, newline, '\n');

% escape comment mark
text = strrep(text, '%', '%%');

end
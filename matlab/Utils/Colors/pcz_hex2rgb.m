function [RgbColor] = pcz_hex2rgb(HexColor)
%%
%  Author: Peter Polcz (ppolcz@gmail.com) 
%  Created on 2023. January 05. (2022b)
%

HexColor = char(HexColor);
HexColor = hex2dec(HexColor(2:end));

b = mod(HexColor,256) / 255;
rg = floor(HexColor / 256);
g = mod(rg,256) / 255;
r = floor(rg / 256) / 255;

RgbColor = [r g b];

end
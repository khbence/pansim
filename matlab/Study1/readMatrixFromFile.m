function matrix = readMatrixFromFile(filename)
    fileID = fopen(filename, 'r');
    
    fileContent = textscan(fileID, '%s', 'Delimiter', '\n');
    fileContent = fileContent{1};
    
    fclose(fileID);
    
    numRows = numel(fileContent);
    matrix = cell(numRows, 8);
    
    for row = 1:numRows
        line = fileContent{row};
        
        elements = strsplit(line, '|');
        
        for col = 1:8
            if col == 8
                matrix{row, col} = str2double(elements{col});
            else
                matrix{row, col} = elements{col};
            end
        end
    end
    
    % matrix = cell2mat(matrix);
end
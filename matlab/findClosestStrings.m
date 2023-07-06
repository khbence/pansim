function closestStrings = findClosestStrings(matrix, referenceValue)

    floatColumn = matrix(:, end);
    stringColumns = matrix(:, 1:end-1);
    
    [~, index] = min(abs(floatColumn - referenceValue));
    
    closestStrings = stringColumns(index, :);
end
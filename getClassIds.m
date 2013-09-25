function [classes, map, uniqueLabels] = getClassIds(textLabels)
%GETCLASSIDS Convert string labels to integer class ids
%   textLabels:     cell array of strings
%   classes:        array of integer class ids
%   map:            containers.Map object mapping the labels to the ids
%   uniqueLabels:   reverse mapping of class ids to labels


N = length(textLabels);
m = 1;

classes = zeros(size(textLabels));
map = containers.Map();

for n = 1:N
    label = textLabels{n};
    
    if isKey(map, label)
        classes(n) = map(label);
    else
        uniqueLabels{m} = label; %#ok
        map(label) = m;
        classes(n) = m;
        m = m + 1;
    end
end

end


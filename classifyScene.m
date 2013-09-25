function classifyScene(trainingDataFile,testDataFile,outFile)
%% Path and Label Import

disp('Importing Training Data')

fid = fopen(trainingDataFile);
t = textscan(fid, '%s %s');
fclose(fid);

wavPaths = t{1};
labels = t{2};
nFiles = length(wavPaths);

disp('Importing Complete')

%% Configuration

wLen = 2^12;
wOverlap = 0.5;
window = hamming(wLen);


%% Feature Selection

disp('Extracting Training Features')

features = cell(nFiles,1);

for fileNum = 1:nFiles
    [rawData, Fs] = wavread(wavPaths{fileNum});
    
    [a1, a2, ~] = alignsignals(rawData(:, 1), rawData(:, 2), round(Fs/10), 'truncate');
    a = (a1 + a2)./2;
    
    bufData = buffer(a, wLen, wLen*wOverlap); % each column is a frame
    
    N = size(bufData, 2);
    bufData = bufData .* repmat(window, 1, N);
    
    rmsVals = rms(bufData, 1);
    
    % loudness measure over frames
    loudness = 20.*log10(rmsVals);
    
    % magnitude response over frames
    y = fft(bufData, wLen, 1);
    magRes = mag2db(abs(y));
    
    % Spectral Sparsity
    specSparsity = max(magRes) ./ sum(magRes, 1);
    
    indices = 1:2:60;
    
    features(fileNum) = {[magRes(indices,:)',loudness',specSparsity']};
    
end

clear rawData a1 a2 a y;

disp('Extraction Complete')

%% Classifier

disp('Training')

prtPath('beta')
training = prtDataSetTimeSeries;
training = training.setX(features);
[classes, ~, uniqueLabels] = getClassIds(labels);
training = training.setY(classes);
training.classNames = uniqueLabels;

states = 2;
mixtures = 2;
classifier = prtClassMap;
hmm = prtRvHmm('components',repmat(prtRvGmm('nComponents',mixtures),states,1));
hmm.minimumComponentMembership = 0;
classifier.rvs = hmm;
decider = prtDecisionMap;
alg = classifier + decider;
alg = alg.train(training);

disp('Training Complete')

%% Testing File

fid = fopen(testDataFile);
t = textscan(fid, '%s %s');
fclose(fid);

wavPaths = t{1};
nFiles = length(wavPaths);

%% Feature Selection

disp('Extracting Testing Features')

features = cell(nFiles,1);

for fileNum = 1:nFiles
    [rawData, Fs] = wavread(wavPaths{fileNum});
    
    [a1, a2, ~] = alignsignals(rawData(:, 1), rawData(:, 2), round(Fs/10), 'truncate');
    a = (a1 + a2)./2;
    
    bufData = buffer(a, wLen, wLen*wOverlap); % each column is a frame
    
    N = size(bufData, 2);
    bufData = bufData .* repmat(window, 1, N);
    
    rmsVals = rms(bufData, 1);
    
    % loudness measure over frames
    loudness = 20.*log10(rmsVals);
    
    % magnitude response over frames
    y = fft(bufData, wLen, 1);
    magRes = mag2db(abs(y));
    
    % Spectral Sparsity
    specSparsity = max(magRes) ./ sum(magRes, 1);
    
    indices = 1:2:60;
    
    features(fileNum) = {[magRes(indices,:)',loudness',specSparsity']};
    
    
end

clear rawData a1 a2 a y;

disp('Extraction Complete')

%% Classification

disp('Classifying')

testing = prtDataSetTimeSeries;
testing = testing.setX(features);
testing.classNames = uniqueLabels;

result = alg.run(testing);

classified = uniqueLabels(result.getData());

disp('Classified')

%% Writing Output File 

formatSpec = '%s\t%s\n';

fid = fopen(outFile,'w');


for i = 1:nFiles
    fprintf(fid,formatSpec,wavPaths{i},classified{i});
end

fclose(fid);

end



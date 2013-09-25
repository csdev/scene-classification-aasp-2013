function classifyScene2(trainingDataFile,testDataFile,outFile)

%% Path and Label Import

disp('Importing Data')

fid = fopen(trainingDataFile);
t = textscan(fid, '%s %s');
fclose(fid);

wavPaths = t{1};
labels = t{2};
nFiles = length(wavPaths);

fid = fopen(testDataFile);
t = textscan(fid, '%s %s');
fclose(fid);

testPaths = t{1};
testFiles = length(testPaths);

disp('Importing Complete')


%% Configuration

dnsamp = 4;

shortFrameLength = 512;
shortFrameOverlap = 256;
shortWindow = hamming(shortFrameLength);

longFrameLength = 16384;
longFrameOverlap = longFrameLength - shortFrameOverlap;
longWindow = hamming(longFrameLength);
nShortFramesInLongFrame = longFrameLength / shortFrameLength * 2 - 1;


%% Feature Selection

disp('Extracting Features')

features = struct;
testFeatures = struct;

for fileNum = 1:nFiles
    % Read in audio data, convert to single channel, then downsample
    [rawData, Fs] = wavread(wavPaths{fileNum});
    a = (rawData(:, 1) + rawData(:, 2))./2;
    a = downsample(a.', dnsamp).';

    shortFrames = buffer(a, shortFrameLength, shortFrameOverlap, 'nodelay');
    nShortFrames = size(shortFrames, 2);
    shortFrames = shortFrames .* repmat(shortWindow, 1, nShortFrames);

    longFrames = buffer(a, longFrameLength, longFrameOverlap, 'nodelay');
    nLongFrames = size(longFrames, 2);
    nDupLongFrames = nShortFrames - nLongFrames;
    longFrames = longFrames .* repmat(longWindow, 1, nLongFrames);
    longFrames = [repmat(longFrames(:, 1), 1, nShortFrames - nLongFrames), longFrames];

    linearLoudness = rms(shortFrames, 1);
    loudness = 20*log10(linearLoudness + 1e-9);
    loudnessAvg = mean(loudness);
    loudnessStd = std(loudness);

    freqRes = fft(shortFrames, shortFrameLength, 1);
    magRes = mag2db(abs(freqRes));

    specSparsity = max(magRes) ./ sum(magRes, 1);

    tempSparsity = zeros(1, nShortFrames);
    for i = nDupLongFrames + 1 : nShortFrames
        x = linearLoudness(i - nShortFramesInLongFrame + 1 : i);
        tempSparsity(i) = max(x) / sum(x);
    end
    tempSparsity(1:nDupLongFrames) = tempSparsity(nDupLongFrames + 1);

    % Prepare feature matrices
    % Each row is a feature; each column is a frame
    data = [loudness; magRes(1:shortFrameLength/2, :); specSparsity; tempSparsity];

    features(fileNum, 1).data = data.';
end

for fileNum = 1:testFiles
    % Read in audio data, convert to single channel, then downsample
    [rawData, Fs] = wavread(testPaths{fileNum});
    a = (rawData(:, 1) + rawData(:, 2))./2;
    a = downsample(a.', dnsamp).';

    shortFrames = buffer(a, shortFrameLength, shortFrameOverlap, 'nodelay');
    nShortFrames = size(shortFrames, 2);
    shortFrames = shortFrames .* repmat(shortWindow, 1, nShortFrames);

    longFrames = buffer(a, longFrameLength, longFrameOverlap, 'nodelay');
    nLongFrames = size(longFrames, 2);
    nDupLongFrames = nShortFrames - nLongFrames;
    longFrames = longFrames .* repmat(longWindow, 1, nLongFrames);
    longFrames = [repmat(longFrames(:, 1), 1, nShortFrames - nLongFrames), longFrames];

    linearLoudness = rms(shortFrames, 1);
    loudness = 20*log10(linearLoudness + 1e-9);
    loudnessAvg = mean(loudness);
    loudnessStd = std(loudness);

    freqRes = fft(shortFrames, shortFrameLength, 1);
    magRes = mag2db(abs(freqRes));

    specSparsity = max(magRes) ./ sum(magRes, 1);

    tempSparsity = zeros(1, nShortFrames);
    for i = nDupLongFrames + 1 : nShortFrames
        x = linearLoudness(i - nShortFramesInLongFrame + 1 : i);
        tempSparsity(i) = max(x) / sum(x);
    end
    tempSparsity(1:nDupLongFrames) = tempSparsity(nDupLongFrames + 1);

    % Prepare feature matrices
    % Each row is a feature; each column is a frame
    data = [loudness; magRes(1:shortFrameLength/2, :); specSparsity; tempSparsity];

    testFeatures(fileNum, 1).data = data.';
end

clearvars -except features labels testFeatures outFile testPaths testFiles

disp('Extraction Complete')


%% Prepare DataSet

disp('Training')

ds = prtDataSetClassMultipleInstance;
ds = ds.setX(features);
[classes, ~, uniqueLabels] = getClassIds(labels);
ds = ds.setY(classes);
ds.classNames = uniqueLabels;


%% Classifier

% nstd = prtOutlierRemovalNStd;
zmuv = prtPreProcZmuv;
pca = prtPreProcPca('nComponents', 25);

svm = prtClassLibSvm;
svm.cachesize = 1000;

c = prtClassBinaryToMaryOneVsAll;
c.baseClassifier = svm;

alg = zmuv + pca + c;

classifier = prtClassMajorityVote;
classifier.baseClassifier = alg;
classifier.rejectLowLikelihoodInstances = true;

classifier = classifier.train(ds);


%% Classifying

disp('Classifying')

test = prtDataSetClassMultipleInstance;
test = test.setX(testFeatures);

result = classifier.run(test);
classified = uniqueLabels(result.getData());


disp('Classified')

%% Writing Output File 

formatSpec = '%s\t%s\n';

fid = fopen(outFile,'w');


for i = 1:testFiles
    fprintf(fid,formatSpec,testPaths{i},classified{i});
end

fclose(fid);

%transfer learning methodology
Datapath = fullfile ('MATLAB', 'PlantTom');
%Store in datastore
imageObject = imageDatastore (Datapath,'IncludeSubfolders',true,'LabelSource','foldernames') ;
%Resize image to match alexnet input
imageObject.ReadSize = numpartitions(imageObject);
imageObject.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
%Parse data
numTrainingFiles = 0.65;
[imTrain,imTest] = splitEachLabel(imageObject,numTrainingFiles,'randomize');
%Display sample images
numTrainImages = numel(imTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imTrain,idx(i));
    imshow(I)
end

%Load alexnet
net = alexnet;
%Transfer the layers to the net except for the last 3
layersTransfer = net.Layers(1:end-3);
%Identify classes to distinguish
numClasses = numel(categories(imTrain.Labels))
%Transfer last three layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
%Shift pixels to deter overfitting

%Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
   ..
    'Shuffle','every-epoch', ...
    'ValidationData',imTest, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
%Train algorithm
netTransfer = trainNetwork(imTrain,layers,options);

%classify algorithm
[YPred,scores] = classify(netTransfer,imTest);
YValidation = imTest.Labels;
accuracy = mean(YPred == YValidation)
plotconfusion (YPred, YValidation)
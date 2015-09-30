function main( varargin )
%MAIN Summary of this function goes here
%   Detailed explanation goes here
gpuDevice( 1 ); reset( gpuDevice ); clear ans;
setpaths;
% global msetting

run(msetting.vlsetuppath);
% addpath(msetting.matconvnetpath);
% addpath(fullfile(msetting.matconvnetpath ,'matlab'));

opts.modelType = 'vgg-m' ;
opts.networkType = 'simplenn' ; % no dag yet. please use simplenn
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
% opts.expDir = msetting.resactrgbpath;
opts.expDir = fullfile(msetting.resultpath,'OPmerge_rand_0.001_mean_trainall');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.sequencenum = msetting.sequencenum;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = msetting.imdbpath;
opts.train.batchSize = 35 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = msetting.continue;
opts.train.gpus = msetting.gpus ;
opts.train.prefetch = false ;
opts.train.sync = false ;
opts.train.cudnn = true ;
opts.train.expDir = opts.expDir ;
if ~opts.batchNormalization
    
  %opts.train.learningRate = [0.01*ones(1, 35) 0.001*ones(1,100) 0.0001*ones(1,300)]*0.1;%logspace(-3, -5, 60) ;
  opts.train.learningRate = [0.01*ones(1, 10) 0.001*ones(1,20) 0.0001*ones(1,30)]*0.1;%logspace(-3, -5, 60) ;
else
  opts.train.learningRate = logspace(-1, -4, 20) ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;


imdb = dbmaker();

imdb.ifpath = imdb.ifpathopti;
imdb.label = imdb.labelact;
imdb.labeluniq = imdb.labeluniqact;
imdb.seqlabel = imdb.seqlabelact;
imdb.unitseqlabel = imdb.unitseqlabelact;

if msetting.trainingmethod == 1
    % divide db into train and val
    trainlog = cell2mat(imdb.seqsets);
    trainlog = (trainlog == 1);
    trainlog = find(trainlog);
    opts.train.train = trainlog(randperm(length(trainlog)));
    vallog = cell2mat(imdb.seqsets);
    vallog = (vallog == 2);
    opts.train.val = find(vallog);
elseif msetting.trainingmethod == 2
    % divide db into train and val
    trainlog = cell2mat(imdb.unitseqsets);
    trainlog = (trainlog == 1);
    trainlog = find(trainlog);
    opts.train.train = trainlog(randperm(length(trainlog)));
    vallog = cell2mat(imdb.unitseqsets);
    vallog = (vallog == 2);
    opts.train.val = find(vallog);
end
% debugvideo(imdb)
  
net = initialize_network(imdb, 'model', opts.modelType, ...
                        'batchNormalization', opts.batchNormalization, ...
                        'weightInitMethod', opts.weightInitMethod, ...
                        'labelsize', length(imdb.labeluniq));                    
                    
if ~isempty(msetting.meanimpath)
    msetting.meanimage = load(msetting.meanimpath);
    %msetting.meanimage = single(im2uint8(msetting.meanimage.avgim));
    msetting.meanimage = single(msetting.meanimage.meanim);
end
if ~isempty(net.normalization.averageImage)
    msetting.meanimage = net.normalization.averageImage;
end

net.normalization.averageImage = msetting.meanimage;
bopts = net.normalization ;
bopts.numThreads = opts.numFetchThreads ;
bopts.sequencenum = opts.sequencenum;

% from imagenet example
switch lower(opts.networkType)
  case 'simplenn'
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1error') ;
  otherwise
    error('Unknown netowrk type ''%s''.', opts.networkType) ;
end

if msetting.mergelayers
    opts.train.errorFunction = 'merge';
else
    opts.train.errorFunction = 'multiclass';
end

fn = getBatchSimpleNNWrapper(bopts) ;
[net,info] = cnn_train_CAD120(net, imdb, fn, opts.train, 'conserveMemory', true) ;

end


function debugvideo(imdb)

    figure(2)
    for ind = 1 : length(imdb.sequences)
        curseq = imdb.sequences{ind};
        for subind = 1 : length(curseq)
            curnum = curseq(subind);
            curlab = imdb.seqlabel{ind};
             
            imagesc(imread(imdb.ifpath{curnum}));
            title(imdb.labeluniq{curlab});
            pause(0.05)
        end
    end

end

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;
end
% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
global msetting

    if msetting.trainingmethod == 1

        for i = 1 : length(batch)
            bb = imdb.sequences{batch(i)};
            currandnum = randi(length(bb)-opts.sequencenum + 1,1);
            images = imdb.ifpath(bb(currandnum:currandnum+opts.sequencenum-1));

            imtemp = get_batch_CAD120(images, opts, ...
                'imageSize', [224, 224], ...
                'border' , [32,117]) ;

            im(:,:,:,i) = imtemp(:,:,:);

        end

        % from here, for custom layer -> softmax merge
        if msetting.mergelayers
            labelact = single(cell2mat(imdb.seqlabelact(batch)));
            labelobj = single(cell2mat(imdb.seqlabelobj(batch)));
            labels = cat(1,labelact,labelobj);
        else
            labels = cell2mat(imdb.seqlabel(batch));
            labels = single(labels);
        end
        % to here
    
    elseif msetting.trainingmethod == 2

        for i = 1 : length(batch)
            bb = imdb.unitsequence{batch(i)};
            images = imdb.ifpath(bb);

            imtemp = get_batch_CAD120(images, opts, ...
                'imageSize', [224, 224], ...
                'border' , [32,117]) ;

            im(:,:,:,i) = imtemp(:,:,:);

        end

        % from here, for custom layer -> softmax merge
        if msetting.mergelayers
            labelact = single(cell2mat(imdb.unitseqlabelact(batch)));
            labelobj = single(cell2mat(imdb.unitseqlabelobj(batch)));
            labels = cat(1,labelact,labelobj);
        else
            labels = cell2mat(imdb.unitseqlabel(batch));
            labels = single(labels);
        end
        % to here        
    end
    
    if 0 %%debug
        figure(2)
        for i = 1 : opts.sequencenum
            curlab = labels(i);
            title(imdb.labeluniq{curlab});
            showingim = reshape(im(:,:,:,i),[224,224,3,opts.sequencenum]);
            implay(uint8(showingim));
            waitforbuttonpress;
        end
    end
end


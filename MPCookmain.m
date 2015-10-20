function MPCookmain( varargin )
%MAIN Summary of this function goes here
%   Detailed explanation goes here
gpuDevice( 1 ); reset( gpuDevice ); clear ans;
MPsetpaths;
% global msetting

addpath(msetting.jsonpath);
run(msetting.vlsetuppath);
% addpath(msetting.matconvnetpath);
% addpath(fullfile(msetting.matconvnetpath ,'matlab'));

opts.modelType = 'vgg-m' ;
opts.networkType = 'simplenn' ; % no dag yet. please use simplenn
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
% opts.expDir = msetting.resactrgbpath;
opts.expDir = fullfile(msetting.resultpath,'RGB_pre_0.005_f5_mean_train3_5');
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
    
  opts.train.learningRate = [0.01*ones(1, 5) 0.001*ones(1,25) 0.0001*ones(1,20)]*0.5;%logspace(-3, -5, 60) 
%   opts.train.learningRate = [0.01*ones(1, 5) 0.001*ones(1,10) 0.0001*ones(1,15)]*0.1;%logspace(-3, -5, 60) ;
%   opts.train.learningRate = logspace(-3, -6, 40) ;
else
  opts.train.learningRate = logspace(-1, -4, 20) ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;


annos = MPdbmaker();

annos.imgfoldpath = msetting.RGBimpath;

if msetting.trainingmethod == 1
    % divide db into train and val
    trainlog = cell2mat(annos.set);
    trainlog = (trainlog == 1);
    trainlog = find(trainlog);
    opts.train.train = trainlog(randperm(length(trainlog)));
    vallog = cell2mat(annos.set);
    vallog = (vallog == 2);
    opts.train.val = find(vallog);
    
elseif msetting.trainingmethod == 2
    
    % divide db into train and val
    trainlog = cell2mat(annos.extset);
    trainlog = (trainlog == 1);
    trainlog = find(trainlog);
    opts.train.train = trainlog(randperm(length(trainlog)));
    vallog = cell2mat(annos.extset);
    vallog = (vallog == 2);
    opts.train.val = find(vallog);
    
elseif msetting.trainingmethod == 3
    
    % divide db into train and val
    trainlog = cell2mat(annos.mulset);
    trainlog = (trainlog == 1);
    trainlog = find(trainlog);
    opts.train.train = trainlog(randperm(length(trainlog)));
    vallog = cell2mat(annos.mulset);
    vallog = (vallog == 2);
    opts.train.val = find(vallog);
    
end
% debugvideo(imdb)


if msetting.extractvector %% extract all
    
    load(msetting.trainedpath);
    
    for i = 3 : -1 : 0, net.layers{end-i} = {};, end
    net.layers = net.layers(~cellfun('isempty',net.layers));
    
    bopts = net.normalization ;
    bopts.numThreads = opts.numFetchThreads ;
    bopts.sequencenum = opts.sequencenum;
    prevind = 0;

    % make prevpathname as 1
    foldpath = annos.imgfoldpath;
    
    for ind = 13078 : length(annos.startFrame)
        tic;
        
        curstart = annos.startFrame(ind);
        curend = annos.endFrame(ind) - 1;
        labelact = find(annos.actMat(ind,:));
        labelobj = annos.objMat(ind,:);
        curset = annos.set{ind};
        data2json{1} = struct('labact',labelact,'labobj',labelobj,'set',curset);
        for cind = curstart : curend-opts.sequencenum+1
            
            images = [];
            for imind = cind : cind+opts.sequencenum-1
                images = [images;{sprintf('%s/%d/%d.jpg',foldpath,ind,imind)}];
            end
            
            im = get_batch_MP(images, bopts, ...
                'imageSize', [224, 224], ...
                'border' , [32,117], ...
                'transformation','none') ;
            
            res = '';
            res = vl_simplenn(net, im, [], res, ...
                'disableDropout', true, ...
                'conserveMemory', false, ...
                'sync', false, ...
                'cudnn', true);
            
            ffl = res(end).x;
            ffl = squeeze(ffl);
            
            data2json{cind-curstart+2}=struct('fnum',cind,'activ',ffl);
        end
        savejson('data',data2json,struct('FloatFormat','%.4f'),struct('FileName',sprintf('jsonsMP/%d.json',ind)));
        data2json = {};
        t = toc;
        s = dir(sprintf('jsonsMP/%d.json',ind));
        fprintf('now %d / %d, len = %d, time = %.2fs, filesize = %.2fMB\n',ind,length(annos.startFrame),annos.endFrame(ind)-annos.startFrame(ind),t, s.bytes/1000000);
    end

    return;
else

    net = initialize_network_MP(annos, 'model', opts.modelType, ...
                            'batchNormalization', opts.batchNormalization, ...
                            'weightInitMethod', opts.weightInitMethod);                    

    if ~isempty(msetting.meanimpath)
        msetting.meanimage = load(msetting.meanimpath);
        %msetting.meanimage = single(im2uint8(msetting.meanimage.avgim));
        msetting.meanimage = single(msetting.meanimage.meanim);
    end
    if ~isempty(net.normalization.averageImage)
        msetting.meanimage = net.normalization.averageImage;
    end

    
    if 1 % override meanim to path meanim
        msetting.meanimage = load(msetting.meanimpath);
        %msetting.meanimage = single(im2uint8(msetting.meanimage.avgim));
        msetting.meanimage = single(msetting.meanimage.meanim); 
    end
    
    net.normalization.averageImage = msetting.meanimage;
    bopts = net.normalization ;
    bopts.numThreads = opts.numFetchThreads ;
    bopts.sequencenum = opts.sequencenum;
    bopts.prefetch = opts.train.prefetch;
    
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
%         opts.train.errorFunction = 'merge';
        opts.train.errorFunction = 'mergeMP';
    else
        opts.train.errorFunction = 'multiclass';
    end

    fn = getBatchSimpleNNWrapper(bopts) ;
    [net,info] = cnn_train_MP(net, annos, fn, opts.train, 'conserveMemory', true) ;

end

end
% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;
end
% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(annos, batch, opts)
% -------------------------------------------------------------------------
global msetting

    if msetting.trainingmethod == 1

        for i = 1 : length(batch)
            curnum = batch(i);
            curstart = annos.startFrame(curnum);
            curend = annos.endFrame(curnum) - 1;
            
            lenF = curend - curstart + 1;
            if (lenF < opts.sequencenum)
                errid = fopen('errorlog.txt','a');
                fprintf(errid,'length below seqnum 10 at ind %d',curnum);
                fclose(errid);
            end
            currandnum = randi(lenF-opts.sequencenum + 1,1);
            
            foldpath = annos.imgfoldpath;
            images = [];
            for imind = 0 : opts.sequencenum-1
                images = [images;{sprintf('%s/%d/%d.jpg',foldpath,curnum,currandnum+imind+curstart)}];
            end

            imtemp = get_batch_MP(images, opts, ...
                'imageSize', [224, 224], ...
                'border' , [32,117],...
                'transformation', msetting.transform) ;

            im(:,:,:,i) = imtemp(:,:,:);

        end

        % from here, for custom layer -> softmax merge
        if msetting.mergelayers
            [i,j] = find(annos.actMat(batch,:));
            [~,ord] = sort(i);
            
            labelact = single(j(ord));
            labelobj = annos.objMat(batch,:);
            labels = cat(2,labelact,labelobj);
        else
            labels = single(find(annos.actMat(batch,:)));
        end
        % to here
    
    elseif msetting.trainingmethod == 2
        
        for i = 1 : length(batch)
            curnum = batch(i);
            curstart = annos.extstartFrame(curnum);
            curend = annos.extendFrame(curnum) - 1;
            curfold = annos.extfold(curnum);
            
            foldpath = annos.imgfoldpath;
            images = [];
            for imind = curstart : curend
                images = [images;{sprintf('%s/%d/%d.jpg',foldpath,curfold,imind)}];
            end

            imtemp = get_batch_MP(images, opts, ...
                'imageSize', [224, 224], ...
                'border' , [32,117],...
                'transformation', msetting.transform) ;

            im(:,:,:,i) = imtemp(:,:,:);

        end

        % from here, for custom layer -> softmax merge
        if msetting.mergelayers
            [i,j] = find(annos.extactMat(batch,:));
            [~,ord] = sort(i);
            
            labelact = single(j(ord));
            labelobj = annos.extobjMat(batch,:);
            labels = cat(2,labelact,labelobj);
        else
            labels = single(find(annos.extactMat(batch,:)));
        end
        % to here   
        
    elseif msetting.trainingmethod == 3
        for i = 1 : length(batch)
            if annos.mulset{batch(i)} == 2
                curnum = annos.mulfold(batch(i));
                curstart = annos.mulvalstart(batch(i));
                
                foldpath = annos.imgfoldpath;
                images = [];
                for imind = 0 : opts.sequencenum-1
                    images = [images;{sprintf('%s/%d/%d.jpg',foldpath,curnum,imind+curstart)}];
                end
                
                imtemp = get_batch_MP(images, opts, ...
                    'imageSize', [224, 224], ...
                    'border' , [32,117],...
                    'transformation', 'none') ;
                im(:,:,:,i) = imtemp(:,:,:);
                
            else
                curnum = annos.mulfold(batch(i));
                curstart = annos.startFrame(curnum);
                curend = annos.endFrame(curnum) - 1;

                lenF = curend - curstart + 1;
                if (lenF < opts.sequencenum)
                    errid = fopen('errorlog.txt','a');
                    fprintf(errid,'length below seqnum 10 at ind %d',curnum);
                    fclose(errid);
                end
                currandnum = randi(lenF-opts.sequencenum + 1,1);

                foldpath = annos.imgfoldpath;
                images = [];
                for imind = 0 : opts.sequencenum-1
                    images = [images;{sprintf('%s/%d/%d.jpg',foldpath,curnum,currandnum+imind+curstart)}];
                end

                imtemp = get_batch_MP(images, opts, ...
                    'imageSize', [224, 224], ...
                    'border' , [32,117],...
                    'transformation', msetting.transform) ;

                im(:,:,:,i) = imtemp(:,:,:);
            end
        end

        % from here, for custom layer -> softmax merge
        if msetting.mergelayers
            batch = annos.mulfold(batch);
            
            [i,j] = find(annos.actMat(batch,:));
            [~,ord] = sort(i);
            
            labelact = single(j(ord));
            labelobj = annos.objMat(batch,:);
            labels = cat(2,labelact,labelobj);
        else
            batch = annos.mulfold(batch);
            labels = single(find(annos.actMat(batch,:)));
        end
    end
    
    if 0 %%debug
        figure(2)
        for i = 1 : length(batch)
            curlab = labels(i,1);
            title(annos.actName{curlab});
            showingim = reshape(im(:,:,:,i),[224,224,3,opts.sequencenum]);
            implay(uint8(showingim));
            waitforbuttonpress;
        end
    end
end


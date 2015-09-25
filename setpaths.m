global msetting

msetting.currentpath = '/media/disk1/bgsim/Subactivity_Multistream';
msetting.src         = 'src';
msetting.srcpath     = fullfile(msetting.currentpath,msetting.src);
addpath(genpath(msetting.srcpath));

%% training method
msetting.trainingmethod  = 1; % 1 : a sequence per epoch, 2 : every sequences per epoch
msetting.sequencenum     = 10;
msetting.continue        = true ;
msetting.gpus            = [1] ;
msetting.mergelayers     = false;

%% training result paths
msetting.resultpath  = fullfile(msetting.currentpath,'res');
msetting.resactrgb   = 'actrgb';
msetting.resactopti  = 'actopti';
msetting.resactdt    = 'actdt';
msetting.resobjrgb   = 'objrgb';
msetting.resobjopti  = 'objopti';
msetting.resobjdt    = 'objdt';

msetting.resactrgbpath   = fullfile(msetting.resultpath,msetting.resactrgb);
msetting.resactoptipath  = fullfile(msetting.resultpath,msetting.resactopti);
msetting.resactdtpath    = fullfile(msetting.resultpath,msetting.resactdt);
msetting.resobjrgbpath   = fullfile(msetting.resultpath,msetting.resobjrgb);
msetting.resobjoptipath  = fullfile(msetting.resultpath,msetting.resobjopti);
msetting.resobjdtpath    = fullfile(msetting.resultpath,msetting.resobjdt);

%% network msetting
msetting.pretrainnetwork = ''; %'imagenet-vgg-m.mat'


%% matconvnet, vlfeat path
msetting.matconvnetpath  = fullfile(msetting.currentpath,'matconvnet-1.0-beta14');
msetting.vlsetuppath     = fullfile(msetting.matconvnetpath, 'matlab/vl_setupnn.m') ;
msetting.vlfeatpath      = '/media/disk1/bgsim/vlfeat/';
msetting.vlfeatsetuppath = fullfile(msetting.vlfeatpath, 'toolbox/vl_setup.m');

%% image paths
msetting.DBname = 'CAD120';
msetting.RGBimpath = '/media/disk1/bgsim/action/dataset/objjpgimages';
msetting.dtimpath = '/media/disk1/bgsim/action/dataset/dtim256';
msetting.ofimpath = '/media/disk1/bgsim/action/dataset/thres5optijpgimages';

%% annotation paths
msetting.annotationpath = '/media/disk1/bgsim/action/dataset/objectannotation';

%% divide train with validation method 
msetting.dividemethod = 2; % 1: random, 2 : by subject (1,2,3 -> train, 5 -> validation)
msetting.trainingratio = 0.8;

%% mean image paths
msetting.meanimage       = [];
msetting.meanRGBimpath   = 'rgbavgimage.mat';
msetting.meanOPimpath    = '';
msetting.meanDTimpath    = '';
msetting.meanDEPTHimpath = '';
msetting.meanimpath      = msetting.meanRGBimpath;

%% image db path
msetting.imdbname = 'imdb.mat';
msetting.imdbpath = fullfile(msetting.currentpath,msetting.imdbname);

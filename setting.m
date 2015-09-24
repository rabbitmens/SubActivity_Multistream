global setting

setting.currentpath = '/media/disk1/bgsim/Subactivity_Multistream';
setting.src         = 'src';
setting.srcpath     = fullfile(setting.currentpath,setting.src);
addpath(genpath(setting.srcpath));

%% training method
setting.trainingmethod  = 1; % 1 : a sequence per epoch, 2 : every sequences per epoch
setting.sequencenum     = 10;

%% matconvnet, vlfeat path
setting.matconvnetpath  = fullfile(setting.currentpath,'matconvnet');
setting.vlfeatpath      = '/media/disk1/bgsim/vlfeat/';

%% image paths
setting.DBname = 'CAD120'
setting.RGBimpath = '/media/disk1/bgsim/action/dataset/objjpgimages';
setting.dtimpath = '/media/disk1/bgsim/action/dataset/dtim256';
setting.ofimpath = '/media/disk1/bgsim/action/dataset/thres5optijpgimages';

%% annotation paths
setting.annotationpath = '/media/disk1/bgsim/action/dataset/objectannotation';

%% divide train with validation method 
setting.dividemethod = 2; % 1: random, 2 : by subject (1,2,3 -> train, 5 -> validation)
setting.trainingratio = 0.8;

%% mean image paths
setting.meanRGBimpath   = '';
setting.meanOPimpath    = '';
setting.meanDTimpath    = '';
setting.meanDEPTHimpath = '';

%% image db path
setting.imdbname = 'imdb.mat'
setting.imdbpath = fullfile(setting.currentpath,setting.imdbname);

%% training result paths
setting.resultpath  = fullfile(setting.currentpath,'res');
setting.resactrgb   = 'actrgb';
setting.resactopti  = 'actopti';
setting.resactdt    = 'actdt';
setting.resobjrgb   = 'objrgb';
setting.resobjopti  = 'objopti';
setting.resobjdt    = 'objdt';

setting.resactrgbpath   = fullfile(setting.resultpath,setting.resactrgb);
setting.resactoptipath  = fullfile(setting.resultpath,setting.resactopti);
setting.resactdtpath    = fullfile(setting.resultpath,setting.resactdt);
setting.resobjrgbpath   = fullfile(setting.resultpath,setting.resobjrgb);
setting.resobjoptipath  = fullfile(setting.resultpath,setting.resobjopti);
setting.resobjdtpath    = fullfile(setting.resultpath,setting.resobjdt);


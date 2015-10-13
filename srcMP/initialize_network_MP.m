function [ net ] = initialize_network_MP( annos, varargin )
global msetting

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.model = 'vgg-m' ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = false ;
opts = vl_argparse(opts, varargin) ;

mergelayer = msetting.mergelayers;

if ~isempty(msetting.pretrainnetwork)
    net = load(msetting.pretrainnetwork);
    % first, have to change single image input to sequence input by
    % msetting.sequencenum
    net.layers{1,1}.weights{1} = repmat(net.layers{1,1}.weights{1},[1,1,msetting.sequencenum]);
    
    % last, have to change size of last fully-connected layer into our
    % label size
    lastlayersize = size(net.layers{end-1}.weights{1});
    if mergelayer == true
        lastlayersize(end) = length(annos.actName) + length(annos.objName)*2;
    else
        lastlayersize(end) = length(annos.actName);
    end
    net.layers{end-1}.weights{1} = init_weight(opts, lastlayersize(1), lastlayersize(2),...
                                                lastlayersize(3), lastlayersize(4), 'single'); %random
    
    lastlayersize = size(net.layers{end-1}.weights{2});
    if mergelayer == true
        lastlayersize(end) = length(annos.actName) + length(annos.objName)*2;
    else
        lastlayersize(end) = length(annos.actName);
    end
    net.layers{end-1}.weights{2} = zeros(lastlayersize, 'single'); 
    
%     net.layers{end} = struct('type', 'softmaxloss', 'name', 'loss') ;
    if mergelayer == true
        net = lastcustomlayer(net,length(annos.actName),length(annos.objName));
        net.classes.description = cat(2,annos.actName,annos.objName);
        net.classes.name = cat(2,annos.actName,annos.objName);
    else
        net.layers{end} = struct('type', 'softmaxloss', 'name', 'loss') ;
        net.classes.description = annos.actName;
        net.classes.name = annos.actName;
    end
    
else
    % vgg-m network
    opts.sequencenum = msetting.sequencenum;
    opts.labelsize = 0;
    if mergelayer == true
        opts.labelsize = length(annos.actName) + length(annos.objName)*2;
    else
        opts.labelsize = length(annos.actName);
    end
    net.normalization.imageSize = [224, 224, 3] ;
    net = vgg_m(net, opts) ;
    
    % final touches
    switch lower(opts.weightInitMethod)
      case {'xavier', 'xavierimproved'}
        net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
    end
    
    net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
    if mergelayer == true
        net = lastcustomlayer(net, length(annos.actName),length(annos.objName));
    end
    net.normalization.border = 256 - net.normalization.imageSize(1:2) ;
    net.normalization.interpolation = 'bicubic' ;
    net.normalization.averageImage = [] ;
    net.normalization.keepAspect = true ;
end

end

function net = lastcustomlayer(net, numact, numobj)

    net.layers{end}.type = 'custom';
    net.layers{end}.dimact = 1 : numact;
    net.layers{end}.numact = numact;
    net.layers{end}.numobj = numobj;
    net.layers{end}.forward = @forward;
    net.layers{end}.backward = @backward;

end
% bgsim 151008 branch softmaxloss into act and obj with multiclass
function res2 = forward( layer, res1, res2)
    X = res1.x;
    gt = layer.class;
    numLayer = 2;
    numobj = layer.numobj;
    numact = layer.numact;
    yact = vl_nnsoftmaxloss(X( :, :, layer.dimact, :), gt(:,1));
%     yobj = vl_nnsoftmaxloss(X( :, :, layer.dimobj, :), gt(2,:));

    yobj = 0;
    for i = 1 : numobj
        curgt = gt(:,i+1);
        curgt = ~curgt + 1; % 0-> false =2, 1 -> true = 1.
        yobji = vl_nnsoftmaxloss(X( :, :, ((i*2-1)+numact):(i*2+numact), :), curgt);
        yobj = yobj + yobji;
    end
    yobj = yobj / numobj;

    res2.x = (yact+yobj) / numLayer;
end
function res1 = backward( layer, res1, res2)
    X = res1.x;
    gt = layer.class;
    numLayer = 2;
    numact = layer.numact;
    numobj = layer.numobj;
    dzdyacts = res2.dzdx / numLayer;
    dzdyobjs = res2.dzdx / numLayer;
    dzdyobjs = dzdyobjs / layer.numobj;
    yact = vl_nnsoftmaxloss(X(:,:,layer.dimact,:),gt(:,1),dzdyacts);
    ytot = yact;
    for i = 1 : numobj
        curgt = gt(:,i+1);
        curgt = ~curgt + 1; % 0-> false =2, 1 -> true = 1.
        yobji = vl_nnsoftmaxloss(X( :, :, ((i*2-1)+numact):(i*2+numact), :), curgt, dzdyobjs);
        ytot = cat(3,ytot,yobji);
    end
    res1.dzdx = ytot;%cat(3,yact,yobj);
end
% bgsim end

% --------------------------------------------------------------------
function net = vgg_m(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
net = add_block(net, opts, '1', 7, 7, 3 * opts.sequencenum, 96, 2, 0) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '2', 5, 5, 96, 256, 2, 1) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net = add_block(net, opts, '3', 3, 3, 256, 512, 1, 1) ;
net = add_block(net, opts, '4', 3, 3, 512, 512, 1, 1) ;
net = add_block(net, opts, '5', 3, 3, 512, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;

net = add_block(net, opts, '6', 6, 6, 512, 4096, 1, 0) ;
net = add_dropout(net, opts, '6') ;

net = add_block(net, opts, '7', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, '7') ;

net = add_block(net, opts, '8', 1, 1, 4096, opts.labelsize, 1, 0) ;
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single')}}, ...
                             'learningRate', [2 1], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end
end
% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end
end
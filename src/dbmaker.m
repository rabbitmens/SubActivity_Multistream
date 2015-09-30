function [ imdb ] = dbmaker( )

global msetting

%% initialize imdb
imdb.ifpath         = {};
imdb.ifpathrgb      = {};
imdb.ifpathdepth    = {};
imdb.ifpathopti     = {};
imdb.ifpathdt       = {};
imdb.ifset          = {};

imdb.dividemethod   = msetting.dividemethod;

imdb.label          = {};
imdb.labelact       = {};
imdb.labelobj       = {};

imdb.labelname      = {};
imdb.labelnameact   = {};
imdb.labelnameobj   = {};
imdb.labeluniq      = {};
imdb.labeluniqact   = {};
imdb.labeluniqobj   = {};

imdb.sequences      = {};
imdb.seqlabel       = {};
imdb.seqlabelact    = {};
imdb.seqlabelobj    = {};
imdb.seqsets        = {};

imdb.unitsequence   = {};
imdb.unitseqlabel   = {};
imdb.unitseqlabelact= {};
imdb.unitseqlabelobj= {};
imdb.unitseqsets    = {};

imdb.meanimage      = '';
imdb.meanimagergb   = msetting.meanRGBimpath;
imdb.meanimageopti  = msetting.meanOPimpath;


%% 
imdb = load(msetting.imdbpath);
if length(imdb.ifpathrgb) ~= length(imdb.ifpathopti)
    imdb.ifpathrgb = jpgfinder(msetting.RGBimpath);
    imdb.ifpathopti = jpgfinder(msetting.ofimpath);
    % imdb.ifpathdt = jpgfinder(msetting.dtimpath);
    [imdb.ifpathrgb, imdb.ifpathopti] = matching(imdb.ifpathrgb, imdb.ifpathopti);
    % matching(imdb.ifpathrgb, imdb.ifpathdt , 'matchrgbdt');
end

if length(imdb.ifpathrgb) ~= length(imdb.labelnameact)
    fprintf('%d, %d',length(imdb.ifpathrgb),length(imdb.ifpathopti))
    imdb = makelabelcad(imdb, msetting.annotationpath, 'labeling.txt');
end

if length(imdb.ifset) ~= length(imdb.ifpathrgb)
    imdb = dividetrainval(imdb);
end
%         imdb.sequences = {};
%         imdb.seqlabelact = {};
%         imdb.seqlabelobj = {};
%         imdb.seqsets     = {};
if isempty(imdb.sequences)
    imdb = makesequences(imdb,msetting.sequencenum);
end

if isempty(imdb.unitsequence)
    imdb = makeunitsequences(imdb, msetting.sequencenum);
end

save( msetting.imdbpath, '-struct', 'imdb') ;

end

function [ imdb ] = makeunitsequences(imdb, sequencenum)

    for ind = 1 : length(imdb.ifpathrgb) - sequencenum + 1

        prevlabel = imdb.labelact{ind};
        spl = strsplit(imdb.ifpathrgb{ind},'/');
        prevrep = spl{end-1};
        
        issequence = true;
        
        for num = 1 : sequencenum - 1
            curlabel = imdb.labelact{ind+num};
            spl = strsplit(imdb.ifpathrgb{ind+num},'/');
            currep = spl{end-1};
            
            if (prevlabel ~= curlabel) || (~strcmp(prevrep,currep))
                issequence = false;
                break;
            end
        end
        
        if issequence
            seq = ind : (ind+sequencenum-1);
            imdb.unitsequence{end+1}   = seq;
            imdb.unitseqlabelact{end+1} = imdb.labelact{ind};
            imdb.unitseqlabelobj{end+1} = imdb.labelobj{ind};
            imdb.unitseqsets{end+1}     = imdb.ifset{ind};
        end
    end

end

function [ imdb ] = makesequences(imdb, sequencenum)

    curseq = [];
    ind = 1;
    prevlabel = imdb.labelact{ind};
    spl = strsplit(imdb.ifpathrgb{ind},'/');
    prevrep = spl{end-1};
    
    for ind = 1 : length(imdb.ifpathrgb)
       
        curlabel = imdb.labelact{ind};
        spl = strsplit(imdb.ifpathrgb{ind},'/');
        currep = spl{end-1};
        
        if (curlabel == prevlabel) && (strcmp(prevrep,currep))
            curseq = [curseq, ind];
        else
            if length(curseq) < sequencenum
                prevlabel = curlabel;
                prevrep = currep;
                curseq = [ind];
            continue;
            end
            imdb.sequences{end+1} = curseq;
            imdb.seqlabelact{end+1} = imdb.labelact{ind-1};
            imdb.seqlabelobj{end+1} = imdb.labelobj{ind-1};
            imdb.seqsets{end+1}     = imdb.ifset{ind-1};
            prevlabel = curlabel;
            prevrep = currep;
            curseq = [ind];
        end
    end
    if length(curseq) >= sequencenum
        imdb.sequences{end+1} = curseq;
        imdb.seqlabelact{end+1} = imdb.labelact{end};
        imdb.seqlabelobj{end+1} = imdb.labelobj{end};
        imdb.seqsets{end+1}     = imdb.ifset{end};
    end

%     for ind = 1 : length(imdb.ifpathrgb) - sequencenum + 1
% 
%         prevlabel = imdb.labelact{ind};
%         spl = strsplit(imdb.ifpathrgb{ind},'/');
%         prevrep = spl{end-1};
%         
%         issequence = true;
%         
%         for num = 1 : sequencenum - 1
%             curlabel = imdb.labelact{ind+num};
%             spl = strsplit(imdb.ifpathrgb{ind+num},'/');
%             currep = spl{end-1};
%             
%             if (prevlabel ~= curlabel) || (~strcmp(prevrep,currep))
%                 issequence = false;
%                 break;
%             end
%         end
%         
%         if issequence
%             seq = ind : (ind+sequencenum-1);
%             imdb.sequences{end+1}   = seq;
%             imdb.seqlabelact{end+1} = imdb.labelact{ind};
%             imdb.seqlabelobj{end+1} = imdb.labelobj{ind};
%             imdb.seqsets{end+1}     = imdb.ifset{ind};
%         end
%     end

end


function [ imdb ] = indexinglabel(imdb)
    imdb.labeluniqact = unique(imdb.labelnameact);
    imdb.labeluniqobj = unique(imdb.labelnameobj);
    
    for ind = 1 : length(imdb.labelnameact)
        labact = find(strcmp(imdb.labelnameact{ind},imdb.labeluniqact));
        labobj = find(strcmp(imdb.labelnameobj{ind},imdb.labeluniqobj));
        imdb.labelact{ind} = labact;
        imdb.labelobj{ind} = labobj;
    end
end

function [ imdb ] = dividetrainval(imdb)
global msetting
    imdb.dividemethod = msetting.dividemethod;
    if msetting.dividemethod == 1
       prevlabel = '';
       prevrep = '';
       prevset = 1;
       for ind = 1 : length(imdb.ifpathrgb)
           curlabel = imdb.labelnameact{ind};
           spl = strsplit(imdb.ifpathrgb{ind},'/');
           currep = spl{end-1};
           
           if strcmp(prevlabel,curlabel) && strcmp(prevrep,currep)
               imdb.ifset{ind} = prevset; 
           else
               prevlabel = curlabel;
               prevrep = currep;
               prevset = (rand(1) > msetting.trainingratio) + 1;
               imdb.ifset{ind} = prevset; 
           end
       end
        
    elseif msetting.dividemethod == 2
        for ind = 1 : length(imdb.ifpathrgb)
            spl = strsplit(imdb.ifpathrgb{ind},'/');
            subj = spl{end-3};
            imdb.ifset{ind} = (strcmpi(subj,'subject5') + 1);
        end
    end
end

function [ imdb ] = trimlabel(imdb)

    emptycells = (~cellfun('isempty',imdb.labelnameact));
    imdb.ifpathrgb = imdb.ifpathrgb(emptycells);
    imdb.ifpathopti = imdb.ifpathopti(emptycells);
    imdb.labelnameact = imdb.labelnameact(emptycells);
    imdb.labelnameobj = imdb.labelnameobj(emptycells);

end

function [ imdb ] = makelabelcad(imdb, annopath, filename)
    labels = {};
    subfold = dir(annopath);
    for subind = 3 : length(subfold)
        actpath = fullfile(annopath,subfold(subind).name);
        actfold = dir(actpath);
        for actind = 3 : length(actfold)
            labpath = fullfile(actpath,actfold(actind).name,filename);
                    
            fid = fopen(labpath);
            line = fgets(fid);
            while ischar(line)
                line = strtrim(line);
                spl = strsplit(line,',');
                curlab = [{subfold(subind).name}, {actfold(actind).name}, spl(1), spl(2),spl(3),spl(4),spl(end)];
                labels{end+1} = curlab;
                line = fgets(fid);
            end
        end
    end
    
    rgbpathsplit = {};
    for ind = 1 : length(imdb.ifpathrgb)
        curpath = imdb.ifpathrgb{ind};
        spl = strsplit(curpath,'/');
        subject = spl{end-3};
        action = spl{end-2};
        replica = spl{end-1};
        curnum = strsplit(spl{end},{'.','_'});
        curnum = str2double(curnum{2});
        cursplit = [subject,action,replica,curnum];
        rgbpathsplit{ind} = cursplit;
    end
    
    for labind = 1 : length(labels)
        curlab = labels{labind}; 
        startnum = str2double(curlab{4});
        endnum = str2double(curlab{5});
        actlabel = curlab{6};
        objlabel = curlab{7};
        for curnum = startnum : endnum
            cursplit = [curlab{1},curlab{2},curlab{3},curnum];
            curind = find(strcmp(cursplit,rgbpathsplit));
            if isempty(curind)
                continue;
            end
            imdb.labelnameact{curind} = actlabel;
            imdb.labelnameobj{curind} = objlabel;
        end
    end
    
    imdb = trimlabel(imdb);
    imdb = indexinglabel(imdb);
%     labind = 1;
%     ind = 1;
%     while ind <= length(imdb.ifpathrgb) && labind <= length(labels)
%         if ind == 64402
%             disp('now')
%         end
%         curpath = imdb.ifpathrgb{ind};
%         spl = strsplit(curpath,'/');
%         curlab = labels{labind};
%         submatch = strcmp(spl{end-3},curlab{1});
%         actmatch = strcmp(spl{end-2},curlab{2});
%         repmatch = strcmp(spl{end-1},curlab{3});
%         startnum = str2double(curlab{4});
%         endnum = str2double(curlab{5});
%         actlabel = curlab{6};
%         objlabel = curlab{7};
%         curnum = strsplit(spl{end},{'.','_'});
%         curnum = str2double(curnum{2});
%         if (submatch && repmatch && actmatch) || (curnum == 1)
%         else
%            % case if frame remains but no annotation exists
%            ind = ind + 1;
%            continue;
%         end
%         if curnum >= startnum && curnum <= endnum
%             imdb.labelnameact{ind} = actlabel;
%             imdb.labelnameobj{ind} = objlabel;
%         else
%             ind = ind - 1;
%             labind = labind + 1;
%         end
%         ind = ind + 1;
%     end
end

function [ respath1, respath2 ] = matching ( path1, path2, DB )
    if nargin < 3
       DB = 'CAD120'; 
    end
    
    if strcmp(DB,'CAD120')
        comppath = path1; tocomp = path2;
        if length(path2) < length(path1)
            comppath = path2; tocomp = path1;
        end
        toind = 1;
        ind = 1;
        while ind <= length(comppath)
            spl1 = strsplit(tocomp{toind},'/');
            spl2 = strsplit(comppath{ind},'/');
            
            if strcmp(spl1(end),spl2(end)) && strcmp(spl1(end-1),spl2(end-1)) && strcmp(spl1(end-2),spl2(end-2)) && strcmp(spl1(end-3),spl2(end-3))
                
            else
                tocomp{toind} = [];
                ind = ind -1;
            end
            ind = ind + 1;
            toind = toind + 1;
        end
        for toind = toind : length(tocomp)
            tocomp{toind} = [];
        end
        % netrgbact.net.layers =netrgbact.net.layers(~cellfun('isempty',netrgbact.net.layers));
        if length(path2) < length(path1)
            respath2 = comppath; respath1 = tocomp(~cellfun('isempty',tocomp));
        else
            respath1 = comppath; respath2 = tocomp(~cellfun('isempty',tocomp));
        end
    end
end

function [ paths ] = jpgfinder ( directory )

    folds = dir(directory);
    founds = {};
    foundlow = {};
    for ind = 3 : length(folds)
        newpath = fullfile(directory,folds(ind).name);
        if isdir(newpath)
            newfound = jpgfinder(newpath);
            foundlow = {foundlow{:}, newfound{:}};
        else
            spls = strsplit(newpath,'.');
            if strcmpi(spls{end},'jpg') || strcmpi(spls{end},'jpeg')
                founds{end+1} = newpath;
            end
        end
    end
    paths = {foundlow{:},founds{:}};
end
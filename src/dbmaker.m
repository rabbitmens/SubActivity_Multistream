function [ imdb ] = dbmaker( )

global setting

%% initialize imdb
imdb.ifpath         = {};
imdb.ifpathrgb      = {};
imdb.ifpathdepth    = {};
imdb.ifpathopti     = {};
imdb.ifpathdt       = {};
imdb.ifset          = {};

imdb.dividemethod   = setting.dividemethod;

imdb.label          = {};
imdb.labelact       = {};
imdb.labelobj       = {};

imdb.labelname      = {};
imdb.labelnameact   = {};
imdb.labelnameobj   = {};

imdb.sequences      = {};
imdb.seqlabel       = {};
imdb.seqlabelact    = {};
imdb.seqlabelobj    = {};
imdb.seqsets        = {};

imdb.unitsequence   = {};
imdb.unitseqlabel   = {};
imdb.unitseqlabelact= {};
imdb.unitseqlabelobj= {};

imdb.meanimage      = '';
imdb.meanimagergb   = setting.meanRGBimpath;
imdb.meanimageopti  = setting.meanOPimpath;


%% 
imdb = load(setting.imdbpath);
if length(imdb.ifpathrgb) ~= length(imdb.ifpathopti)
    imdb.ifpathrgb = jpgfinder(setting.RGBimpath);
    imdb.ifpathopti = jpgfinder(setting.ofimpath);
    % imdb.ifpathdt = jpgfinder(setting.dtimpath);
    [imdb.ifpathrgb, imdb.ifpathopti] = matching(imdb.ifpathrgb, imdb.ifpathopti);
    % matching(imdb.ifpathrgb, imdb.ifpathdt , 'matchrgbdt');
end

if length(imdb.ifpathrgb) ~= length(imdb.labelnameact)
    fprintf('%d, %d',length(imdb.ifpathrgb),length(imdb.ifpathopti))
    imdb = makelabelcad(imdb, setting.annotationpath, 'labeling.txt');
end

if length(imdb.ifset) ~= length(imdb.ifpathrgb)
    imdb = dividetrainval(imdb);
end

save( setting.imdbpath, '-struct', 'imdb') ;

end

function [ imdb ] = dividetrainval(imdb)
global setting
    imdb.dividemethod = setting.dividemethod;
    if setting.dividemethod == 1
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
               prevset = (rand(1) > setting.trainingratio) + 1;
               imdb.ifset{ind} = prevset; 
           end
       end
        
    elseif setting.dividemethod == 2
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
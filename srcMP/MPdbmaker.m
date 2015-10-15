function [ annos ] = MPdbmaker( )

global msetting

annos = load(msetting.imdbpath);

%% divide attributes into activity and others (object)

uniqact = unique(annos.activity);
attplace = {};
for ind = 1 : length(uniqact)
    [i,j] = find(strcmp(annos.attributeMap,uniqact{ind}));
    attplace{ind} = j;
end
remact = uniqact(~cellfun('isempty',attplace));
rematt = cell2mat(attplace(~cellfun('isempty',attplace)));
actmat = annos.iaMat(:,rematt);
summat = sum(actmat,2);
[~,exceptions] = find(summat'==0); % will split none=0, train=1, val=2, test=3. exceptions will be none.

annos.actName = remact;
annos.actPlace = rematt;
annos.actMat = actmat;

objplace = ones(1,size(annos.iaMat,2));
objplace(rematt)=0;
[~,objatt] = find(objplace);
objmat = annos.iaMat(:,objatt);
summat = sum(objmat,2);
[~,exceptobj] = find(summat'==0); % will split none=0, train=1, val=2, test=3. exceptions will be none.
exceptions = [exceptions, exceptobj];

annos.objName = annos.attributeMap(objatt);
annos.objPlace = objatt;
annos.objMat = objmat;
annos.exceptions = exceptions;

% exceptions from act = 115, obj = 276, total = 391. 
% exception rate = 391/14105 = 0.0277

%% split train, val, test, none. none means exceptions that has no label on activity or object.

splitfoldpath = msetting.splitpath;
trainfile = 'subjectsTrainAttr.txt';
valfile = 'subjectsVal.txt';
testfile = 'subjectsTest.txt';

fid = fopen(fullfile(splitfoldpath,trainfile),'r');
x = fread(fid,'*char'); x = strsplit(x','\n');
trains = cell2mat(cellfun(@str2num,x,'un',0));

fid = fopen(fullfile(splitfoldpath,valfile),'r');
x = fread(fid,'*char'); x = strsplit(x','\n');
vals = cell2mat(cellfun(@str2num,x,'un',0));

fid = fopen(fullfile(splitfoldpath,testfile),'r');
x = fread(fid,'*char'); x = strsplit(x','\n');
tests = cell2mat(cellfun(@str2num,x,'un',0));

annos.set = {};
for ind = 1 : length(annos.subject)
    cursub = annos.subject(ind);
    if find(exceptions==ind)
        annos.set{ind} = 0;
    elseif find(trains==cursub)
        annos.set{ind} = 1;
    elseif find(vals==cursub)
        annos.set{ind} = 2;
    elseif find(tests==cursub)
        annos.set{ind} = 3;
    else
        fprintf('something wrong at %d\n',ind);
    end
    
    lenF = (annos.endFrame(ind) -1) -annos.startFrame(ind) +1; %%% -1 from endFrame becauseof Opticalflow, +1 to obtain length
    if(lenF < 10)
        annos.set{ind} = 0;
    end
end

if ~isfield(annos,'extset')
    annos = extendsequences(annos,msetting.sequencenum);
end

if ~isfield(annos,'mulset')
    annos = multiplysequences(annos,msetting.sequencenum,msetting.multiply);
end

save( msetting.imdbpath, '-struct', 'annos') ;

end
function annos = multiplysequences(annos,sequencenum,multiply)

    annos.mulset = {};
    annos.mulfold = [];
    annos.mulvalstart = [];
    
    for ind = 1 : length(annos.set)
        if rem(ind,100)==0
            fprintf('multiple sequences, now %d / %d\n',ind,length(annos.startFrame));
        end
        
        if annos.set{ind} == 0
            continue;
        end
        
        flen = (annos.endFrame(ind)-annos.startFrame(ind));
        
        if annos.set{ind} == 1
            mullen = flen/sequencenum;
            if mullen > multiply, mullen = multiply;, end
            
            for i = 1 : mullen
                annos.mulset{end+1} = 1;
                annos.mulfold(end+1) = ind;
            end
            
        elseif annos.set{ind} == 2
            flen = flen + 1 - sequencenum + 1;
            for i = 1 : flen
                annos.mulset{end+1} = 2;
                annos.mulfold(end+1) = ind;
                annos.mulvalstart(length(annos.mulset)) = i+annos.startFrame(ind)-1;
            end
        end
        
    end

end
function annos = extendsequences(annos,sequencenum)

    annos.extstartFrame = [];
    annos.extendFrame = [];
    annos.extactMat = [];
    annos.extobjMat = [];
    annos.extset = {};
    annos.extfold = [];
    
    multiple = msetting.multiply;
    
    for ind = 1 : length(annos.startFrame)
        if rem(ind,100)==0
            fprintf('extend sequences, now %d / %d\n',ind,length(annos.startFrame));
        end
        
        if annos.set{ind} == 0
           continue;
        end       
        
        curactMat = annos.actMat(ind,:);
        curobjMat = annos.objMat(ind,:);
        curset = annos.set{ind};
        
        if (annos.endFrame(ind)-annos.startFrame(ind))/sequencenum < multiple
            curF = annos.startFrame(ind);
            while (curF <= annos.endFrame(ind)-sequencenum+1)
                annos.extstartFrame(end+1) = curF;
                annos.extendFrame(end+1) = curF+sequencenum-1;
                annos.extactMat = [annos.extactMat ; curactMat];
                annos.extobjMat = [annos.extobjMat ; curobjMat];
                annos.extset{end+1} = curset;
                annos.extfold(end+1) = ind;
                curF = curF+10;
            end
        else
            curend = annos.endFrame(ind);
            cursta = annos.startFrame(ind);
            
            curlen = curend-cursta;
            
            iternum = 1;
            maxiternum = 10;
            currand = [];
            while iternum<maxiternum

                currand = [currand,randi(curlen-sequencenum+1,1,multiple-length(currand))];
                currand = sort(currand);
                idx = 2;
                while idx <= length(currand)
                    dif = currand(idx) - currand(idx-1);
                    if dif<sequencenum
                        currand(idx) = [];
                        continue;
                    end
                    idx = idx + 1;
                end
                iternum = iternum + 1;
                if length(currand) == multiple
                    break;
                end
            end
            currand = [currand,randi(curlen-sequencenum+1,1,multiple-length(currand))];
            currand = sort(currand);
            
            for idx = 1 : length(currand)
                annos.extstartFrame(end+1) = currand(idx)+cursta-1;
                annos.extendFrame(end+1) = currand(idx)+sequencenum-1+cursta-1;
                annos.extactMat = [annos.extactMat ; curactMat];
                annos.extobjMat = [annos.extobjMat ; curobjMat];
                annos.extset{end+1} = curset;
                annos.extfold(end+1) = ind;
            end
        end
    end
end

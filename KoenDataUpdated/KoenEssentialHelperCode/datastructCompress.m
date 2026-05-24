% function datastructCompress(filename)
% Compress Koen datastructs
% 
%
%%
data = load(filename);
% remove casig and casigraw
for i = 1:length(data.datastructPost)
    data.datastructPost(i).Res = rmfield(data.datastructPost(i).Res, 'CaSig');
    data.datastructPost(i).Res = rmfield(data.datastructPost(i).Res, 'CaSigRaw');
    
    data.datastructPre(i).Res = rmfield(data.datastructPre(i).Res, 'CaSig');
    data.datastructPre(i).Res = rmfield(data.datastructPre(i).Res, 'CaSigRaw');
    
    if ~all(all(all(diff(data.datastructPre(4).runSpeed, [], 3)==0)))
        fprintf('not all values in 3rd dimension were the same!\n')
        i
    end
    if ~all(all(all(diff(data.datastructPost(4).runSpeed, [], 3)==0)))
        fprintf('not all values in 3rd dimension were the same!\n')
        i
    end
    data.datastructPre(i).runSpeed(:,:,2:end) = [];
    data.datastructPost(i).runSpeed(:,:,2:end) = [];
end
clearvars i
saveName = [filename, '_essentials.mat'];
save(saveName, '-struct', 'data', '-v7.3')
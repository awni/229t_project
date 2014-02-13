function plot_results(suffix)
    addpath Simple_tSNE
    
    if ~exist('suffix','var') || isempty(suffix),
        suffix = [];
    end

    grad = load(['grad' suffix '.txt']);
    grad=bsxfun(@minus,grad, mean(grad));
    cost = load(['cost' suffix '.txt']);
    param = load(['param' suffix '.txt']);
    points_per_trial = 1000;
    
    h = figure;
    title(['results' suffix]);
    
    %% PCA approach
%     [u,s,v]=svds(cov(grad),2);
%     disp(diag(s));
%     t = cost';
%     tmp=param*v;
%     tmp = [tmp t(:)];
% 
%     hold on;
%     f = 1;
%     for i = 1:floor(size(tmp,1)/points_per_trial),
%         x=tmp((i-1)*points_per_trial+1:i*points_per_trial,:);
%         mesh([x(:,f),x(:,f)],[x(:,f+1),x(:,f+1)],[x(:,end),x(:,end)]);
%     end

    %% tSNE approach
    
    tmp = tsne(param);
    hold on;
    for i = 1:floor(size(tmp,1) / points_per_trial),
        mesh([tmp(:,1),tmp(:,1)], [tmp(:,2),tmp(:,2)], [cost(:),cost(:)]);
    end

    drawnow;
    
    saveas(h, ['results' suffix '.fig']);
end
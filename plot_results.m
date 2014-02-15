function plot_results(suffix)
    addpath Simple_tSNE
    
    if ~exist('suffix','var') || isempty(suffix),
        suffix = [];
    end

    grad = load(['grad' suffix '.txt']);
    grad = bsxfun(@minus, grad, mean(grad));
    cost = load(['cost' suffix '.txt']);
    param = load(['param' suffix '.txt']);
    points_per_trial = 100;
    
%     % downsample
%     param = param(1:10:end,:);
%     grad = grad(1:10:end,:);
%     cost = cost(:,1:10:end);
    
    h = figure;
    
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
    cost = cost';
    cost = cost(:);
    for i = 1:floor(size(tmp,1) / points_per_trial),
        idx = (i-1)*points_per_trial+1:i*points_per_trial;
        mesh([tmp(idx,1),tmp(idx,1)], [tmp(idx,2),tmp(idx,2)], [cost(idx),cost(idx)]);
    end

    drawnow;
    
    saveas(h, ['cost' suffix '.fig']);
    
    set(gcf,'PaperPosition',[0 0.1 5 4]); set(gcf,'PaperSize',[5 4.1]); print(gcf, ['cost' suffix '.pdf'],'-r200','-dpdf'); 
end
function plot_results(suffix)
    addpath Simple_tSNE
    
    if ~exist('suffix','var') || isempty(suffix),
        suffix = [];
    end
    
    h = figure;
    
    if ~iscell(suffix),

        grad = load(['grad' suffix '.txt']);
        grad = bsxfun(@minus, grad, mean(grad));
        cost = load(['cost' suffix '.txt']);
        param = load(['param' suffix '.txt']);
        points_per_trial = 100;



        %% PCA approach
        [u,s,v]=svds(cov(grad),2);
        disp(diag(s));
        t = cost';
        tmp=grad*v;
        tmp = [tmp t(:)];
    
        hold on;
        f = 1;
        for i = 1:floor(size(tmp,1)/points_per_trial),
            x=tmp((i-1)*points_per_trial+1:i*points_per_trial,:);
            mesh([x(:,f),x(:,f)],[x(:,f+1),x(:,f+1)],[x(:,end),x(:,end)]);
        end

        %% tSNE approach

%         tmp = tsne(param);
%         hold on;
%         cost = cost';
%         cost = cost(:);
%         for i = 1:floor(size(tmp,1) / points_per_trial),
%             idx = (i-1)*points_per_trial+1:i*points_per_trial;
%             mesh([tmp(idx,1),tmp(idx,1)], [tmp(idx,2),tmp(idx,2)], [cost(idx),cost(idx)]);
%         end

        drawnow;

        saveas(h, ['grad' suffix '.fig']);

        set(h,'PaperPosition',[0 0.1 7 5]); set(h,'PaperSize',[7 5.1]); print(h, ['grad' suffix '.pdf'],'-r200','-dpdf'); 
    else
        color = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1; 1 0 1; 0 0 0];
        N = length(suffix);
        grad=zeros(); cost=[]; param=[];
        points_per_trial = 100;
        trials=20;
        
        ppf = points_per_trial * trials;    % points per file
        
        for i = 1:N,
            gnew = load(['grad' suffix{i} '.txt']);
            cnew = load(['cost' suffix{i} '.txt']);
            pnew = load(['param' suffix{i} '.txt']);
            
            cost = [cost; cnew];
            
            if (size(pnew,2)>size(param,2)),
                pt = zeros((i)*ppf,size(pnew,2));
                pt(1:(i-1)*ppf,1:size(param,2)) = param;
                param = pt;
                param((i-1)*ppf+1:(i)*ppf,1:size(pnew,2)) = pnew;
                
                gt = zeros((i)*ppf,size(gnew,2));
                gt(1:(i-1)*ppf,1:size(grad,2)) = grad;
                grad = gt;
                grad((i-1)*ppf+1:(i)*ppf,1:size(gnew,2)) = gnew;
            else
                param((i-1)*ppf+1:(i)*ppf,1:size(pnew,2)) = pnew;
                grad((i-1)*ppf+1:(i)*ppf,1:size(gnew,2)) = gnew;
            end
        end
        grad = bsxfun(@minus, grad, mean(grad));
        
        cost = cost(:,1:2:end);
        param = param(1:2:end,:);
        points_per_trial=50;
        
        tmp = tsne(grad);
        hold on;
        cost = cost';
        cost = cost(:);
        for i = 1:floor(size(tmp,1) / points_per_trial),
            idx = (i-1)*points_per_trial+1:i*points_per_trial;
%             c = bsxfun(@times,color(ceil(i/trials),:), (cost(idx)-min(cost(idx)))./(max(cost(idx))-min(cost(idx))));
            mesh([tmp(idx,1),tmp(idx,1)], [tmp(idx,2),tmp(idx,2)], ones(points_per_trial,2)*ceil(i/trials));
        end

        drawnow;
        
        s = ['[' suffix{1} ']'];
        for i = 2:N,
            s = [s '[' suffix{i} ']'];
        end

        set(h,'PaperPosition',[0 0.1 7 5]); set(h,'PaperSize',[7 5.1]); print(h, ['grad' s '.pdf'],'-r200','-dpdf'); 
        saveas(h, ['grad' s '.fig']);
    end
end
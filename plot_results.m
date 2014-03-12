function plot_results(suffix)
    addpath Simple_tSNE
    
    if ~exist('suffix','var') || isempty(suffix),
        suffix = [];
    end
    
%     h = figure;
    
    if ~iscell(suffix),

        grad = load(['grad' suffix '.txt']);
        %grad = bsxfun(@minus, grad, mean(grad));
        cost = load(['cost' suffix '.txt']);
        param = load(['param' suffix '.txt']);
        points_per_trial = 100;
        
        fprintf('Num param: %d\n', size(grad,2));
        
        num_trials = floor(size(param,1) / points_per_trial);
        
%         t = zeros(points_per_trial, size(grad,2));
%         for i = 1:num_trials,
%             idx = (i-1)*points_per_trial+1:i*points_per_trial;
%             t = t + param(idx, :).^2;
%         end
%         
%         subplot(2,1,1);
%         imagesc((param)'); colorbar;
%         subplot(2,1,2);
%         plot(mean(log(t),2));
        
        
%         drawnow;
% 
%         saveas(h, ['param' suffix '.fig']);
% 
%         set(h,'PaperPosition',[0 0.1 10 7]); set(h,'PaperSize',[10 7.1]); print(h, ['param' suffix '.pdf'],'-r200','-dpdf'); 
        
        h = figure;

        t = zeros(points_per_trial, size(grad,2));
        for i = 1:num_trials,
            idx = (i-1)*points_per_trial+1:2:i*2*points_per_trial;
            t = t + grad(idx, :).^2;
        end
        
        t2 = zeros(size(t));
        t2(1,:) = t(1,:);
        for i = 2:points_per_trial,
            t2(i,:) = 1*t(i,:)+.9*t2(i-1,:); 
        end
        
%         temp=bsxfun(@rdivide, abs(grad), sum(abs(grad)));
%         temp=sum(temp.*log(temp+1e-6));
%         
%         plot(temp);
%         
%         saveas(h, ['ppt0.fig']);
% 
%         set(h,'PaperPosition',[0 0.1 10 5]); set(h,'PaperSize',[10 5.1]); print(h, ['ppt0.pdf'],'-r600','-dpdf'); 
% 
%         h = figure;

        
        imagesc(grad'); colorbar; 
        load ppt1cm;
        colormap(ppt1cm);
        ylabel('Param #'); xlabel('Training Iteration');
        title('Gradient trace');
        saveas(h, ['ppt1.fig']);

        set(h,'PaperPosition',[0 0.1 10 5]); set(h,'PaperSize',[10 5.1]); print(h, ['ppt1.pdf'],'-r600','-dpdf'); 
        
        h = figure;
        imagesc(cumsum(t',2)); colorbar;
        ylabel('Param #'); xlabel('Training Iteration');
        title('Adagrad denominator');
        saveas(h, ['ppt2.fig']);

        set(h,'PaperPosition',[0 0.1 10 5]); set(h,'PaperSize',[10 5.1]); print(h, ['ppt2.pdf'],'-r600','-dpdf'); 
        
        h = figure;
        imagesc(t2');  colorbar;
        ylabel('Param #'); xlabel('Training Iteration');
        title('Ada-delta denominator');
        saveas(h, ['ppt3.fig']);

        set(h,'PaperPosition',[0 0.1 10 5]); set(h,'PaperSize',[10 5.1]); print(h, ['ppt3.pdf'],'-r600','-dpdf'); 
        
        
%         drawnow;
% 
%         saveas(h, ['grad' suffix '.fig']);
% 
%         set(h,'PaperPosition',[0 0.1 10 5]); set(h,'PaperSize',[10 7.1]); print(h, ['grad' suffix '.pdf'],'-r200','-dpdf'); 
        
%         h = figure;
%         
%         t = zeros(points_per_trial, 1);
%         for i = 1:num_trials,
%             idx = (i-1)*points_per_trial+1:2:i*2*points_per_trial;
%             t = t + sqrt(sum((grad(idx+1, :)-grad(idx,:)).^2,2) ./ sum((grad(idx,:)).^2,2));
%         end
%         
%         plot(log(t));
%         
%         drawnow;
%         saveas(h, ['second' suffix '.fig']);
%         set(h,'PaperPosition',[0 0.1 7 5]); set(h,'PaperSize',[7 5.1]); print(h, ['second' suffix '.pdf'],'-r200','-dpdf'); 
        
%         t = bsxfun(@rdivide, t, sqrt(sum(t.^2,2)));
%         angles = zeros(points_per_trial-1,1);
%         for i = 1:points_per_trial-1,
%             t0 = grad(i*2,:) - grad(i*2-1,:);
%             t1 = grad((i+1)*2,:) - grad((i+1)*2-1,:);
%             angles(i) = t0 * t1' / norm(t0,2) / norm(t1,2);
%         end
%         
%         h = figure;
%         plot(angles, '.k');
%         drawnow;
%         saveas(h, ['second_cosine' suffix '.fig']);
%         set(h,'PaperPosition',[0 0.1 10 7]); set(h,'PaperSize',[10 7.1]); print(h, ['second_cosine' suffix '.pdf'],'-r200','-dpdf'); 

        %% PCA approach
%         [u,s,v]=svds(cov(grad),2);
%         disp(diag(s));
%         t = cost';
%         tmp=grad*v;
%         tmp = [tmp t(:)];
%     
%         hold on;
%         f = 1;
%         for i = 1:floor(size(tmp,1)/points_per_trial),
%             x=tmp((i-1)*points_per_trial+1:i*points_per_trial,:);
%             mesh([x(:,f),x(:,f)],[x(:,f+1),x(:,f+1)],[x(:,end),x(:,end)]);
%         end

        %% tSNE approach

%         tmp = tsne(param);
%         hold on;
%         cost = cost';
%         cost = cost(:);
%         for i = 1:floor(size(tmp,1) / points_per_trial),
%             idx = (i-1)*points_per_trial+1:i*points_per_trial;
%             mesh([tmp(idx,1),tmp(idx,1)], [tmp(idx,2),tmp(idx,2)], [cost(idx),cost(idx)]);
%         end

        
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
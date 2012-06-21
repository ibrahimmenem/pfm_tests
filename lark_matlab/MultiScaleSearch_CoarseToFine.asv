function MultiScaleSearch_CoarseToFine(Q,T,F_Q,F_T,S,E,flag,alpha,target,blocksize)

% [PARAMETERS]
% smap   : saliency map
% blocksize : size of block
% thres : threshold for saliency

% [HISTORY]
% May 31, 2011 : created by Hae Jong

tic;
M_Q = size(Q,1); N_Q = size(Q,2);
M_T = size(T,1); N_T = size(T,2);

stepsize = min(N_Q,M_Q); % you can have finer stepsize by having below 0.6
for kkk = 1:20
    if max(M_T,N_T) < 400
        SC_t(kkk) = (400+stepsize*(kkk-15)*1.5)/400;
    elseif max(M_T,N_T) > 600
        SC_t(kkk) = (600+stepsize*(kkk-15)*1.5)/600;
    else
        SC_t(kkk) = (max(M_T,N_T)+stepsize*(kkk-15)*1.5)/max(M_T,N_T);
    end
end
SC_t = SC_t(SC_t>0.4); %0.1
SC_t = SC_t(SC_t<1.6); %1.6
SC = [0.3 0.5]; % coarse-to-fine search scale factor
space = 1; % sampling factor for feature matrix
interval = 1; % search at 1 pixel apart


fprintf('Multi-scale search.\n');
progress = 0;
 
for n = 1:length(SC_t)
    flags = flag;
    
    for m = 1:length(SC)
        progress = progress + 1/length(SC)*length(SC_t);
        if (progress > 0.025)
            progress = progress - 0.025;
            fprintf('.')
        end
        % Rescale query and query feature
        Qs = imresize(Q,SC(m));
        F_Qs = imresize(F_Q,SC(m));
        F_Q1 = F_Qs(1:space:end, 1:space:end,:);
        norm_FQ = norm(F_Q1(:),'fro');
        F_Ts = imresize(F_T,SC_t(n)*SC(m),'lanczos3');
        Ts1 = imresize(T,SC_t(n)*SC(m),'lanczos3');
        Ts{m,n} = zeros(size(Ts1));
        Ts{m,n} = Ts1;
        RMs{m,n} = zeros(size(Ts{m,n}));
        flags;
        f_max = zeros(blocksize(1),blocksize(2));
        for x = 1:blocksize(1)
            for y = 1:blocksize(2)
                % Remap the starting points and ending points of blocks
                % according to the scale.
                S1{x,y}.x = max(floor(S{x,y}.x*SC(m)*SC_t(n)),1);
                E1{x,y}.x = min(max(floor(E{x,y}.x*SC(m)*SC_t(n)),1),size(F_Ts,2)-size(F_Qs,2)+1);
                S1{x,y}.y = max(floor(S{x,y}.y*SC(m)*SC_t(n)),1);
                E1{x,y}.y = min(max(floor(E{x,y}.y*SC(m)*SC_t(n)),1),size(F_Ts,1)-size(F_Qs,1)+1);
                
                if E1{x,y}.y < S1{x,y}.y
                    temp = S1{x,y}.y;                    
                    S1{x,y}.y = E1{x,y}.y;
                    E1{x,y}.y = temp;                    
                end
                 if E1{x,y}.x < S1{x,y}.x
                    temp = S1{x,y}.x;                    
                    S1{x,y}.x = E1{x,y}.x;
                    E1{x,y}.x = temp;                    
                 end
                % Compute MCS only when the block is salient
                if flags(x,y) == 1
                    for i = S1{x,y}.y:interval:E1{x,y}.y
                        for j = S1{x,y}.x:interval:E1{x,y}.x
                            if i+size(F_Qs,1)-1 <= size(F_Ts,1) && j+size(F_Qs,2)-1 <= size(F_Ts,2)
                                F_T_i = F_Ts(i:space:i+size(F_Qs,1)-1, j:space:j+size(F_Qs,2)-1,:);
                                rho = F_Q1(:)'*F_T_i(:)/(norm_FQ*norm(F_T_i(:),'fro'));
                                RMs{m,n}(floor(size(F_Qs,1)/2)+i,floor(size(F_Qs,2)/2)+j) = (rho^2)/(1-rho^2);
                            end
                        end
                    end
                    
                    f = RMs{m,n}(floor(size(F_Qs,1)/2)+S1{x,y}.y:min(floor(size(F_Qs,1)/2)+E1{x,y}.y,size(RMs{m,n},1)),floor(size(F_Qs,2)/2)+S1{x,y}.x:min(floor(size(F_Qs,2)/2)+E1{x,y}.x,size(RMs{m,n},2)));       
                     
                    f_max(x,y) = max(f(:));
                   % if the maximum resemblance value in the block is smaller than the threshold, do not search in the finer scale       
                    if f_max(x,y) < 0.2
                        flags(x,y) = 0;
                    end
                    clear f;
                end
            end
        end
    end
    % perform significance testing at each scale
    [RM2,RM3] = stage3forMultiscale(RMs{length(SC),n},Qs,0.9);   % chkim 0.9? RMs{1:end, n} ?? 
    % rescale resemblance maps to the origianl target size
    RM2s(:,:,n) = imresize(RM2,[size(F_T,1),size(F_T,2)],'nearest');
    RM3s(:,:,n) = imresize(RM3,[size(F_T,1),size(F_T,2)],'nearest');
  
end

% 
 E_RM = max(RM2s,[],3);
[E_RM1,s_ind] = max(RM3s,[],3); % ML estimation of scale
% perform significance testing to the final resemblance
[E_RM2,RM3] = FinalStage3(E_RM1,Q(1:size(F_Q,1),1:size(F_Q,2)),T(1:size(F_T,1),1:size(F_T,2)),alpha,s_ind,SC_t,1);
disp(['Search time: ' num2str(toc) ' sec']);   

figure, sc(cat(3,imresize(E_RM,[size(target,1),size(target,2)]),double(target(:,:,1))),'prob_jet'); colorbar;
figure, sc(cat(3,imresize(E_RM2,[size(target,1),size(target,2)]),double(target(:,:,1))),'prob_jet'); colorbar;

end

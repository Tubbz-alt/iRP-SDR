% iRP-SDR (Hung, 08/01/2020)
%
% ----- INPUT -----
% y: n by 1 response vector
% x: n by p covariate matrix
% d_env_ub: upper bound of dim(S_env)
% N: no. random partitions
% sir_ind: 1 = SIR, 2 = DR, 3 = SAVE
% h: slicing number of SDR methods
%
% ----- OUTPUT -----
% Ku: integrated kernel matrix
% Ku_temp = Ku * cov(x)^-1;   

function [Ku, Ku_temp] = rp_sdr(y, x, d_env_ub, N, sdr_ind, h)
   
[n,p] = size(x);
% Sx = cov(x);
Ku_temp = zeros(p);   

d_env_data = 0;
for j = 1:N
    %%%%% random r %%%%%
    r = randsample(unique(floor(d_env_ub./[1:d_env_ub])), 1); 
    
    perm_ind = randperm(p);
    Benv_j = [];
    dc_j = [];
    for ii = 1:ceil(p/r)
        if ii < ceil(p/r)
            O_ind = sort( perm_ind(1+(ii-1)*r : ii*r) );
        else
            O_ind = sort( perm_ind(1+(ii-1)*r : end) );
        end
        dc = DC(y, x(:, O_ind));
        Benv_j = [Benv_j, O_ind];
        dc_j = [dc_j; dc*ones(length(O_ind),1)];
    end     
    d_env = length(Benv_j);
    
    Benv_j_drop = [];
    while d_env > d_env_ub    % upper bounded by d_env_ub
        Benv_j_drop_0 = Benv_j_drop;   % previous drop 
        min_dc_j = min(dc_j);
        Benv_j_drop = Benv_j(dc_j == min_dc_j);   % current drop 
        Benv_j = Benv_j(dc_j > min_dc_j);
        dc_j = dc_j(dc_j > min_dc_j);
        d_env = length(Benv_j);
    end
    
    diff = d_env_ub - d_env;  % fill up to d_env_ub
    if diff > 0  
        if length(Benv_j_drop) >= diff
            dc_drop = [];
            for jj = 1:length(Benv_j_drop)
                dc_drop(jj) = DC(y, x(:, Benv_j_drop(jj)));
            end
            [~,ind_back] = maxk(dc_drop, diff);
            Benv_j = [Benv_j, Benv_j_drop(ind_back)];
            d_env = length(Benv_j);
        else
            dc_drop_0 = [];
            for jj = 1:length(Benv_j_drop_0)
                dc_drop_0(jj) = DC(y, x(:, Benv_j_drop_0(jj)));
            end
            [~,ind_back] = maxk(dc_drop_0, diff-length(Benv_j_drop));
            Benv_j = [Benv_j, Benv_j_drop, Benv_j_drop_0(ind_back)];
            d_env = length(Benv_j);
            disp('Use drop_back_0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        end
    end

    d_env_data = [d_env_data, d_env];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if sdr_ind == 1 
        [B_j_temp, L_j] = dr_sir(x(:,Benv_j), y, d_env, h);
    elseif sdr_ind == 2
        [B_j_temp, L_j] = dr_dr(x(:,Benv_j), y, d_env, h); 
    elseif sdr_ind == 3
        [B_j_temp, L_j] = dr_save(x(:,Benv_j), y, d_env, h); 
    end
    B_j = zeros(p, d_env);
    B_j(Benv_j,:) = B_j_temp;  
    Ku_temp = Ku_temp + B_j*diag(L_j)*B_j';   
end
Ku = Ku_temp * cov(x);   
tabulate(d_env_data)
end


%%
function [dc] = DC(y, x)
    
n = size(x,1);
xx = x*x';
dx = diag(xx);
for  i = 1:n
    yi_temp = abs(y(i)-y);
    xi_temp = (dx(i) - 2*xx(:,i) + dx).^0.5; 

    yi(i) = sum(yi_temp);
    xi(i) = sum(xi_temp);
    xy_i(i) = sum(yi_temp.*xi_temp);    
end
S1 = sum(xy_i)/n^2;
S2 = sum(yi)*sum(xi)/n^4;
S3 = sum(yi.*xi)/n^3;

xx1 = 2*(mean(dx)-mean(x)*mean(x)');
xx2 = sum(xi)^2/n^4;
xx3 = sum(xi.^2)/n^3;

dc = (S1+S2-2*S3) / (xx1+xx2-2*xx3).^0.5;  % ignore dcov(Y,Y)
end



function [sir_dir, D] = dr_sir(K, y, d, NumOfSlice)

[n, p] = size(K);

% standardize
Kmean=mean(K,1);
s_K=cov(K)+10^-8*eye(p);
[uu, ss]=svd(s_K);
s_K_inv2=uu*ss^(-1/2)*uu';
z=(K-ones(n,1)*Kmean)*s_K_inv2; 

% slicing data
if (ischar(NumOfSlice))
else
    [~, Index] = sort(y);
    z = z(Index,:);   % z sorted by y
    n1=floor(n/NumOfSlice);  
    y=ones(n,1)*NumOfSlice;  % categorized y
    for i1 = 1:(NumOfSlice-1)
        y(((i1-1)*n1+1):(i1*n1),1) = i1;
    end
end
class = unique(y);
NumOfSlice = length(class);

G1=zeros(p,p);
for k = 1:NumOfSlice
    p_k=mean(y==class(k));
    smean=mean(z(y==class(k),:))';
    G1 = G1 + (smean*smean')*p_k;
end
[U, D] = svds(G1, d);
D = diag(D);  % leading eigenvalues (z-scale)
sir_dir = s_K_inv2*U;  % dr in original scale
end



function [dr_dir, D] = dr_dr(K, y, d, NumOfSlice)

[n, p] = size(K);

% standardize
Kmean=mean(K,1);
s_K=cov(K)+10^-8*eye(p);
[uu, ss]=svd(s_K);
s_K_inv2=uu*ss^(-1/2)*uu';
z=(K-ones(n,1)*Kmean)*s_K_inv2; 

% pre-process of data
if (ischar(NumOfSlice))
else
    [~, Index] = sort(y);
    z = z(Index,:);   % z sorted by y
    n1=floor(n/NumOfSlice);  
    y=ones(n,1)*NumOfSlice;   % categorized y
    for i1 = 1:(NumOfSlice-1)
        y(((i1-1)*n1+1):(i1*n1),1) = i1;
    end
end
class = unique(y);
NumOfSlice = length(class);

G1=zeros(p,p);
G2=zeros(p,p);
G3=zeros(1,1);
for k = 1:NumOfSlice
    p_k=mean(y==class(k));
    smean=mean(z(y==class(k),:))';
    smean2=z(y==class(k),:)'*z(y==class(k),:)/sum(y==class(k));

    G1 = G1 + (smean2-eye(p))^2*p_k;
    G2 = G2 + smean*smean'*p_k;
    G3 = G3 + smean'*smean*p_k;
end
G = 2*( G1+G2*G2+G3*G2 );  % kernel matrix of DR
[U, D] = svds(G, d);
D=diag(D);
dr_dir = s_K_inv2*U;
end


function [save_dir, D] = dr_save(K, y, d, NumOfSlice)

[n, p] = size(K);

% standardize
s_K=cov(K);
[uu, ss]=svd(s_K);
s_K_inv2=uu*ss^(-1/2)*uu';
z=(K-ones(n,1)*mean(K,1))*s_K_inv2; 

% pre-process of data
if (ischar(NumOfSlice))
else
    [sorty, Index] = sort(y);
    z = z(Index,:);          % z sorted by y
        
    m=floor(n/NumOfSlice);   % no. per slice
    r=n-NumOfSlice*m;
    y=[];
    for i= 1:r
        y=[y;i*ones(m+1,1)];   % allocate the rest to the first r slice
    end
    for i=r+1:NumOfSlice
        y=[y;i*ones(m,1)];
    end
end
class = unique(y);
NumOfSlice = length(class);

G1=zeros(p,p);
for k = 1:NumOfSlice
    ind=(y==k);
    G1 = G1 + mean(ind)*( eye(p) - cov(z(ind,:)) )^2;   %kernel matrix of SAVE (z-scale)
end
[U, D] = svds(G1, d);
D = diag(D);
save_dir = s_K_inv2*U;
end








% Post processing
clc;clear
BATCH_SIZE_b = 250;
BATCH_SIZE_i = 250;
BATCH_SIZE_e = 100000;
Scol = [];
k = 0;
index_b = [];
index_i = [];
index_e = [];
for t = 0:0.005:0.5
  for x = -0.005:0.005/100:0.005
      k = k + 1;
      disp(t)
      if t < 1e-6
          index_i = [index_i;k];
      elseif abs(x-5)<1e-8 || abs(x+5)<1e-8
          index_b = [index_b;k];
      else
          index_e = [index_e;k];
      end
      Scol = [Scol;t x];
  end
end
b_num = length(index_b); i_num = length(index_i); e_num = length(index_e);
% % if b_num>BATCH_SIZE_b
% %     error('b_num>BATCH_SIZE_b')
% % end
% % if i_num>BATCH_SIZE_i
% %     error('i_num>BATCH_SIZE_i')
% % end
% % if e_num>BATCH_SIZE_e
% %     error('e_num>BATCH_SIZE_e')
% % end

% S_Col = zeros(BATCH_SIZE_b + BATCH_SIZE_i + BATCH_SIZE_e, 2);
% S_Col(1:b_num, :) = Scol(index_b, :);
% S_Col((b_num+1):(b_num+i_num), :) = Scol(index_i, :);
% S_Col((b_num+i_num + 1):(b_num+i_num + e_num), :) = Scol(index_e, :);
S_Col = zeros(26000, 2);
S_Col(1:201*101,:) = Scol;
% save Scol.mat Scol
% save S_Col.mat S_Col
%%
N_p = size(Scol,1);
Sum_N = ceil(N_p/20100);

for i = 1:Sum_N
    endi = 20100;
    temp = zeros(20100,2);
    startno = (i-1)*20100 + 1;
    endno = startno + 20100-1;
    if i == Sum_N
        endno = startno + rem(N_p,20100) - 1;
        endi = rem(N_p,20100);
    end
    temp(1:endi,:) = Scol(startno:endno,:);
    eval(['save S_Col' num2str(i) '.mat temp'])
end

%%
% clc;clear
% load Scol.mat
load aa.mat
t = reshape(Scol(:,1),[201,101]);
x = reshape(Scol(:,2),[201,101]);
aa_ = zeros(201*101, 13);
aa_(:,:) = aa(1:201*101, :);

aa_c = [Scol aa_];
aa_c = sortrows(aa_c, [1, 2]);

u = reshape(aa_c(:,3),[201,101]);

contourf(t,x,u,500,'linewidth',0.00000)
%%
Col = 0;
%%
load Ua.mat
Ua(:, 1) = [];
kk = size(Ua,2);
Col = Col + kk;
end_no = Col/13;
start_no = (Col-kk)/13+1;
for i = start_no:end_no
    startno = (i-start_no)*13 + 1;
    endno = (i-start_no + 1)*13;
    aa_ = zeros(201*101, 13);
    aa_(:,:) = Ua(1:201*101, startno:endno);
    aa_c = [Scol aa_];
    aa_c = sortrows(aa_c, [1, 2]);
    u = reshape(aa_c(:,3),[201,101]);
    eval(['u' sprintf('%.2d', i) ' = u;'])
end
%%
UCOL=[];
for k = 1:1:200
   eval(['UCOL = [UCOL u' sprintf('%.2d', k) '(:, 101)];']) 
end
%%
aa = linspace(0,1,201)';
bb = aa;
for i=1:100
  aa = [aa bb];
end

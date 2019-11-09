%%
clear;clc
niu = 0.002;
N_sum = 500;
Scol = [];
k = 0;
index_b = [];
index_i = [];
index_e = [];
for t = 1:0.01:1
  for x = -1:0.01:1
      k = k + 1;
      disp(k)
      if t < 1e-6
          index_i = [index_i;k];
      elseif abs(x) < 1e-8
          index_b = [index_b;k];
      else
          index_e = [index_e;k];
      end
      SIGMA_UP = 0; SIGMA_DOWN = 0;
      for n = 1:N_sum
         SIGMA_UP = SIGMA_UP + exp(-niu*n*n*pi*pi*t)*n*besseli(n,-1/2/pi/niu)*sin(n*pi*x);
         SIGMA_DOWN = SIGMA_DOWN + exp(-niu*n*n*pi*pi*t)*besseli(n,-1/2/pi/niu)*cos(n*pi*x);
      end
      u = 4*niu*pi*SIGMA_UP/(besseli(0,-1/2/pi/niu)+2*SIGMA_DOWN);
      Scol = [Scol;t x u];
  end
end

t_ = reshape(Scol(:,1), [201,101]);
x_ = reshape(Scol(:,2), [201,101]);
u_ = reshape(Scol(:,3), [201,101]);
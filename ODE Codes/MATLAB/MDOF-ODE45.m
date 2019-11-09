%% MDOF equation -- ODE45

% the motion equation is:
% m*acc + c*vel + alpha*k*disp + (1-alpha)*k*res = p  
% physical parameters

M = [1, 1, 1]; K= 100*[1, 1, 1]; C = [2, 2, 2];
dt = 0.02;
% bouc-wen model paramseters
alpha = 0.1; A = 1; beta = 0.5; gamma = 0.05; n = 1; 
% exteral force p
load('El_centro.mat')  % as p
ef = -1*ef;
t_steps = linspace(0, length(ef)*0.01, length(ef));  % discrete time steps
% SDOF(t,y,t_steps,p, m, c, k, A, alpha, beta, gamma, n)
tspan = [0:0.001:40];
ic = zeros(1,9);
opts = odeset('RelTol',1e-6,'AbsTol',1e-8);
tic
[t,y] = ode45(@(t,y) MDOF(t,y, t_steps, ef, M, C, K, A, alpha, beta, gamma, n), tspan, ic, opts);
solving_time = toc
save('./result/MDOF.mat', 'solving_time', 't', 'y')
figure()
plot(t,y(:,1:3))


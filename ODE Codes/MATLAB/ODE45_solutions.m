%% ODE45 solutions for VDP equations
%% case 1-1
clear; clc
alpha = 0.1; 
opts = odeset('RelTol',1e-4,'AbsTol',1e-8);
tic
[t,y] = ode45(@(t,y) vdp_alpha(t,y,alpha),[0:1e-3:50],[1, 1], opts);
solving_time = toc
save('./result/case1-1.mat','solving_time', 't', 'y')

%% case 1-2
clear; clc
alpha = 0.1; 

opts = odeset('RelTol',1e-4,'AbsTol',1e-8);
tic
[t,y] = ode45(@(t,y) vdp_alpha(t,y,alpha),[0:1e-3:50],[6, 0], opts);
solving_time = toc
save('./result/case1-2.mat','solving_time', 't', 'y')
%% case 2-1
clear; clc
alpha = 1; 

opts = odeset('RelTol',1e-4,'AbsTol',1e-8);
tic
[t,y] = ode45(@(t,y) vdp_alpha(t,y,alpha),[0:1e-3:50],[1, 1], opts);
solving_time = toc
save('./result/case2-1.mat','solving_time', 't', 'y')
plot(y(:,1), y(:,2))

%% case 2-2
clear; clc
alpha = 1; 

opts = odeset('RelTol',1e-4,'AbsTol',1e-8);
tic
[t,y] = ode45(@(t,y) vdp_alpha(t,y,alpha),[0:1e-3:50],[6, 0], opts);
solving_time = toc
save('./result/case2-2.mat','solving_time', 't', 'y')
plot(y(:,1), y(:,2))

%% case 2-2
clear; clc
alpha = 1; 

opts = odeset('RelTol',1e-4,'AbsTol',1e-8);
tic
[t,y] = ode45(@(t,y) vdp_alpha(t,y,alpha),[0:1e-3:50],[0, 6], opts);
solving_time = toc
save('./result/case2-3.mat','solving_time', 't', 'y')
plot(y(:,1), y(:,2))
%% case 3-1
clear; clc
omega = 7; alpha = 5; beta = 5;

pt = linspace(0,1e2,1e4); p = cos(omega*pt);
opts = odeset('RelTol',1e-4,'AbsTol',1e-8);
tic
[t,y] = ode45(@(t,y) vdp_alpha_p(t,y,alpha, beta, pt, p),[0:1e-4:50],[0, 0], opts);
solving_time = toc
save('./result/case3-1.mat','solving_time', 't', 'y')
plot(y(:,1), y(:,2))

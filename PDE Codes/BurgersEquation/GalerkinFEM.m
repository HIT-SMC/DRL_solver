clc;clear
m = 0; 
global miu
miu = 0.002;
x = linspace(-1, 1, 150);
t = linspace(0, 1, 101);

sol = pdepe(m, @pdefun, @pdeic, @pdebc, x, t); % fun, initial, boundarys
u = sol(:, :, 1);

surf(x,t,u) 
title('Numerical solution computed with 20 mesh points.')
xlabel('Distance x')
ylabel('Time t')

% A solution profile can also be illuminating.
figure
plot(x,u(end,:))
title('Solution at t = 1')
xlabel('Distance x')
ylabel('u(x,1)')
uuuu = u(end,:)';
function [c, f, s] = pdefun(x, t, u, DuDx)
global miu
c = 1;
f = miu*DuDx;
s = -u*DuDx;
end

function u0 = pdeic(x)
u0 = -sin(pi*x);
end


function [pl, ql, pr, qr] = pdebc(xl, ul, xr, ur, t)
pl = ul;
ql = 0;
pr = ur;
qr = 0;
end
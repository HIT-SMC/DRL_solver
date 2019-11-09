%%
clc;clear
u_abs_col = [];
u_real_col = [];
u_imag_col = [];
for t = 0:0.01:1
    disp(t)
    close all
    dom = [-10 10];
    tspan = [0 t];
    S = spinop(dom,tspan);
    S.lin = @(u) 0.5i*diff(u, 2);
    S.nonlin= @(u) 1i*abs(u).^2.*u;
    A = 1; B = 1;
    x = chebfun(@(x) x,dom);
    u0 = @(x) 4./(exp(x) + exp(-x));
    S.init = chebfun(u0, dom);
    [u, bb] = spin(S,1001,0.0001, 'plot', 'off');
    u_abs_ = abs(u.funs{1, 1}.onefun.values);
    u_real_ = real(u.funs{1, 1}.onefun.values);
    u_imag_ = imag(u.funs{1, 1}.onefun.values);
    figure
    % plot(u_abs_.funs{1, 1}.onefun.values)
    % hold on
    plot(u_abs_)
    hold on
    plot(u_real_)
    hold on
    plot(u_imag_)
    u_abs_col = [u_abs_col u_abs_];
    u_real_col = [u_real_col u_real_];
    u_imag_col = [u_imag_col u_imag_];
    clearvars -except t u_abs_col u_real_col u_imag_col
end
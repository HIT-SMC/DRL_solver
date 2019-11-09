function dydt = lorenzeq(t, y)
% lorenz equation
sigma = 10.;  rho = 10.;  beta = 8./3.;
A = [-sigma, sigma, 0;
    rho, -1, -y(1);
    0, y(1), -beta];
dydt = A*y;


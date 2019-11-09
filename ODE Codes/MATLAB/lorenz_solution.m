% lorenz solution
tspan = [0:0.001:100];
[t1,y1] = ode45(@(t,y) lorenzeq(t,y), tspan, [0,2,0]);
[t2,y2] = ode45(@(t,y) lorenzeq(t,y), tspan, [0,-2,0]);
[t3,y3] = ode45(@(t,y) lorenzeq(t,y), tspan, [0,2.01,0]);

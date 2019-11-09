function dydt = vdp_alpha_p(t,y, alpha, beta, pt, p_w)
%VDP1  Evaluate the van der Pol ODEs for mu = 1
%
%   See also ODE113, ODE23, ODE45.

%   Jacek Kierzenka and Lawrence F. Shampine
%   Copyright 1984-2014 The MathWorks, Inc.
p = interp1(pt, p_w, t);
dydt = [y(2); beta*p + alpha*(1-y(1)^2)*y(2)-y(1)];

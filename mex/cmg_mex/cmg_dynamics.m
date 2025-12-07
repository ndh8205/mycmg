function x_dot = cmg_dynamics(x, u, params)
% cmg_dynamics: CMG system dynamics
%
% State x (8x1): [roll; pitch; yaw; p; q; r; gamma1; gamma2]
% Control u (2x1): [gamma1_dot; gamma2_dot]

omega = x(4:6);
g1 = x(7);
g2 = x(8);

J = params.J;
J_inv = params.J_inv;
h1 = params.h1;
h2 = params.h2;

sin_g1 = sin(g1); cos_g1 = cos(g1);
sin_g2 = sin(g2); cos_g2 = cos(g2);

% CMG angular momentum
h_cmg = [0; h1*cos_g1 + h2*cos_g2; h1*sin_g1 + h2*sin_g2];

% CMG torque
tau_cmg = [0;
           -h1*sin_g1*u(1) - h2*sin_g2*u(2);
            h1*cos_g1*u(1) + h2*cos_g2*u(2)];

% Gyroscopic + Coupling
M = tau_cmg - cross(omega, J*omega) - cross(omega, h_cmg);

x_dot = [omega; J_inv*M; u];

end
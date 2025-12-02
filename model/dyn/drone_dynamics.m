function [x_dot, varargout] = drone_dynamics(x, u, w, dt, params, k)
% drone_dynamics: 6-DoF drone dynamics model
%
% State vector x (26x1):
%   x(1:3)   - position [m] (NED frame)
%   x(4:6)   - velocity [m/s] (body frame)
%   x(7:10)  - quaternion [qw,qx,qy,qz] (body to NED)
%   x(11:13) - angular velocity [rad/s] (body frame)
%   x(14:17) - motor angular velocity [rad/s]
%   x(18:20) - gyro bias [rad/s]
%   x(21:23) - accelerometer bias [m/s^2]
%   x(24:26) - magnetometer bias [Gauss]
%
% Input:
%   u  - 4x1 motor speed command [rad/s]
%   w  - 9x1 process noise (bias random walk)
%   dt - time step [s]
%   params - parameter struct
%   k  - current step index
%
% Output:
%   x_dot - 26x1 state derivative

% Extract states
pos   = x(1:3);
vel_b = x(4:6);
quat  = x(7:10);
omega = x(11:13);
omega_m = x(14:17);

% Parameters
m   = params.drone.body.m;
J   = params.drone.body.J;
g   = params.env.g;

% Normalize quaternion
quat = quat / norm(quat);

% DCM (body to NED)
R_b2n = GetDCM_QUAT(quat);
R_n2b = R_b2n';

%% Force and Moment Calculation (using control_allocator)
omega_sq = omega_m.^2;
[cmd_vec, ~] = control_allocator(omega_sq, params, 'forward');
T_total = cmd_vec(1);
M_body  = cmd_vec(2:4);
F_body  = [0; 0; -T_total];

%% Equations of Motion

% Position derivative (NED = R_b2n * vel_body)
pos_dot = R_b2n * vel_b;

% Velocity derivative (body frame)
gravity_body = R_n2b * [0; 0; g];
vel_dot = F_body / m + gravity_body - skew3(omega) * vel_b;

% Quaternion derivative
quat_dot = Derivative_Quat(quat, omega);

% Angular velocity derivative
omega_dot = J \ (M_body - skew3(omega) * (J * omega));

% Motor dynamics
omega_m_dot = motor_dynamics(omega_m, u, params);

% Bias dynamics
b = x(18:26);
b_dot = bias_dynamics(b, w, params);

%% Ground collision (NED: z >= 0 means at or below ground)
if pos(3) >= 0
    pos_dot = zeros(3,1);
    vel_dot = zeros(3,1);
    omega_dot = zeros(3,1);
end

%% Output
x_dot = [pos_dot; vel_dot; quat_dot; omega_dot; omega_m_dot; b_dot];

% Optional outputs for srk4 compatibility
if nargout > 1
    varargout{1} = T_total;  % Thrust
    varargout{2} = [];       % alpha_tot
    varargout{3} = [];       % phi_A
    varargout{4} = F_body;   % F_aero
    varargout{5} = M_body;   % M_aero
    varargout{6} = [];       % moment_coupling
end

end
function [x_dot, varargout] = drone_dynamics(x, u, w, dt, params, k, t, dist_state)
% drone_dynamics: 6-DoF drone dynamics model (Quad/Hexa) with disturbances
%
% State vector x:
%   Quadrotor (26x1):
%     x(1:3)   - position [m] (NED frame)
%     x(4:6)   - velocity [m/s] (body frame)
%     x(7:10)  - quaternion [qw,qx,qy,qz] (body to NED)
%     x(11:13) - angular velocity [rad/s] (body frame)
%     x(14:17) - motor angular velocity [rad/s]
%     x(18:20) - gyro bias [rad/s]
%     x(21:23) - accelerometer bias [m/s^2]
%     x(24:26) - magnetometer bias [Gauss]
%
%   Hexarotor (28x1):
%     x(1:3)   - position [m] (NED frame)
%     x(4:6)   - velocity [m/s] (body frame)
%     x(7:10)  - quaternion [qw,qx,qy,qz] (body to NED)
%     x(11:13) - angular velocity [rad/s] (body frame)
%     x(14:19) - motor angular velocity [rad/s]
%     x(20:22) - gyro bias [rad/s]
%     x(23:25) - accelerometer bias [m/s^2]
%     x(26:28) - magnetometer bias [Gauss]
%
% Input:
%   u  - motor speed command [rad/s] (4x1 or 6x1)
%   w  - 9x1 process noise (bias random walk)
%   dt - time step [s]
%   params - parameter struct
%   k  - current step index
%   t  - current time [s] (optional, for disturbance)
%   dist_state - disturbance state struct (optional)
%
% Output:
%   x_dot - state derivative
%   varargout{1} - T_total
%   varargout{2} - [] (reserved)
%   varargout{3} - [] (reserved)
%   varargout{4} - F_body
%   varargout{5} - M_body
%   varargout{6} - dist_state (updated)

%% Handle optional inputs
if nargin < 7
    t = k * dt;  % Estimate time from step index
end
if nargin < 8
    dist_state = [];
end

%% Determine drone type
if isfield(params.drone, 'type') && strcmp(params.drone.type, 'hexa')
    n_motor = 6;
    motor_idx = 14:19;
    bias_idx = 20:28;
else
    n_motor = 4;
    motor_idx = 14:17;
    bias_idx = 18:26;
end

% Extract states
pos   = x(1:3);
vel_b = x(4:6);
quat  = x(7:10);
omega = x(11:13);
omega_m = x(motor_idx);

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
M_ctrl  = cmd_vec(2:4);
F_thrust = [0; 0; -T_total];

%% Disturbance Calculation
F_dist = zeros(3,1);
M_dist = zeros(3,1);

if isfield(params, 'dist') && params.dist.enable && ~isempty(dist_state)
    % Torque disturbance
    [M_dist, dist_state] = dist_torque(t, params, dist_state);
    
    % Wind disturbance (force in body frame)
    [F_dist, dist_state] = dist_wind(vel_b, R_b2n, t, dt, params, dist_state);
end

% Total force and moment
F_body = F_thrust + F_dist;
M_body = M_ctrl + M_dist;

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
b = x(bias_idx);
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
    varargout{1} = T_total;     % Thrust
    varargout{2} = [];          % reserved
    varargout{3} = [];          % reserved
    varargout{4} = F_body;      % Total force (body)
    varargout{5} = M_body;      % Total moment (body)
    varargout{6} = dist_state;  % Updated disturbance state
end

end
function [x_dot, varargout] = drone_dynamics(x, u, w, dt, params, k)
% drone_dynamics: 6-DoF drone dynamics model (Quad/Hexa)

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

%% Force and Moment Calculation
omega_sq = omega_m.^2;
[cmd_vec, ~] = control_allocator(omega_sq, params, 'forward');
T_total = cmd_vec(1);
M_body  = cmd_vec(2:4);
F_body  = [0; 0; -T_total];

%% Equations of Motion
pos_dot = R_b2n * vel_b;
gravity_body = R_n2b * [0; 0; g];
vel_dot = F_body / m + gravity_body - skew3(omega) * vel_b;
quat_dot = Derivative_Quat(quat, omega);
omega_dot = J \ (M_body - skew3(omega) * (J * omega));
omega_m_dot = motor_dynamics(omega_m, u, params);

% Bias dynamics
b = x(bias_idx);
b_dot = bias_dynamics(b, w, params);

%% Ground collision
if pos(3) >= 0
    pos_dot = zeros(3,1);
    vel_dot = zeros(3,1);
    omega_dot = zeros(3,1);
end

%% Output
x_dot = [pos_dot; vel_dot; quat_dot; omega_dot; omega_m_dot; b_dot];

if nargout > 1
    varargout{1} = T_total;
    varargout{2} = [];
    varargout{3} = [];
    varargout{4} = F_body;
    varargout{5} = M_body;
    varargout{6} = [];
end

end
function [q_des, thrust_cmd, ctrl_state] = position_controller(pos_des, pos, vel, yaw_des, ctrl_state, gains, params)
% position_controller: Position PID controller with attitude/thrust output
%
% Inputs:
%   pos_des    - 3x1 desired position [m] (NED frame)
%   pos        - 3x1 current position [m] (NED frame)
%   vel        - 3x1 current velocity [m/s] (NED frame)
%   yaw_des    - desired yaw angle [rad]
%   ctrl_state - struct with integrator states
%   gains      - struct with PID gains (Kp, Ki, Kd: 3x1)
%   params     - system parameters
%
% Outputs:
%   q_des      - 4x1 desired quaternion
%   thrust_cmd - total thrust command [N]
%   ctrl_state - updated integrator states

%% Parameter extraction
pos_des = pos_des(:);
pos = pos(:);
vel = vel(:);

Kp = gains.Kp(:);
Ki = gains.Ki(:);
Kd = gains.Kd(:);
int_limit = gains.int_limit(:);

m = params.drone.body.m;
g = params.env.g;

%% Position error (NED)
e_pos = pos_des - pos;

%% Velocity error (assume desired velocity = 0 for waypoint hold)
vel_des = zeros(3,1);
e_vel = vel_des - vel;

%% Integrator with anti-windup
ctrl_state.int_pos = ctrl_state.int_pos + e_pos * ctrl_state.dt;
ctrl_state.int_pos = max(min(ctrl_state.int_pos, int_limit), -int_limit);

%% PID: desired acceleration (NED)
a_des = Kp .* e_pos + Ki .* ctrl_state.int_pos + Kd .* e_vel;

%% Acceleration to thrust and attitude
%  F_des = m * (a_des - g_ned)  where g_ned = [0; 0; g]
%  Thrust acts along -z_body, so F_body = [0; 0; -T]
%  F_ned = R_b2n * F_body = -T * z_b_ned
%  Therefore: z_b_ned = -F_des / |F_des|

g_ned = [0; 0; g];
F_des = m * (a_des - g_ned);  % Desired force in NED

% Total thrust magnitude
thrust_cmd = norm(F_des);

% Desired body z-axis in NED frame
if thrust_cmd > 1e-6
    z_b_des = -F_des / thrust_cmd;
else
    z_b_des = [0; 0; -1];  % Default: level
end

%% Construct desired rotation matrix
%  Given z_b_des and yaw_des, construct R_b2n_des

% Desired x-axis direction (from yaw)
x_c = [cos(yaw_des); sin(yaw_des); 0];

% Desired y-axis: z × x (then normalize)
y_b_des = cross(z_b_des, x_c);
y_b_des = y_b_des / norm(y_b_des);

% Recompute x-axis to ensure orthogonality
x_b_des = cross(y_b_des, z_b_des);
x_b_des = x_b_des / norm(x_b_des);

% Rotation matrix (body to NED)
R_b2n_des = [x_b_des, y_b_des, z_b_des];

%% Convert to quaternion
q_des = DCM2Quat(R_b2n_des);
q_des = q_des(:);

% Ensure positive scalar part
if q_des(1) < 0
    q_des = -q_des;
end

end
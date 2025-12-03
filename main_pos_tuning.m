%% main_tiltrotor_waypoint.m - Tiltrotor Waypoint Control Test
clear; clc; close all;
addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));

%% Parameters
params = params_tiltrotor_init();

%% Simulation settings
sim_hz = 1000;
dt = 1/sim_hz;
t_end = 30;
t = 0:dt:t_end;
N = length(t);

%% Process noise covariance
Q = zeros(9,9);

%% Waypoints [x, y, z] in NED
waypoints = [
    0,   0,  -10;   % WP1: Hover at start
    10,  0,  -10;   % WP2: Forward
    10,  10, -10;   % WP3: Right
    0,   10, -10;   % WP4: Back
    0,   0,  -10;   % WP5: Return
];

n_wp = size(waypoints, 1);
wp_idx = 1;
wp_threshold = 1.0;  % [m] arrival threshold
yaw_des = 0;

%% Control gains
% Position PID
gains_pos.Kp = [2; 2; 3];
gains_pos.Ki = [0.1; 0.1; 0.2];
gains_pos.Kd = [2; 2; 2];
gains_pos.int_limit = [2; 2; 2];

% Attitude PID
gains_att.Kp = [8; 8; 4];
gains_att.Ki = [0; 0; 0];
gains_att.Kd = [2; 2; 1];
gains_att.int_limit = [1; 1; 1];

%% Control state initialization
ctrl_state.int_pos = zeros(3,1);
ctrl_state.int_att = zeros(3,1);
ctrl_state.dt = dt;

%% Initial state (30x1)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;

omega_hover = sqrt(m * g / (4 * k_T));

x0 = zeros(30, 1);
x0(1:3)   = [0; 0; -0.1];             % position (NED)
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = [1; 0; 0; 0];             % quaternion (level)
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:17) = omega_hover * ones(4,1);  % motor speed
x0(18:26) = zeros(9, 1);              % biases
x0(27:28) = [0; 0];                   % tilt angles (hover = 0)
x0(29:30) = [0; 0];                   % tilt rates

%% Data storage
X = zeros(30, N);
U = zeros(9, N);
POS_DES = zeros(3, N);
X(:,1) = x0;
POS_DES(:,1) = waypoints(1,:)';

%% Simulation loop
for k = 1:N-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    omega = x_k(11:13);
    tilt = x_k(27:28);
    
    % Velocity in NED
    R_b2n = GetDCM_QUAT(q);
    vel_ned = R_b2n * vel_b;
    
    % Current waypoint
    pos_des = waypoints(wp_idx, :)';
    
    % Waypoint arrival check
    dist_to_wp = norm(pos_ned - pos_des);
    if dist_to_wp < wp_threshold && wp_idx < n_wp
        wp_idx = wp_idx + 1;
        ctrl_state.int_pos = zeros(3,1);  % Reset integrator
    end
    pos_des = waypoints(wp_idx, :)';
    POS_DES(:,k) = pos_des;
    
    % Position control
    [q_des, thrust_cmd, ctrl_state] = position_controller(...
        pos_des, pos_ned, vel_ned, yaw_des, ctrl_state, gains_pos, params);
    
    % Attitude control
    [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains_att, dt);
    
    % Control allocation (hover mode: tilt = 0)
    cmd_vec = [thrust_cmd; tau_cmd];
    omega_sq = tiltrotor_allocator(cmd_vec, tilt, params, 'inverse');
    omega_sq = max(omega_sq, 0);
    u_motor = sqrt(omega_sq);
    
    % Saturate motor commands
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u_motor = max(min(u_motor, omega_max), omega_min);
    
    % Tilt command (hover = 0)
    u_tilt = [0; 0];
    
    % Full input vector
    u = zeros(9, 1);
    u(1:4) = u_motor;
    u(5:6) = u_tilt;
    u(7:9) = 0;
    
    U(:,k) = u;
    
    % Integration
    [x_next, ~, ~, ~, ~, ~, ~] = srk4(@tiltrotor_dynamics, x_k, u, Q, dt, params, k, dt);
    
    % Normalize quaternion and ensure continuity
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);
POS_DES(:,end) = POS_DES(:,end-1);

%% Plot results
figure('Position', [100 100 1400 900]);

% X position
subplot(3,3,1);
plot(t, X(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, POS_DES(1,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('X [m]');
title('X Position'); grid on;
legend('Actual', 'Desired');

% Y position
subplot(3,3,2);
plot(t, X(2,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, POS_DES(2,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Y [m]');
title('Y Position'); grid on;
legend('Actual', 'Desired');

% Z position (altitude)
subplot(3,3,3);
plot(t, -X(3,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, -POS_DES(3,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude'); grid on;
legend('Actual', 'Desired');

% Euler angles
subplot(3,3,4);
euler = zeros(3,N);
for k = 1:N
    euler(:,k) = Quat2Euler(X(7:10,k));
end
plot(t, rad2deg(euler));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi'); title('Euler Angles'); grid on;

% Motor commands
subplot(3,3,5);
plot(t, U(1:4,:));
xlabel('Time [s]'); ylabel('\omega [rad/s]');
legend('M0','M1','M2','M3'); title('Motor Commands'); grid on;

% Motor speed actual
subplot(3,3,6);
plot(t, X(14:17,:));
xlabel('Time [s]'); ylabel('\omega [rad/s]');
legend('M0','M1','M2','M3'); title('Motor Speed (Actual)'); grid on;

% Velocity NED
subplot(3,3,7);
vel_ned_hist = zeros(3,N);
for i = 1:N
    R = GetDCM_QUAT(X(7:10,i));
    vel_ned_hist(:,i) = R * X(4:6,i);
end
plot(t, vel_ned_hist);
xlabel('Time [s]'); ylabel('Velocity [m/s]');
legend('Vn','Ve','Vd'); title('Velocity (NED)'); grid on;

% Position error
subplot(3,3,8);
pos_err = sqrt((X(1,:)-POS_DES(1,:)).^2 + (X(2,:)-POS_DES(2,:)).^2 + (X(3,:)-POS_DES(3,:)).^2);
plot(t, pos_err);
xlabel('Time [s]'); ylabel('Error [m]');
title('Position Error'); grid on;

% 3D trajectory
subplot(3,3,9);
plot3(X(1,:), X(2,:), -X(3,:), 'b-', 'LineWidth', 1.5); hold on;
plot3(waypoints(:,1), waypoints(:,2), -waypoints(:,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot3(X(1,1), X(2,1), -X(3,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Alt [m]');
title('3D Trajectory'); grid on; axis equal;
legend('Path', 'Waypoints', 'Start');
view(45, 30);

sgtitle('Tiltrotor Waypoint Control (Hover Mode)');
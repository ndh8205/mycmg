%% main.m - Position Control Test (Hexarotor)
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Parameters
params = params_init('hexa');

%% Disturbance settings
% dist_preset = 'nominal';  % 'nominal', 'level1', 'level2', 'level3'
% dist_preset = 'level1';  % 'nominal', 'level1', 'level2', 'level3'
dist_preset = 'level2';  % 'nominal', 'level1', 'level2', 'level3'
% dist_preset = 'level3';  % 'nominal', 'level1', 'level2', 'level3'
% dist_preset = 'paper';  % 'nominal', 'level1', 'level2', 'level3'
% dist_preset = 'level_hell';  % 'nominal', 'level1', 'level2', 'level3'

[params, dist_state] = dist_init(params, dist_preset);
if params.dist.uncertainty.enable
    params_true = apply_uncertainty(params);
else
    params_true = params;
end

%% Simulation settings
sim_hz = 1000;
dt = 1/sim_hz;
t_end = 30;
t = 0:dt:t_end;
N = length(t);

omega_bar2RPM = 60 / (2 * pi);

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

% waypoints = [
%     0,   0,  -10;   % WP1: Hover at start
%     0,  0,  -10;   % WP2: Forward
%     0,  0, -10;   % WP3: Right
%     0,   0, -10;   % WP4: Back
%     0,   0,  -10;   % WP5: Return
% ];

n_wp = size(waypoints, 1);
wp_idx = 1;
wp_threshold = 0.5;  % [m] arrival threshold
yaw_des = 0;

%% Control gains

% Position PID
gains_pos.Kp = [1; 1; 2];
gains_pos.Ki = [0.2; 0.2; 0.2];
gains_pos.Kd = [2; 2; 3];
gains_pos.int_limit = [2; 2; 2];

% Attitude PID
gains_att.Kp = [8; 8; 6];
gains_att.Ki = [0.1; 0.1; 0.1];
gains_att.Kd = [2; 2; 1.5];
gains_att.int_limit = [1; 1; 1];

%% Control state initialization
ctrl_state.int_pos = zeros(3,1);
ctrl_state.int_att = zeros(3,1);
ctrl_state.dt = dt;

%% Initial state (28x1 for Hexa)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -0.01];            % position (NED)
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = [1; 0; 0; 0];             % quaternion (level)
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:19) = omega_hover * ones(6,1);  % motor speed (6 motors)
x0(20:28) = zeros(9, 1);              % biases

%% Data storage
X = zeros(28, N);
U = zeros(6, N);
POS_DES = zeros(3, N);
TAU_DIST = zeros(3, N);
F_DIST = zeros(3, N);
X(:,1) = x0;
POS_DES(:,1) = waypoints(1,:)';

%% Simulation loop
fprintf('Running simulation (dist: %s)...\n', dist_preset);
for k = 1:N-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    omega = x_k(11:13);
    
    % Velocity in NED
    R_b2n = GetDCM_QUAT(q);
    vel_ned = R_b2n * vel_b;
    
    % Current waypoint
    pos_des = waypoints(wp_idx, :)';
    
    % Waypoint arrival check
    dist_to_wp = norm(pos_ned - pos_des);
    if dist_to_wp < wp_threshold && wp_idx < n_wp
        wp_idx = wp_idx + 1;
        ctrl_state.int_pos = zeros(3,1);
    end
    pos_des = waypoints(wp_idx, :)';
    POS_DES(:,k) = pos_des;
    
    % Position control
    [q_des, thrust_cmd, ctrl_state] = position_pid(...
        pos_des, pos_ned, vel_ned, yaw_des, ctrl_state, gains_pos, params);
    
    % Attitude control
    [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains_att, dt);
    
    % Control allocation
    cmd_vec = [thrust_cmd; tau_cmd];
    omega_sq = control_allocator(cmd_vec, params, 'inverse');
    omega_sq = max(omega_sq, 0);
    u = sqrt(omega_sq);
    
    % Saturate motor commands
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u = max(min(u, omega_max), omega_min);
    U(:,k) = u;
    
    % Log disturbance
    if params.dist.enable
        [tau_d, ~] = dist_torque(t(k), params, dist_state);
        [F_d, ~] = dist_wind(vel_b, R_b2n, t(k), dt, params, dist_state);
        TAU_DIST(:,k) = tau_d;
        F_DIST(:,k) = F_d;
    end
    
    % Integration (use params_true for dynamics)
    [x_next, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x_k, u, Q, dt, params_true, k, dt, t(k), dist_state);
    
    % Normalize quaternion and ensure continuity
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);
POS_DES(:,end) = POS_DES(:,end-1);
fprintf('Simulation complete.\n');

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
legend('\phi','\theta','\psi');
title('Euler Angles'); grid on;

% Motor speed (actual)
subplot(3,3,5);
plot(t, X(14:19,:) * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Speed (Actual)'); grid on;

% 3D trajectory
subplot(3,3,6);
plot3(X(1,:), X(2,:), -X(3,:), 'b-', 'LineWidth', 1.5); hold on;
plot3(waypoints(:,1), waypoints(:,2), -waypoints(:,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot3(X(1,1), X(2,1), -X(3,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Alt [m]');
title('3D Trajectory'); grid on; axis equal;
legend('Path', 'Waypoints', 'Start');
view(45, 30);

% Torque disturbance
subplot(3,3,7);
plot(t, TAU_DIST');
xlabel('Time [s]'); ylabel('Torque [Nm]');
legend('\tau_x','\tau_y','\tau_z');
title('Torque Disturbance'); grid on;

% Wind force
subplot(3,3,8);
plot(t, F_DIST');
xlabel('Time [s]'); ylabel('Force [N]');
legend('F_x','F_y','F_z');
title('Wind Force (Body)'); grid on;

% Position error
subplot(3,3,9);
pos_err = X(1:3,:) - POS_DES;
plot(t, vecnorm(pos_err), 'k', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('|Error| [m]');
title('Position Error'); grid on;

sgtitle(sprintf('Hexarotor Position Control (Dist: %s)', dist_preset));

%% Print summary
fprintf('\n=== Simulation Summary ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Final position error: %.3f m\n', norm(X(1:3,end) - POS_DES(:,end)));
fprintf('Max position error: %.3f m\n', max(vecnorm(pos_err)));
fprintf('Max attitude: %.2f deg\n', max(abs(rad2deg(euler(:)))));


%% Export to CSV
data = [t', X(1:3,:)', POS_DES', rad2deg(euler)', X(11:13,:)', U', X(14:19,:)', ...
        TAU_DIST', F_DIST'];
header = {'t','pos_x','pos_y','pos_z','pos_des_x','pos_des_y','pos_des_z',...
          'euler_roll','euler_pitch','euler_yaw','omega_p','omega_q','omega_r',...
          'motor_cmd_1','motor_cmd_2','motor_cmd_3','motor_cmd_4','motor_cmd_5','motor_cmd_6',...
          'motor_actual_1','motor_actual_2','motor_actual_3','motor_actual_4','motor_actual_5','motor_actual_6',...
          'tau_dist_x','tau_dist_y','tau_dist_z','F_dist_x','F_dist_y','F_dist_z'};
filename = sprintf('pos_pid_%s.csv', dist_preset);
fid = fopen(filename, 'w');
fprintf(fid, '%s,', header{1:end-1}); fprintf(fid, '%s\n', header{end});
fclose(fid);
dlmwrite(filename, data, '-append', 'precision', '%.6f');
fprintf('Data saved to: %s\n', filename);
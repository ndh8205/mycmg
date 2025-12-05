%% main_smc_pos_test.m - Hexarotor SMC Position Control Test
% Fractional-order SMC for position, attitude, altitude
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Parameters
params = params_init('hexa');

%% Disturbance settings
% dist_preset = 'nominal';
% dist_preset = 'level1';
% dist_preset = 'level2';
% dist_preset = 'level3';
dist_preset = 'level_hell';

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
n_wp = size(waypoints, 1);
wp_idx = 1;
wp_threshold = 0.5;  % [m] arrival threshold
yaw_des = 0;

% %% SMC Gains
gains_pos_smc.a = [10; 10; 10];          % 슬라이딩 면 게인
gains_pos_smc.b = 0;         % (사용 안함)
gains_pos_smc.lambda1 = [0.8; 0.8; 0.8];  % 선형 도달 게인
gains_pos_smc.lambda2 = [0.3; 0.3; 0.55];  % 분수차 게인 (작게!)
gains_pos_smc.r = 0.98;

% Attitude SMC gains
gains_att_smc.a = 15;
gains_att_smc.b = 15;
gains_att_smc.lambda1 = 5.4;
gains_att_smc.lambda2 = 5.4;
gains_att_smc.r = 0.95;

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
TAU_CMD = zeros(3, N);
TAU_DIST = zeros(3, N);
F_DIST = zeros(3, N);
X(:,1) = x0;
POS_DES(:,1) = waypoints(1,:)';

%% Simulation loop
fprintf('Running SMC position control (dist: %s)...\n', dist_preset);
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
        fprintf('  t=%.1fs: Reached WP%d, heading to WP%d\n', t(k), wp_idx-1, wp_idx);
    end
    pos_des = waypoints(wp_idx, :)';
    POS_DES(:,k) = pos_des;
    
    % SMC Position control
    [q_des, thrust_cmd, ctrl_state] = position_smc(...
        pos_des, pos_ned, vel_ned, yaw_des, ctrl_state, gains_pos_smc, params);
    
    % SMC Attitude control
    [tau_cmd, ctrl_state] = attitude_smc(q_des, q, omega, ctrl_state, gains_att_smc, params);
    TAU_CMD(:,k) = tau_cmd;
    
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
TAU_CMD(:,end) = TAU_CMD(:,end-1);
fprintf('Simulation complete.\n');

%% Extract Euler angles
euler = zeros(3,N);
for k = 1:N
    euler(:,k) = Quat2Euler(X(7:10,k));
end

%% Plot results
figure('Position', [100 100 1400 900], 'Name', 'SMC Position Control');

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

% Control torque
subplot(3,3,8);
plot(t, TAU_CMD');
xlabel('Time [s]'); ylabel('Torque [Nm]');
legend('\tau_x','\tau_y','\tau_z');
title('SMC Torque Command'); grid on;

% Position error
subplot(3,3,9);
pos_err = X(1:3,:) - POS_DES;
plot(t, vecnorm(pos_err), 'k', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('|Error| [m]');
title('Position Error'); grid on;

sgtitle(sprintf('Hexarotor SMC Position Control (Dist: %s)', dist_preset));

%% Print summary
fprintf('\n=== SMC Position Control Summary ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Final position error: %.3f m\n', norm(X(1:3,end) - POS_DES(:,end)));
fprintf('Max position error: %.3f m\n', max(vecnorm(pos_err)));
fprintf('Max attitude: %.2f deg\n', max(abs(rad2deg(euler(:)))));
fprintf('Waypoints reached: %d/%d\n', wp_idx, n_wp);
fprintf('====================================\n');
%% main_pos_mppi_mex.m - Hexarotor MPPI Position Control with Waypoints
% CUDA MEX version with motor dynamics
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Check MEX file
if ~exist('hexa_mppi_mex', 'file')
    error('hexa_mppi_mex not found. Run: mexcuda hexa_mppi_mex.cu -lcurand');
end

%% Parameters (Hexarotor)
params = params_init('hexa');

%% Disturbance settings
dist_preset = 'nominal';
% dist_preset = 'level1';
% dist_preset = 'level2';

[params, dist_state] = dist_init(params, dist_preset);
if params.dist.uncertainty.enable
    params_true = apply_uncertainty(params);
else
    params_true = params;
end

%% Simulation settings
sim_hz = 1000;
dt_sim = 1/sim_hz;
t_end = 30;
t = 0:dt_sim:t_end;
N_sim = length(t);

ctrl_hz = 50;  % MPPI control rate
ctrl_decimation = sim_hz / ctrl_hz;
dt_ctrl = 1 / ctrl_hz;

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

mppi_params.K = 4096*10;
mppi_params.N = 100;
mppi_params.dt = dt_ctrl;
mppi_params.lambda = 800.7310;
mppi_params.nu = 10.0;
mppi_params.sigma = 15.1658;  % 144.8 RPM
mppi_params.w_pos_xy = 1000.0037;
mppi_params.w_pos_z = 1000.2281;
mppi_params.w_vel_xy = 1000.2683;
mppi_params.w_vel_z = 529.8645;
mppi_params.w_att = 142.9748;
mppi_params.w_yaw = 3776.0578;
mppi_params.w_omega_rp = 0.0075;
mppi_params.w_omega_yaw = 360.6177;
mppi_params.w_terminal = 1.2051;
mppi_params.R = 1.81e-08;

%% Drone parameters for MEX (single precision struct)
drone_params.m = single(params.drone.body.m);
drone_params.Jxx = single(params.drone.body.J(1,1));
drone_params.Jyy = single(params.drone.body.J(2,2));
drone_params.Jzz = single(params.drone.body.J(3,3));
drone_params.g = single(params.env.g);
drone_params.k_T = single(params.drone.motor.k_T);
drone_params.k_M = single(params.drone.motor.k_M);
drone_params.L = single(params.drone.body.L);
drone_params.omega_max = single(params.drone.motor.omega_b_max);
drone_params.omega_min = single(params.drone.motor.omega_b_min);
drone_params.tau_up = single(params.drone.motor.tau_up);
drone_params.tau_down = single(params.drone.motor.tau_down);

%% Initial state (28x1 for hexa)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));
fprintf('Hover motor speed: %.2f RPM (%.2f rad/s)\n', omega_hover * omega_bar2RPM, omega_hover);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -0.01];            % position (NED), start near ground
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = [1; 0; 0; 0];             % quaternion (level)
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:19) = omega_hover * ones(6,1);  % motor speed (6 motors)
x0(20:28) = zeros(9, 1);              % biases

%% Initialize MPPI state
mppi_state.u_seq = single(omega_hover * ones(6, mppi_params.N));
mppi_state.pos_des = single(waypoints(1,:)');
mppi_state.yaw_des = single(yaw_des);

%% Motor state estimator initialization
omega_motor_est = omega_hover * ones(6,1);
u_prev = omega_hover * ones(6,1);

%% Data storage
X = zeros(28, N_sim);
U = zeros(6, N_sim);
POS_DES = zeros(3, N_sim);
TAU_DIST = zeros(3, N_sim);
F_DIST = zeros(3, N_sim);

% MPPI diagnostics
N_ctrl = ceil(N_sim / ctrl_decimation);
MPPI_STATS.min_cost = zeros(1, N_ctrl);
MPPI_STATS.avg_cost = zeros(1, N_ctrl);
MPPI_STATS.cost_breakdown = zeros(6, N_ctrl);
MPPI_STATS.sat_ratio = zeros(1, N_ctrl);
MPPI_STATS.ess = zeros(1, N_ctrl);
MPPI_STATS.time = zeros(1, N_ctrl);
MPPI_STATS.exec_time = zeros(1, N_ctrl);

X(:,1) = x0;
POS_DES(:,1) = waypoints(1,:)';
u_current = omega_hover * ones(6,1);
ctrl_idx = 0;

%% Simulation loop
fprintf('Running MPPI position control (dist: %s)...\n', dist_preset);
fprintf('K=%d, N=%d, sigma=%.1f, lambda=%.1f\n', ...
    mppi_params.K, mppi_params.N, mppi_params.sigma, mppi_params.lambda);
fprintf('Waypoints: %d\n', n_wp);

tic_total = tic;
for k = 1:N_sim-1
    x_k = X(:,k);
    
    % Extract position for waypoint check
    pos_ned = x_k(1:3);
    
    % Waypoint arrival check
    pos_des_current = waypoints(wp_idx, :)';
    dist_to_wp = norm(pos_ned - pos_des_current);
    if dist_to_wp < wp_threshold && wp_idx < n_wp
        wp_idx = wp_idx + 1;
        fprintf('  t=%.1fs: Reached WP%d, heading to WP%d\n', t(k), wp_idx-1, wp_idx);
    end
    pos_des_current = waypoints(wp_idx, :)';
    POS_DES(:,k) = pos_des_current;
    
    % MPPI control update at control rate
    if mod(k-1, ctrl_decimation) == 0
        ctrl_idx = ctrl_idx + 1;
        
        % Update motor state estimate (1st order dynamics)
        for i = 1:6
            if u_prev(i) >= omega_motor_est(i)
                tau_i = params.drone.motor.tau_up;
            else
                tau_i = params.drone.motor.tau_down;
            end
            omega_motor_est(i) = omega_motor_est(i) + dt_ctrl * (u_prev(i) - omega_motor_est(i)) / tau_i;
        end
        
        % Update desired position
        mppi_state.pos_des = single(pos_des_current);
        
        % Prepare state for MEX (19x1, single precision)
        x_mppi = single([x_k(1:3); x_k(4:6); x_k(7:10); x_k(11:13); omega_motor_est]);
        
        % Shift control sequence (warm start)
        mppi_state.u_seq(:, 1:end-1) = mppi_state.u_seq(:, 2:end);
        
        % Call MPPI MEX
        tic_mppi = tic;
        [u_opt, u_seq_new, stats] = hexa_mppi_mex(...
            x_mppi, mppi_state.u_seq, mppi_state.pos_des, mppi_state.yaw_des, ...
            drone_params, mppi_params);
        exec_time = toc(tic_mppi);
        
        % Update state
        u_current = double(u_opt);
        mppi_state.u_seq = u_seq_new;
        u_prev = u_current;
        
        % Store diagnostics
        MPPI_STATS.min_cost(ctrl_idx) = stats.min_cost;
        MPPI_STATS.avg_cost(ctrl_idx) = stats.avg_cost;
        MPPI_STATS.cost_breakdown(:, ctrl_idx) = stats.cost_breakdown;
        MPPI_STATS.sat_ratio(ctrl_idx) = stats.saturation_ratio;
        MPPI_STATS.ess(ctrl_idx) = stats.effective_sample_size;
        MPPI_STATS.time(ctrl_idx) = t(k);
        MPPI_STATS.exec_time(ctrl_idx) = exec_time;
    end
    
    % Apply control
    u = u_current;
    
    % Saturate motor commands
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u = max(min(u, omega_max), omega_min);
    U(:,k) = u;
    
    % Log disturbance
    if params.dist.enable
        vel_b = x_k(4:6);
        q = x_k(7:10);
        R_b2n = GetDCM_QUAT(q);
        [tau_d, ~] = dist_torque(t(k), params, dist_state);
        [F_d, ~] = dist_wind(vel_b, R_b2n, t(k), dt_sim, params, dist_state);
        TAU_DIST(:,k) = tau_d;
        F_DIST(:,k) = F_d;
    end
    
    % Integration
    [x_next, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x_k, u, Q, dt_sim, params_true, k, dt_sim, t(k), dist_state);
    
    % Normalize quaternion and ensure continuity
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);
POS_DES(:,end) = POS_DES(:,end-1);
total_time = toc(tic_total);

fprintf('Simulation complete. Total: %.2fs, Real-time factor: %.1fx\n', ...
    total_time, t_end/total_time);
fprintf('MPPI avg exec time: %.3f ms (%.1f Hz)\n', ...
    mean(MPPI_STATS.exec_time(1:ctrl_idx))*1000, 1/mean(MPPI_STATS.exec_time(1:ctrl_idx)));

%% Extract Euler angles
euler = zeros(3, N_sim);
for k = 1:N_sim
    euler(:,k) = Quat2Euler(X(7:10,k));
end

%% ==================== PLOTTING ====================

%% Figure 1: Position Control Results
figure('Position', [50 50 1400 900], 'Name', 'MPPI Position Control Results');

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

sgtitle(sprintf('Hexarotor MPPI Position Control (Dist: %s, K=%d)', dist_preset, mppi_params.K));

%% Figure 2: MPPI Diagnostics
figure('Position', [100 100 1400 700], 'Name', 'MPPI Position Diagnostics');

t_ctrl = MPPI_STATS.time(1:ctrl_idx);

% Cost breakdown (stacked area)
subplot(2,3,1);
cost_labels = {'Position', 'Velocity', 'Attitude', 'Yaw', 'Omega', 'Control'};
area(t_ctrl, MPPI_STATS.cost_breakdown(:, 1:ctrl_idx)');
xlabel('Time [s]'); ylabel('Cost');
title('Cost Breakdown (Stacked)');
legend(cost_labels, 'Location', 'best');
grid on;

% Cost over time
subplot(2,3,2);
semilogy(t_ctrl, MPPI_STATS.min_cost(1:ctrl_idx), 'b-', 'LineWidth', 1.5);
hold on;
semilogy(t_ctrl, MPPI_STATS.avg_cost(1:ctrl_idx), 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Cost (log)');
title('Min/Avg Cost');
legend('Min', 'Avg');
grid on;

% Saturation ratio
subplot(2,3,3);
plot(t_ctrl, MPPI_STATS.sat_ratio(1:ctrl_idx) * 100, 'LineWidth', 1.5);
hold on;
yline(50, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Saturation [%]');
title('Control Saturation Ratio');
ylim([0 100]);
grid on;

% Effective Sample Size
subplot(2,3,4);
plot(t_ctrl, MPPI_STATS.ess(1:ctrl_idx), 'LineWidth', 1.5);
hold on;
yline(mppi_params.K * 0.1, 'r--', 'LineWidth', 1);
yline(mppi_params.K * 0.01, 'r:', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('ESS');
title(sprintf('Effective Sample Size (K=%d)', mppi_params.K));
legend('ESS', '10% K', '1% K');
grid on;

% Execution time
subplot(2,3,5);
plot(t_ctrl, MPPI_STATS.exec_time(1:ctrl_idx) * 1000, 'LineWidth', 1.5);
hold on;
yline(dt_ctrl * 1000, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Exec Time [ms]');
title('MPPI Execution Time');
legend('Actual', sprintf('Budget (%.1fms)', dt_ctrl*1000));
grid on;

% XY Phase portrait
subplot(2,3,6);
plot(X(1,:), X(2,:), 'b-', 'LineWidth', 1);
hold on;
plot(waypoints(:,1), waypoints(:,2), 'ro-', 'MarkerSize', 8, 'LineWidth', 2);
plot(X(1,1), X(2,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X [m]'); ylabel('Y [m]');
title('XY Trajectory');
legend('Path', 'Waypoints', 'Start');
grid on; axis equal;

sgtitle('MPPI Position Control Diagnostics');

%% Print summary
fprintf('\n=== MPPI Position Control Summary ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Final position error: %.3f m\n', norm(X(1:3,end) - POS_DES(:,end)));
fprintf('Max position error: %.3f m\n', max(vecnorm(pos_err)));
fprintf('Max attitude: %.2f deg\n', max(abs(rad2deg(euler(:)))));
fprintf('Waypoints reached: %d/%d\n', wp_idx, n_wp);
fprintf('=====================================\n');

fprintf('\n=== MPPI Performance ===\n');
fprintf('Avg saturation: %.1f%%\n', mean(MPPI_STATS.sat_ratio(1:ctrl_idx))*100);
fprintf('Avg ESS: %.1f (%.2f%% of K)\n', mean(MPPI_STATS.ess(1:ctrl_idx)), ...
    mean(MPPI_STATS.ess(1:ctrl_idx))/mppi_params.K*100);
fprintf('Avg exec time: %.2f ms\n', mean(MPPI_STATS.exec_time(1:ctrl_idx))*1000);
fprintf('========================\n');
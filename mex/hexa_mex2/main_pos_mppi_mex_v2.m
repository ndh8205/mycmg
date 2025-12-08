%% main_pos_mppi_mex_v2.m - MPPI v2 Position Control Test
% Waypoint following with quadratic cost structure
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Check MEX
if ~exist('hexa_mppi_mex_v2', 'file')
    error('hexa_mppi_mex_v2 not found. Run: mexcuda hexa_mppi_mex_v2.cu -lcurand');
end

%% Parameters
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

ctrl_hz = 100;
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
wp_threshold = 1.0;  % [m] arrival threshold
yaw_des = 0;

%% MPPI Parameters (v2)
mppi_params.nu = 10.0;          % Fixed
mppi_params.K = 4096;
mppi_params.N = 50;
mppi_params.dt = 0.0100;
mppi_params.lambda = 1095.7288;
mppi_params.sigma = 80.6496;
mppi_params.R = 8.10e-08;
mppi_params.w_pos = 8922.3325;
mppi_params.w_vel = 684.9382;
mppi_params.w_att = 11978.3654;
mppi_params.w_omega = 1.5109;
mppi_params.w_terminal = 0.4010;
mppi_params.w_smooth = 0.0011;
mppi_params.crash_cost = 10000;
mppi_params.crash_angle = deg2rad(80);

mppi_params.K_pid = mppi_params.K * 0.2; % PD 롤아웃 개수
mppi_params.Kp_pos = [1; 1; 2];   % Position P gain
mppi_params.Kd_pos = [2; 2; 3];   % Position D gain
mppi_params.Kp_att = [8; 8; 6];   % Attitude P gain
mppi_params.Kd_att = [2; 2; 1.5]; % Attitude D gain
mppi_params.sigma_pid = 0.5;      % PD 게인 노이즈 스케일 (±20%)

%% Initial state
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));
fprintf('Hover motor speed: %.2f RPM\n', omega_hover * omega_bar2RPM);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -1];              % Start at WP1
x0(4:6)   = [0; 0; 0];
x0(7:10)  = [1; 0; 0; 0];             % Level attitude
x0(11:13) = [0; 0; 0];
x0(14:19) = omega_hover * ones(6,1);
x0(20:28) = zeros(9, 1);

%% Initialize MPPI state (v2 format)
mppi_state.u_seq = single(omega_hover * ones(6, mppi_params.N));
mppi_state.u_prev = single(omega_hover * ones(6, 1));
mppi_state.pos_des = single(waypoints(1,:)');
mppi_state.q_des = single([1; 0; 0; 0]);
mppi_state.omega_motor_est = omega_hover * ones(6,1);

%% Data storage
X = zeros(28, N_sim);
U = zeros(6, N_sim);
POS_DES = zeros(3, N_sim);
TAU_DIST = zeros(3, N_sim);
F_DIST = zeros(3, N_sim);
X(:,1) = x0;
POS_DES(:,1) = waypoints(1,:)';
u_current = omega_hover * ones(6,1);

% MPPI diagnostics
MPPI_COST = zeros(1, N_sim);
MPPI_ESS = zeros(1, N_sim);
MPPI_SAT = zeros(1, N_sim);
exec_times = [];

% Waypoint tracking
wp_reached_time = zeros(n_wp, 1);
wp_reached_time(1) = 0;

%% Simulation loop
fprintf('Running MPPI v2 position control (dist: %s)...\n', dist_preset);
tic_total = tic;

for k = 1:N_sim-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    
    % Waypoint arrival check
    pos_des = waypoints(wp_idx, :)';
    dist_to_wp = norm(pos_ned - pos_des);
    if dist_to_wp < wp_threshold && wp_idx < n_wp
        wp_idx = wp_idx + 1;
        wp_reached_time(wp_idx) = t(k);
        fprintf('  t=%.1fs: Reached WP%d, heading to WP%d\n', t(k), wp_idx-1, wp_idx);
    end
    pos_des = waypoints(wp_idx, :)';
    POS_DES(:,k) = pos_des;
    
    % MPPI control update
    if mod(k-1, ctrl_decimation) == 0
        % Update desired position
        mppi_state.pos_des = single(pos_des);
        
        % Update desired quaternion (level with yaw)
        q_des = GetQUAT(yaw_des, 0, 0);
        mppi_state.q_des = single(q_des(:));
        
        % Update motor state estimate
        for i = 1:6
            if mppi_state.u_prev(i) >= mppi_state.omega_motor_est(i)
                tau_i = params.drone.motor.tau_up;
            else
                tau_i = params.drone.motor.tau_down;
            end
            mppi_state.omega_motor_est(i) = mppi_state.omega_motor_est(i) + ...
                dt_ctrl * (mppi_state.u_prev(i) - mppi_state.omega_motor_est(i)) / tau_i;
        end
        
        % Shift control sequence
        mppi_state.u_seq(:, 1:end-1) = mppi_state.u_seq(:, 2:end);
        
        tic_mppi = tic;
        [u_opt, mppi_state] = hexa_mppi_controller_mex_v2(x_k, mppi_state, mppi_params, params);
        exec_time = toc(tic_mppi);
        exec_times(end+1) = exec_time;
        
        u_current = u_opt;
        
        % Log diagnostics
        MPPI_COST(k) = mppi_state.stats.min_cost;
        MPPI_ESS(k) = mppi_state.stats.effective_sample_size;
        MPPI_SAT(k) = mppi_state.stats.saturation_ratio;
    end
    
    % Apply control
    u = u_current;
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u = max(min(u, omega_max), omega_min);
    U(:,k) = u;
    
    % Log disturbance
    R_b2n = GetDCM_QUAT(q);
    if params.dist.enable
        [tau_d, ~] = dist_torque(t(k), params, dist_state);
        [F_d, ~] = dist_wind(vel_b, R_b2n, t(k), dt_sim, params, dist_state);
        TAU_DIST(:,k) = tau_d;
        F_DIST(:,k) = F_d;
    end
    
    % Integration
    [x_next, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x_k, u, Q, dt_sim, params_true, k, dt_sim, t(k), dist_state);
    
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);
POS_DES(:,end) = POS_DES(:,end-1);

total_time = toc(tic_total);
fprintf('Simulation complete. (%.1f s real time)\n', total_time);

%% Extract Euler angles
euler = zeros(3, N_sim);
for k = 1:N_sim
    euler(:,k) = Quat2Euler(X(7:10,k));
end

%% Position error
pos_err = X(1:3,:) - POS_DES;
pos_err_norm = vecnorm(pos_err);

%% Plot results
figure('Position', [100 100 1400 900], 'Name', 'MPPI v2 Position Control');

% X position
subplot(3,4,1);
plot(t, X(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, POS_DES(1,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('X [m]');
title('X Position'); grid on;
legend('Actual', 'Desired');

% Y position
subplot(3,4,2);
plot(t, X(2,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, POS_DES(2,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Y [m]');
title('Y Position'); grid on;
legend('Actual', 'Desired');

% Altitude
subplot(3,4,3);
plot(t, -X(3,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, -POS_DES(3,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude'); grid on;
legend('Actual', 'Desired');

% Position error
subplot(3,4,4);
plot(t, pos_err_norm, 'k', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('|Error| [m]');
title('Position Error'); grid on;

% Euler angles
subplot(3,4,5);
plot(t, rad2deg(euler));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi');
title('Euler Angles'); grid on;

% Angular velocity
subplot(3,4,6);
plot(t, rad2deg(X(11:13,:)));
xlabel('Time [s]'); ylabel('Rate [deg/s]');
legend('p','q','r');
title('Angular Velocity'); grid on;

% Motor speed
subplot(3,4,7);
plot(t, X(14:19,:) * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Speed [RPM]');
title('Motor Speed (Actual)'); grid on;

% 3D trajectory
subplot(3,4,8);
plot3(X(1,:), X(2,:), -X(3,:), 'b-', 'LineWidth', 1.5); hold on;
plot3(waypoints(:,1), waypoints(:,2), -waypoints(:,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot3(X(1,1), X(2,1), -X(3,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Alt [m]');
title('3D Trajectory'); grid on; axis equal;
legend('Path', 'Waypoints', 'Start');
view(45, 30);

% MPPI cost
subplot(3,4,9);
cost_idx = MPPI_COST > 0;
semilogy(t(cost_idx), MPPI_COST(cost_idx), 'b', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Cost');
title('MPPI Min Cost'); grid on;

% ESS
subplot(3,4,10);
ess_idx = MPPI_ESS > 0;
plot(t(ess_idx), MPPI_ESS(ess_idx), 'b', 'LineWidth', 1); hold on;
yline(mppi_params.K * 0.05, 'r--', '5% threshold');
xlabel('Time [s]'); ylabel('ESS');
title(sprintf('Effective Sample Size (K=%d)', mppi_params.K)); grid on;

% Saturation
subplot(3,4,11);
sat_idx = MPPI_SAT > 0;
plot(t(sat_idx), MPPI_SAT(sat_idx) * 100, 'b', 'LineWidth', 1); hold on;
yline(30, 'r--', '30% threshold');
xlabel('Time [s]'); ylabel('Saturation [%]');
title('Control Saturation'); grid on;

% XY trajectory (top view)
subplot(3,4,12);
plot(X(1,:), X(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(waypoints(:,1), waypoints(:,2), 'ro-', 'MarkerSize', 10, 'LineWidth', 2);
plot(X(1,1), X(2,1), 'gs', 'MarkerSize', 12, 'LineWidth', 2);
plot(X(1,end), X(2,end), 'mp', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('X [m]'); ylabel('Y [m]');
title('XY Trajectory (Top View)'); grid on; axis equal;
legend('Path', 'Waypoints', 'Start', 'End');

sgtitle(sprintf('MPPI v2 Position Control (Dist: %s)', dist_preset));

%% Print summary
fprintf('\n=== MPPI v2 Position Control Summary ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Waypoints reached: %d/%d\n', wp_idx, n_wp);
fprintf('Final position error: %.3f m\n', pos_err_norm(end));
fprintf('Max position error: %.3f m\n', max(pos_err_norm));
fprintf('RMSE position: %.3f m\n', sqrt(mean(pos_err_norm.^2)));
fprintf('Max attitude: %.2f deg\n', max(abs(rad2deg(euler(:)))));
fprintf('MPPI exec time: %.2f ± %.2f ms\n', mean(exec_times)*1000, std(exec_times)*1000);
fprintf('ESS: %.1f ± %.1f (%.1f%% of K)\n', mean(MPPI_ESS(ess_idx)), std(MPPI_ESS(ess_idx)), ...
    mean(MPPI_ESS(ess_idx))/mppi_params.K*100);
fprintf('Saturation: %.1f%% ± %.1f%%\n', mean(MPPI_SAT(sat_idx))*100, std(MPPI_SAT(sat_idx))*100);
fprintf('========================================\n');

%% Waypoint timing
fprintf('\n=== Waypoint Timing ===\n');
for i = 1:n_wp
    if wp_reached_time(i) > 0 || i == 1
        fprintf('WP%d: t=%.2f s\n', i, wp_reached_time(i));
    else
        fprintf('WP%d: Not reached\n', i);
    end
end

%% Export to CSV
data = [t', X(1:3,:)', POS_DES', rad2deg(euler)', X(11:13,:)', U', X(14:19,:)'];
header = {'t','pos_x','pos_y','pos_z','pos_des_x','pos_des_y','pos_des_z',...
          'euler_roll','euler_pitch','euler_yaw','omega_p','omega_q','omega_r',...
          'motor_cmd_1','motor_cmd_2','motor_cmd_3','motor_cmd_4','motor_cmd_5','motor_cmd_6',...
          'motor_actual_1','motor_actual_2','motor_actual_3','motor_actual_4','motor_actual_5','motor_actual_6'};
filename = sprintf('pos_mppi_v2_%s.csv', dist_preset);
fid = fopen(filename, 'w');
fprintf(fid, '%s,', header{1:end-1}); fprintf(fid, '%s\n', header{end});
fclose(fid);
dlmwrite(filename, data, '-append', 'precision', '%.6f');
fprintf('\nData saved to: %s\n', filename);
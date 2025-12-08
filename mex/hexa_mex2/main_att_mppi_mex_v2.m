%% main_att_mppi_mex_v2.m - Hexarotor MPPI Attitude Control (Improved Cost Function)
% CUDA MEX version with:
%   - Quadratic attitude cost (quaternion error)
%   - Smoothness penalty
%   - Crash detection
%   - Simplified parameters (9 instead of 12)
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Check MEX file
if ~exist('hexa_mppi_mex_v2', 'file')
    error('hexa_mppi_mex_v2 not found. Run: mexcuda hexa_mppi_mex_v2.cu -lcurand');
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
t_end = 20;
t = 0:dt_sim:t_end;
N_sim = length(t);

ctrl_hz = 100;  % MPPI control rate
ctrl_decimation = sim_hz / ctrl_hz;
dt_ctrl = 1 / ctrl_hz;

omega_bar2RPM = 60 / (2 * pi);

%% Process noise covariance
Q = zeros(9,9);

%% MPPI Parameters (Simplified - START HERE FOR TUNING)
% 
% Tuning Guide:
%   1. Start with attitude-only (w_pos=0, w_vel small)
%   2. Increase w_att until attitude stabilizes
%   3. Add w_omega if oscillating
%   4. Enable w_pos for position hold
%   5. Adjust w_smooth if jerky
%
% Core MPPI parameters
% mppi_params.nu = 10.0;          % Fixed
% mppi_params.K = 4096;
% mppi_params.N = 50;
% mppi_params.dt = 0.0100;
% mppi_params.lambda = 985.3825;
% mppi_params.sigma = 66.1663;
% mppi_params.R = 7.46e-08;
% mppi_params.w_pos = 8793.7940;
% mppi_params.w_vel = 418.6912;
% mppi_params.w_att = 8698.2582;
% mppi_params.w_omega = 1.1669;
% mppi_params.w_terminal = 0.7433;
% mppi_params.w_smooth = 0.0020;
% mppi_params.crash_cost = 10000;
% mppi_params.crash_angle = deg2rad(80);

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


mppi_params.K_pid = 32;           % PD 롤아웃 개수 (32의 배수)
mppi_params.Kp_pos = [1; 1; 2];   % Position P gain
mppi_params.Kd_pos = [2; 2; 3];   % Position D gain
mppi_params.Kp_att = [8; 8; 6];   % Attitude P gain
mppi_params.Kd_att = [2; 2; 1.5]; % Attitude D gain
mppi_params.sigma_pid = 0.2;      % PD 게인 노이즈 스케일 (±20%)

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
fprintf('Sigma: %.1f rad/s = %.1f RPM (%.1f%% of hover)\n', ...
    mppi_params.sigma, mppi_params.sigma * omega_bar2RPM, mppi_params.sigma/omega_hover*100);

% Initial euler: roll=20, pitch=15, yaw=10 deg (attitude recovery test)
euler0 = deg2rad([20; 15; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));
q0 = q0(:);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -1];              % position (NED), 10m altitude
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = q0;                       % quaternion (tilted)
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:19) = omega_hover * ones(6,1);  % motor speed (6 motors)
x0(20:28) = zeros(9, 1);              % biases

%% Desired states
pos_des = [0; 0; -10];                % Hold position
q_des = [1; 0; 0; 0];                 % Level attitude (hover)

%% Initialize MPPI state
mppi_state.u_seq = single(omega_hover * ones(6, mppi_params.N));
mppi_state.u_prev = single(omega_hover * ones(6, 1));  % NEW: for smoothness
mppi_state.pos_des = single(pos_des);
mppi_state.q_des = single(q_des);     % NEW: desired quaternion

%% Motor state estimator initialization
omega_motor_est = omega_hover * ones(6,1);

%% Data storage
X = zeros(28, N_sim);
U = zeros(6, N_sim);
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
u_current = omega_hover * ones(6,1);
ctrl_idx = 0;

%% Simulation loop
fprintf('\n=== MPPI v2 Attitude Control ===\n');
fprintf('Dist: %s\n', dist_preset);
fprintf('K=%d, N=%d, sigma=%.1f, lambda=%.1f\n', ...
    mppi_params.K, mppi_params.N, mppi_params.sigma, mppi_params.lambda);
fprintf('Weights: pos=%.1f, vel=%.1f, att=%.1f, omega=%.1f, smooth=%.2f\n', ...
    mppi_params.w_pos, mppi_params.w_vel, mppi_params.w_att, mppi_params.w_omega, mppi_params.w_smooth);
fprintf('================================\n\n');

tic_total = tic;
for k = 1:N_sim-1
    x_k = X(:,k);
    
    % MPPI control update at control rate
    if mod(k-1, ctrl_decimation) == 0
        ctrl_idx = ctrl_idx + 1;
        
        % Update motor state estimate (1st order dynamics)
        for i = 1:6
            if mppi_state.u_prev(i) >= omega_motor_est(i)
                tau_i = params.drone.motor.tau_up;
            else
                tau_i = params.drone.motor.tau_down;
            end
            omega_motor_est(i) = omega_motor_est(i) + dt_ctrl * (mppi_state.u_prev(i) - omega_motor_est(i)) / tau_i;
        end
        mppi_state.omega_motor_est = omega_motor_est;
        
        % Shift control sequence (warm start)
        mppi_state.u_seq(:, 1:end-1) = mppi_state.u_seq(:, 2:end);
        
        % Call MPPI MEX v2
        tic_mppi = tic;
        [u_opt, mppi_state] = hexa_mppi_controller_mex_v2(x_k, mppi_state, mppi_params, params);
        exec_time = toc(tic_mppi);
        
        u_current = u_opt;
        
        % Store diagnostics
        MPPI_STATS.min_cost(ctrl_idx) = mppi_state.stats.min_cost;
        MPPI_STATS.avg_cost(ctrl_idx) = mppi_state.stats.avg_cost;
        MPPI_STATS.cost_breakdown(:, ctrl_idx) = mppi_state.stats.cost_breakdown;
        MPPI_STATS.sat_ratio(ctrl_idx) = mppi_state.stats.saturation_ratio;
        MPPI_STATS.ess(ctrl_idx) = mppi_state.stats.effective_sample_size;
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

%% Plot results
figure('Position', [100 100 1400 900], 'Name', 'MPPI v2 Attitude Control');

% Altitude
subplot(3,3,1);
plot(t, -X(3,:), 'b', 'LineWidth', 1.5);
hold on;
yline(-pos_des(3), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude'); grid on;
legend('Actual', 'Desired');

% Euler angles
subplot(3,3,2);
plot(t, rad2deg(euler));
hold on;
yline(0, 'k--');
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi'); 
title('Euler Angles'); grid on;

% Angular velocity
subplot(3,3,3);
plot(t, rad2deg(X(11:13,:)));
xlabel('Time [s]'); ylabel('Angular Rate [deg/s]');
legend('p','q','r'); 
title('Angular Velocity (Body)'); grid on;

% Motor commands
subplot(3,3,4);
plot(t, U * omega_bar2RPM);
hold on;
yline(omega_hover * omega_bar2RPM, 'k--', 'Hover');
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
title('Motor Commands'); grid on;

% Motor speed (actual)
subplot(3,3,5);
plot(t, X(14:19,:) * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
title('Motor Speed (Actual)'); grid on;

% Horizontal position (drift check)
subplot(3,3,6);
plot(t, X(1:2,:));
xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y');
title('Horizontal Position (NED)'); grid on;

% Quaternion error (attitude tracking metric)
subplot(3,3,7);
q_err = zeros(1, N_sim);
for k = 1:N_sim
    q_k = X(7:10, k);
    q_dot = abs(q_des' * q_k);  % |q_des . q|
    q_err(k) = 2 * acos(min(q_dot, 1)) * 180/pi;  % Geodesic distance in deg
end
plot(t, q_err, 'b', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Attitude Error [deg]');
title('Quaternion Error (Geodesic)'); grid on;

% Cost breakdown
subplot(3,3,8);
t_ctrl = MPPI_STATS.time(1:ctrl_idx);
cost_labels = {'Pos', 'Vel', 'Att', 'Omega', 'Ctrl', 'Smooth'};
area(t_ctrl, MPPI_STATS.cost_breakdown(:, 1:ctrl_idx)');
xlabel('Time [s]'); ylabel('Cost');
title('Cost Breakdown'); grid on;
legend(cost_labels, 'Location', 'best');

% ESS
subplot(3,3,9);
plot(t_ctrl, MPPI_STATS.ess(1:ctrl_idx), 'LineWidth', 1.5);
hold on;
yline(mppi_params.K * 0.1, 'r--', '10% K');
yline(mppi_params.K * 0.01, 'r:', '1% K');
xlabel('Time [s]'); ylabel('ESS');
title(sprintf('Effective Sample Size (K=%d)', mppi_params.K)); grid on;

sgtitle(sprintf('MPPI v2 Attitude Control (Dist: %s)', dist_preset));

%% Figure 2: Phase Portraits
figure('Position', [150 150 1200 400], 'Name', 'Phase Portraits');

subplot(1,3,1);
plot(rad2deg(euler(1,:)), rad2deg(X(11,:)), 'b-', 'LineWidth', 0.5);
hold on;
plot(rad2deg(euler(1,1)), rad2deg(X(11,1)), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(0, 0, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Roll [deg]'); ylabel('Roll Rate [deg/s]');
title('Roll Phase Portrait'); grid on;
legend('Trajectory', 'Start', 'Target');

subplot(1,3,2);
plot(rad2deg(euler(2,:)), rad2deg(X(12,:)), 'b-', 'LineWidth', 0.5);
hold on;
plot(rad2deg(euler(2,1)), rad2deg(X(12,1)), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(0, 0, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Pitch [deg]'); ylabel('Pitch Rate [deg/s]');
title('Pitch Phase Portrait'); grid on;

subplot(1,3,3);
plot(rad2deg(euler(3,:)), rad2deg(X(13,:)), 'b-', 'LineWidth', 0.5);
hold on;
plot(rad2deg(euler(3,1)), rad2deg(X(13,1)), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(0, 0, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Yaw [deg]'); ylabel('Yaw Rate [deg/s]');
title('Yaw Phase Portrait'); grid on;

sgtitle('Phase Portraits (MPPI v2)');

%% Print final state
fprintf('\n=== Final State ===\n');
fprintf('Altitude: %.2f m (desired: %.2f m), error: %.3f m\n', -X(3,end), -pos_des(3), abs(-X(3,end) - (-pos_des(3))));
fprintf('Roll:  %.2f deg\n', rad2deg(euler(1,end)));
fprintf('Pitch: %.2f deg\n', rad2deg(euler(2,end)));
fprintf('Yaw:   %.2f deg\n', rad2deg(euler(3,end)));
fprintf('XY drift: %.3f m\n', norm(X(1:2,end)));
fprintf('Final attitude error: %.2f deg (geodesic)\n', q_err(end));

% Settling time analysis (2 deg threshold)
settling_threshold = 2;  % deg
settled = q_err < settling_threshold;
settle_idx = find(settled, 1, 'first');
if ~isempty(settle_idx) && all(settled(settle_idx:end))
    fprintf('Settling time (2 deg): %.2f s\n', t(settle_idx));
else
    fprintf('Settling time (2 deg): NOT SETTLED\n');
end

fprintf('===================\n');

%% Performance summary
fprintf('\n=== Performance Summary ===\n');
fprintf('Avg ESS: %.1f (%.1f%% of K)\n', mean(MPPI_STATS.ess(1:ctrl_idx)), ...
    mean(MPPI_STATS.ess(1:ctrl_idx))/mppi_params.K*100);
fprintf('Avg saturation: %.1f%%\n', mean(MPPI_STATS.sat_ratio(1:ctrl_idx))*100);
fprintf('Avg exec time: %.2f ms\n', mean(MPPI_STATS.exec_time(1:ctrl_idx))*1000);

% RMSE (last 50%)
ss_start = round(N_sim * 0.5);
rmse_att = sqrt(mean(q_err(ss_start:end).^2));
rmse_alt = sqrt(mean((-X(3, ss_start:end) - (-pos_des(3))).^2));
fprintf('RMSE (ss): att=%.3f deg, alt=%.4f m\n', rmse_att, rmse_alt);
fprintf('===========================\n');
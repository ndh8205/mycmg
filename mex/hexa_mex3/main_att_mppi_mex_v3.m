%% main_att_mppi_mex_v3.m - Hexarotor MPPI Attitude Control (SMC Reference)
% CUDA MEX version with:
%   - SMC reference rollouts (replaces PID)
%   - Position SMC: fractional-order sliding mode
%   - Attitude SMC: fractional-order sliding mode
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Check MEX file
if ~exist('hexa_mppi_mex_v3', 'file')
    error('hexa_mppi_mex_v3 not found. Run: mexcuda hexa_mppi_mex_v3.cu -lcurand');
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

%% MPPI Parameters with SMC gains (from tuned values)
% Core MPPI parameters
mppi_params.nu = 10.0;
mppi_params.K = 4096;
mppi_params.N = 50;
mppi_params.dt = 0.0100;
mppi_params.lambda = 1095.7288;
mppi_params.sigma = 80.6496;
% mppi_params.R = 8.10e-08;
mppi_params.R = 1e-2;

% Cost weights
mppi_params.w_pos = 8922.3325;
mppi_params.w_vel = 684.9382;
mppi_params.w_att = 11978.3654;
mppi_params.w_omega = 1.5109;
mppi_params.w_terminal = 0.4010;
mppi_params.w_smooth = 0.1;
mppi_params.crash_cost = 10000;
mppi_params.crash_angle = deg2rad(80);

% SMC rollout parameters
mppi_params.K_smc = mppi_params.K * 0.2; % SMC rollout 개수
mppi_params.sigma_smc = 0.6;      % SMC 게인 노이즈 스케일 (±20%)

% Position SMC gains (from main_pos_tuning_hexa_smc.m)
mppi_params.smc_pos_a = [10; 10; 10];           % 슬라이딩 면 게인
mppi_params.smc_pos_l1 = [0.8; 0.8; 0.8];       % 선형 도달 게인
mppi_params.smc_pos_l2 = [0.3; 0.3; 0.55];      % 분수차 게인
mppi_params.smc_pos_r = 0.98;                    % 분수차 지수

% Attitude SMC gains (from main_att_tuning_hexa_smc.m)
mppi_params.smc_att_a = 15;        % 선형 오차 게인
mppi_params.smc_att_b = 15;        % 분수차 오차 게인
mppi_params.smc_att_l1 = 5.4;      % 선형 도달 게인
mppi_params.smc_att_l2 = 5.4;      % 분수차 도달 게인
mppi_params.smc_att_r = 0.95;      % 분수차 지수

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
x0(1:3)   = [0; 0; -10];              % position (NED), 10m altitude
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
mppi_state.u_prev = single(omega_hover * ones(6, 1));
mppi_state.pos_des = single(pos_des);
mppi_state.q_des = single(q_des);

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
MPPI_STATS.best_is_smc = zeros(1, N_ctrl);

X(:,1) = x0;
u_current = omega_hover * ones(6,1);
ctrl_idx = 0;

%% Simulation loop
fprintf('\n=== MPPI v3 (SMC Reference) Attitude Control ===\n');
fprintf('Dist: %s\n', dist_preset);
fprintf('K=%d, K_smc=%d, N=%d, sigma=%.1f, lambda=%.1f\n', ...
    mppi_params.K, mppi_params.K_smc, mppi_params.N, mppi_params.sigma, mppi_params.lambda);
fprintf('SMC Pos: a=[%.1f,%.1f,%.1f], l1=[%.1f,%.1f,%.1f], r=%.2f\n', ...
    mppi_params.smc_pos_a(1), mppi_params.smc_pos_a(2), mppi_params.smc_pos_a(3), ...
    mppi_params.smc_pos_l1(1), mppi_params.smc_pos_l1(2), mppi_params.smc_pos_l1(3), ...
    mppi_params.smc_pos_r);
fprintf('SMC Att: a=%.1f, b=%.1f, l1=%.1f, l2=%.1f, r=%.2f\n', ...
    mppi_params.smc_att_a, mppi_params.smc_att_b, ...
    mppi_params.smc_att_l1, mppi_params.smc_att_l2, mppi_params.smc_att_r);
fprintf('==================================================\n\n');

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
        
        % Call MPPI MEX v3
        tic_mppi = tic;
        [u_opt, mppi_state] = hexa_mppi_controller_mex_v3(x_k, mppi_state, mppi_params, params);
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
        MPPI_STATS.best_is_smc(ctrl_idx) = mppi_state.stats.best_is_smc;
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
figure('Position', [100 100 1400 900], 'Name', 'MPPI v3 (SMC) Attitude Control');

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
    q_dot = abs(q_des' * q_k);
    q_err(k) = 2 * acos(min(q_dot, 1)) * 180/pi;
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

% SMC best ratio
subplot(3,3,9);
smc_ratio = movmean(MPPI_STATS.best_is_smc(1:ctrl_idx), 10) * 100;
plot(t_ctrl, smc_ratio, 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('SMC Best [%]');
title('SMC Reference Selection Rate'); grid on;
ylim([0 100]);

sgtitle(sprintf('MPPI v3 (SMC Reference) Attitude Control (Dist: %s)', dist_preset));

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

sgtitle('Phase Portraits (MPPI v3 - SMC Reference)');

%% Print final state
fprintf('\n=== Final State ===\n');
fprintf('Altitude: %.2f m (desired: %.2f m), error: %.3f m\n', -X(3,end), -pos_des(3), abs(-X(3,end) - (-pos_des(3))));
fprintf('Roll:  %.2f deg\n', rad2deg(euler(1,end)));
fprintf('Pitch: %.2f deg\n', rad2deg(euler(2,end)));
fprintf('Yaw:   %.2f deg\n', rad2deg(euler(3,end)));
fprintf('XY drift: %.3f m\n', norm(X(1:2,end)));
fprintf('Final attitude error: %.2f deg (geodesic)\n', q_err(end));

% Settling time analysis (2 deg threshold)
settling_threshold = 2;
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
fprintf('SMC best ratio: %.1f%%\n', mean(MPPI_STATS.best_is_smc(1:ctrl_idx))*100);

% RMSE (last 50%)
ss_start = round(N_sim * 0.5);
rmse_att = sqrt(mean(q_err(ss_start:end).^2));
rmse_alt = sqrt(mean((-X(3, ss_start:end) - (-pos_des(3))).^2));
fprintf('RMSE (ss): att=%.3f deg, alt=%.4f m\n', rmse_att, rmse_alt);
fprintf('===========================\n');
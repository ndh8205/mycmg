%% main_att_mppi_mex.m - Hexarotor MPPI Attitude & Altitude Control (CUDA MEX)
% Comprehensive diagnostics version
clear; clc; 
% close all;

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
t_end = 10;
t = 0:dt_sim:t_end;
N_sim = length(t);

ctrl_hz = 50;  % MPPI control rate
ctrl_decimation = sim_hz / ctrl_hz;
dt_ctrl = 1 / ctrl_hz;

omega_bar2RPM = 60 / (2 * pi);

%% Process noise covariance
Q = zeros(9,9);

% %% MPPI Parameters
% mppi_params.K = 4096*4;           % Number of samples
% mppi_params.N = 50;             % Horizon steps dt = 0.02
% mppi_params.dt = dt_ctrl;       % Control period
% % mppi_params.lambda = 25.0;      % Temperature
% mppi_params.nu = 10.0;          % Importance sampling parameter
% % mppi_params.sigma = 300 / omega_bar2RPM; % Noise std [RPM]
% 
% mppi_params.lambda = 103.03;
% mppi_params.sigma = 25.1705;  % 240.4 RPM

% % Cost weights
% mppi_params.w_pos_xy = 10.0;    % Position XY
% mppi_params.w_pos_z = 10.0;     % Position Z (altitude)
% mppi_params.w_vel_xy = 5.0;     % Velocity XY
% mppi_params.w_vel_z = 1.0;      % Velocity Z
% mppi_params.w_att = 5.0;       % Attitude (roll/pitch)
% mppi_params.w_yaw = 20.5;       % Yaw angle
% mppi_params.w_omega_rp = 1.0;  % Angular velocity roll/pitch
% mppi_params.w_omega_yaw = 5.5; % Angular velocity yaw
% mppi_params.w_terminal = 5.0;  % Terminal cost multiplier
% mppi_params.R = 0.0001;         % Control cost (very small)

% % Cost weights
% mppi_params.w_pos_xy = 10.0;    % Position XY
% mppi_params.w_pos_z = 10.0;     % Position Z (altitude)
% mppi_params.w_vel_xy = 5.0;     % Velocity XY
% mppi_params.w_vel_z = 1.0;      % Velocity Z
% mppi_params.w_att = 5.0;       % Attitude (roll/pitch)
% mppi_params.w_yaw = 4.0;       % Yaw angle
% mppi_params.w_omega_rp = 1.0;  % Angular velocity roll/pitch
% mppi_params.w_omega_yaw = 0; % Angular velocity yaw
% mppi_params.w_terminal = 5.0;  % Terminal cost multiplier
% mppi_params.R = 0.0001;         % Control cost (very small)

% mppi_params.w_pos_xy = 10.0;    % Position XY
% mppi_params.w_pos_z = 43.71;
% mppi_params.w_vel_xy = 5.0;     % Velocity XY
% mppi_params.w_vel_z = 0.87;
% mppi_params.w_att = 23.74;
% mppi_params.w_yaw = 18.28;
% mppi_params.w_omega_rp = 8.17;
% mppi_params.w_omega_yaw = 9.40;
% mppi_params.w_terminal = 5.0;  % Terminal cost multiplier
% mppi_params.R = 0.0001;         % Control cost (very small)

% mppi_params.K = 4096;
% mppi_params.N = 50;
% mppi_params.dt = dt_ctrl;
% mppi_params.lambda = 83.56;
% mppi_params.nu = 10.0;
% mppi_params.sigma = 28.7821;  % 274.8 RPM
% 
% mppi_params.w_pos_xy = 34.13;
% mppi_params.w_pos_z = 31.22;
% mppi_params.w_vel_xy = 10.27;
% mppi_params.w_vel_z = 19.44;
% mppi_params.w_att = 36.41;
% mppi_params.w_yaw = 17.83;
% mppi_params.w_omega_rp = 4.18;
% mppi_params.w_omega_yaw = 6.97;
% mppi_params.w_terminal = 1.46;
% mppi_params.R = 1.73e-05;

% === Copy to main_att_mppi_mex.m ===
mppi_params.K = 4096*8;
mppi_params.N = 50;
mppi_params.dt = dt_ctrl;
mppi_params.lambda = 13.69;
mppi_params.nu = 10.0;
mppi_params.sigma = 19.9805;  % 190.8 RPM
mppi_params.w_pos_xy = 8.64;
mppi_params.w_pos_z = 26.65;
mppi_params.w_vel_xy = 1.05;
mppi_params.w_vel_z = 1.32;
mppi_params.w_att = 11.11;
mppi_params.w_yaw = 28.44;
mppi_params.w_omega_rp = 5.61;
mppi_params.w_omega_yaw = 8.25;
mppi_params.w_terminal = 3.15;
mppi_params.R = 2.06e-05;


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

%% Initial state (28x1 for hexa)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));
fprintf('Hover motor speed: %.2f RPM (%.2f rad/s)\n', omega_hover * omega_bar2RPM, omega_hover);

% Initial euler: roll=20, pitch=15, yaw=10 deg (same as SMC test)
euler0 = deg2rad([20; 15; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));
q0 = q0(:);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -0.01];              % position (NED), 10m altitude
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = q0;                       % quaternion
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:19) = omega_hover * ones(6,1);  % motor speed (6 motors)
x0(20:28) = zeros(9, 1);              % biases

%% Desired states
pos_des = [0; 0; -10];  % maintain position
yaw_des = 0;            % level yaw

%% Initialize MPPI state
mppi_state.u_seq = single(omega_hover * ones(6, mppi_params.N));
mppi_state.pos_des = single(pos_des);
mppi_state.yaw_des = single(yaw_des);

%% Data storage
X = zeros(28, N_sim);
U = zeros(6, N_sim);
TAU_DIST = zeros(3, N_sim);
F_DIST = zeros(3, N_sim);

% MPPI diagnostics
N_ctrl = ceil(N_sim / ctrl_decimation);
MPPI_STATS.min_cost = zeros(1, N_ctrl);
MPPI_STATS.avg_cost = zeros(1, N_ctrl);
MPPI_STATS.cost_breakdown = zeros(6, N_ctrl);  % [pos, vel, att, yaw, omega, ctrl]
MPPI_STATS.sat_ratio = zeros(1, N_ctrl);
MPPI_STATS.ess = zeros(1, N_ctrl);
MPPI_STATS.time = zeros(1, N_ctrl);
MPPI_STATS.exec_time = zeros(1, N_ctrl);

X(:,1) = x0;
u_current = omega_hover * ones(6,1);
ctrl_idx = 0;

%% Simulation loop
fprintf('Running MPPI simulation (dist: %s)...\n', dist_preset);
fprintf('K=%d, N=%d, sigma=%.1f, lambda=%.1f\n', ...
    mppi_params.K, mppi_params.N, mppi_params.sigma, mppi_params.lambda);

tic_total = tic;
for k = 1:N_sim-1
    x_k = X(:,k);
    
    % MPPI control update at control rate
    if mod(k-1, ctrl_decimation) == 0
        ctrl_idx = ctrl_idx + 1;
        
        % Prepare state for MEX (13x1, single precision)
        x_mppi = single([x_k(1:3); x_k(4:6); x_k(7:10); x_k(11:13)]);
        
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
        pos_ned = x_k(1:3);
        vel_b = x_k(4:6);
        q = x_k(7:10);
        R_b2n = GetDCM_QUAT(q);
        [tau_d, ~] = dist_torque(t(k), params, dist_state);
        [F_d, ~] = dist_wind(vel_b, R_b2n, t(k), dt_sim, params, dist_state);
        TAU_DIST(:,k) = tau_d;
        F_DIST(:,k) = F_d;
    end
    
    % Integration (use params_true for dynamics)
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

%% ==================== PLOTTING ====================

%% Figure 1: Main Results
figure('Position', [50 50 1400 900], 'Name', 'MPPI Attitude Control Results');

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
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi');
title('Euler Angles'); grid on;
yline(0, 'k--');

% Angular velocity
subplot(3,3,3);
plot(t, rad2deg(X(11:13,:)));
xlabel('Time [s]'); ylabel('Angular Rate [deg/s]');
legend('p','q','r');
title('Angular Velocity (Body)'); grid on;

% Motor commands with envelope
subplot(3,3,4);
plot(t, U * omega_bar2RPM);
hold on;
yline(params.drone.motor.omega_b_max * omega_bar2RPM, 'r--', 'LineWidth', 1);
yline(params.drone.motor.omega_b_min * omega_bar2RPM, 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Commands'); grid on;

% Motor speed (actual)
subplot(3,3,5);
plot(t, X(14:19,:) * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Speed (Actual)'); grid on;

% Horizontal position (drift check)
subplot(3,3,6);
plot(t, X(1:2,:));
xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y');
title('Horizontal Position (NED)'); grid on;

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

% Position error magnitude
subplot(3,3,9);
plot(t, vecnorm(X(1:2,:)), 'b', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('|XY Error| [m]');
title('Horizontal Drift'); grid on;

sgtitle(sprintf('Hexarotor MPPI Control (Dist: %s, K=%d)', dist_preset, mppi_params.K));

%% Figure 2: MPPI Diagnostics
figure('Position', [100 100 1400 700], 'Name', 'MPPI Diagnostics');

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
if mean(MPPI_STATS.sat_ratio(1:ctrl_idx)) > 0.3
    text(t_ctrl(end)/2, 70, 'WARNING: High saturation!', 'Color', 'r', 'FontWeight', 'bold');
end

% Effective Sample Size
subplot(2,3,4);
plot(t_ctrl, MPPI_STATS.ess(1:ctrl_idx), 'LineWidth', 1.5);
hold on;
yline(mppi_params.K * 0.1, 'r--', 'LineWidth', 1);  % 10% threshold
yline(mppi_params.K * 0.01, 'r:', 'LineWidth', 1);  % 1% threshold
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

% Phase portrait (roll error vs roll rate)
subplot(2,3,6);
plot(rad2deg(euler(1,:)), rad2deg(X(11,:)), 'b-', 'LineWidth', 0.5);
hold on;
plot(rad2deg(euler(1,1)), rad2deg(X(11,1)), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(0, 0, 'rx', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Roll [deg]'); ylabel('Roll Rate [deg/s]');
title('Phase Portrait (Roll)');
legend('Trajectory', 'Start', 'Target');
grid on; axis equal;

sgtitle('MPPI Diagnostics');

%% Figure 3: Phase Portraits (All States)
figure('Position', [150 50 1200 900], 'Name', 'Phase Portraits');

% Roll
subplot(2,3,1);
plot(rad2deg(euler(1,:)), rad2deg(X(11,:)), 'b-', 'LineWidth', 0.5);
hold on;
plot(rad2deg(euler(1,1)), rad2deg(X(11,1)), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(0, 0, 'rx', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Roll [deg]'); ylabel('Roll Rate [deg/s]');
title('Roll'); grid on;

% Pitch
subplot(2,3,2);
plot(rad2deg(euler(2,:)), rad2deg(X(12,:)), 'b-', 'LineWidth', 0.5);
hold on;
plot(rad2deg(euler(2,1)), rad2deg(X(12,1)), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(0, 0, 'rx', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Pitch [deg]'); ylabel('Pitch Rate [deg/s]');
title('Pitch'); grid on;

% Yaw
subplot(2,3,3);
plot(rad2deg(euler(3,:)), rad2deg(X(13,:)), 'b-', 'LineWidth', 0.5);
hold on;
plot(rad2deg(euler(3,1)), rad2deg(X(13,1)), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(rad2deg(yaw_des), 0, 'rx', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Yaw [deg]'); ylabel('Yaw Rate [deg/s]');
title('Yaw'); grid on;

% Altitude (z position vs z velocity)
vel_ned = zeros(3, N_sim);
for kk = 1:N_sim
    R_b2n = GetDCM_QUAT(X(7:10,kk));
    vel_ned(:,kk) = R_b2n * X(4:6,kk);
end
subplot(2,3,4);
plot(-X(3,:), -vel_ned(3,:), 'b-', 'LineWidth', 0.5);
hold on;
plot(-X(3,1), -vel_ned(3,1), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(-pos_des(3), 0, 'rx', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Altitude [m]'); ylabel('Vertical Vel [m/s]');
title('Altitude'); grid on;

% X position
subplot(2,3,5);
plot(X(1,:), vel_ned(1,:), 'b-', 'LineWidth', 0.5);
hold on;
plot(X(1,1), vel_ned(1,1), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(pos_des(1), 0, 'rx', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('X [m]'); ylabel('X Vel [m/s]');
title('X Position'); grid on;

% Y position
subplot(2,3,6);
plot(X(2,:), vel_ned(2,:), 'b-', 'LineWidth', 0.5);
hold on;
plot(X(2,1), vel_ned(2,1), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(pos_des(2), 0, 'rx', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Y [m]'); ylabel('Y Vel [m/s]');
title('Y Position'); grid on;

sgtitle('Phase Portraits (All States)');
% 
% %% Print summary
% fprintf('\n=== MPPI Final State ===\n');
% fprintf('Preset: %s\n', dist_preset);
% fprintf('Altitude: %.2f m (desired: %.2f m)\n', -X(3,end), -pos_des(3));
% fprintf('Roll: %.2f deg\n', rad2deg(euler(1,end)));
% fprintf('Pitch: %.2f deg\n', rad2deg(euler(2,end)));
% fprintf('Yaw: %.2f deg\n', rad2deg(euler(3,end)));
% fprintf('XY drift: %.2f m\n', norm(X(1:2,end)));
% fprintf('========================\n');
% 
% fprintf('\n=== MPPI Performance ===\n');
% fprintf('Avg saturation: %.1f%%\n', mean(MPPI_STATS.sat_ratio(1:ctrl_idx))*100);
% fprintf('Avg ESS: %.1f (%.2f%% of K)\n', mean(MPPI_STATS.ess(1:ctrl_idx)), ...
%     mean(MPPI_STATS.ess(1:ctrl_idx))/mppi_params.K*100);
% fprintf('Avg exec time: %.2f ms\n', mean(MPPI_STATS.exec_time(1:ctrl_idx))*1000);
% fprintf('========================\n');
% 
%% Settling time analysis
settling_threshold = 2;  % deg
settled = abs(rad2deg(euler)) < settling_threshold;
settling_time = zeros(3,1);
for i = 1:3
    idx = find(settled(i,:) & all(settled(i,:)), 1, 'first');
    if ~isempty(idx)
        settling_time(i) = t(idx);
    else
        settling_time(i) = t_end;
    end
end
fprintf('\nSettling time (2 deg threshold):\n');
fprintf('  Roll:  %.2f s\n', settling_time(1));
fprintf('  Pitch: %.2f s\n', settling_time(2));
fprintf('  Yaw:   %.2f s\n', settling_time(3));
% 
% %% Export to CSV
% data = [t', X(1:3,:)', rad2deg(euler)', X(11:13,:)', U', X(14:19,:)', ...
%         TAU_DIST', F_DIST'];
% header = {'t','pos_x','pos_y','pos_z','euler_roll','euler_pitch','euler_yaw',...
%           'omega_p','omega_q','omega_r','motor_cmd_1','motor_cmd_2','motor_cmd_3',...
%           'motor_cmd_4','motor_cmd_5','motor_cmd_6','motor_actual_1','motor_actual_2',...
%           'motor_actual_3','motor_actual_4','motor_actual_5','motor_actual_6',...
%           'tau_dist_x','tau_dist_y','tau_dist_z','F_dist_x','F_dist_y','F_dist_z'};
% filename = sprintf('att_mppi_%s.csv', dist_preset);
% fid = fopen(filename, 'w');
% fprintf(fid, '%s,', header{1:end-1}); fprintf(fid, '%s\n', header{end});
% fclose(fid);
% dlmwrite(filename, data, '-append', 'precision', '%.6f');
% fprintf('Data saved to: %s\n', filename);
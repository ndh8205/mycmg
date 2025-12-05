%% main_smc_test.m - Hexarotor SMC Attitude & Altitude Control Test
% Fractional-order Sliding Mode Controller based on paper
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Parameters (Hexarotor)
params = params_init('hexa');

%% Disturbance settings
% dist_preset = 'nominal';
% dist_preset = 'level1';
dist_preset = 'level2';
% dist_preset = 'level3';
% dist_preset = 'paper';

[params, dist_state] = dist_init(params, dist_preset);
if params.dist.uncertainty.enable
    params_true = apply_uncertainty(params);
else
    params_true = params;
end

%% Simulation settings
sim_hz = 1000;
dt = 1/sim_hz;
t_end = 10;
t = 0:dt:t_end;
N = length(t);

omega_bar2RPM = 60 / (2 * pi);

%% Process noise covariance
Q = zeros(9,9);

%% SMC Gains (from paper)

% Attitude SMC gains (Fractional-order SMC)
% 슬라이딩 면: s = ω + a*e_v + b*|e_v|^r * sign(e_v)
% 도달 법칙:  ṡ = -λ₁*s - λ₂*|s|^r * sign(s)

gains_att_smc.a = 15;        % 선형 오차 게인: 클수록 빠른 수렴, 너무 크면 오버슈트
gains_att_smc.b = 15;        % 분수차 오차 게인: 수렴 초기 가속, 정상상태 정밀도 향상
gains_att_smc.lambda1 = 5.4; % 선형 도달 게인: 슬라이딩 면 도달 속도 (채터링 vs 속도 트레이드오프)
gains_att_smc.lambda2 = 5.4; % 분수차 도달 게인: 슬라이딩 면 근처 수렴 가속
gains_att_smc.r = 0.95;     % 분수차 지수 (0<r<1): 1에 가까울수록 선형, 작을수록 finite-time 수렴

% Altitude SMC gains (Fractional-order SMC)
% 슬라이딩 면: s = -ż + a*|e|^r * sign(e)
% 도달 법칙:  ṡ = -λ₁*s - λ₂*|s|^r * sign(s)

gains_alt_smc.a = 10;        % 고도 오차 게인: 클수록 공격적 응답
gains_alt_smc.lambda1 = 10;   % 선형 도달 게인: 고도 수렴 속도
gains_alt_smc.lambda2 = 0.8; % 분수차 도달 게인: 정상상태 근처 수렴 가속
gains_alt_smc.r = 0.98;     % 분수차 지수: 0.98은 거의 선형에 가까움 (부드러운 응답)

%% Control state initialization (for interface compatibility)
ctrl_state.int_att = zeros(3,1);
ctrl_state.int_alt = 0;

%% Initial state (28x1 for hexa)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));
fprintf('Hover motor speed: %.2f RPM\n', omega_hover * omega_bar2RPM);

% Initial euler: roll=20, pitch=15, yaw=10 deg (same as paper test)
euler0 = deg2rad([20; 15; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));
q0 = q0(:);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -10];              % position (NED), 10m altitude
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = q0;                       % quaternion
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:19) = omega_hover * ones(6,1);  % motor speed (6 motors)
x0(20:28) = zeros(9, 1);              % biases

%% Desired states
q_des = [1; 0; 0; 0];   % level attitude
alt_des = 10;           % maintain 10m altitude

%% Data storage
X = zeros(28, N);
U = zeros(6, N);
TAU_CMD = zeros(3, N);
TAU_DIST = zeros(3, N);
F_DIST = zeros(3, N);
S_ATT = zeros(3, N);  % sliding surface
X(:,1) = x0;

%% Simulation loop
fprintf('Running SMC simulation (dist: %s)...\n', dist_preset);
for k = 1:N-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    omega = x_k(11:13);
    
    % Current altitude (positive up)
    alt = -pos_ned(3);
    
    % Vertical velocity (NED -> positive up)
    R_b2n = GetDCM_QUAT(q);
    vel_ned = R_b2n * vel_b;
    vel_z = -vel_ned(3);
    
    % SMC Attitude control
    [tau_cmd, ctrl_state] = attitude_smc(q_des, q, omega, ctrl_state, gains_att_smc, params);
    TAU_CMD(:,k) = tau_cmd;
    
    % SMC Altitude control
    [thrust_cmd, ctrl_state] = altitude_smc(alt_des, alt, vel_z, q, ctrl_state, gains_alt_smc, dt, params);
    
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
TAU_CMD(:,end) = TAU_CMD(:,end-1);
fprintf('Simulation complete.\n');

%% Extract Euler angles
euler = zeros(3,N);
for k = 1:N
    euler(:,k) = Quat2Euler(X(7:10,k));
end

%% Plot results
figure('Position', [100 100 1400 900], 'Name', 'SMC Attitude & Altitude Control');

% Altitude
subplot(3,3,1);
plot(t, -X(3,:), 'b', 'LineWidth', 1.5);
hold on;
yline(alt_des, 'r--', 'LineWidth', 1.5);
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

% Control torque command
subplot(3,3,4);
plot(t, TAU_CMD');
xlabel('Time [s]'); ylabel('Torque [Nm]');
legend('\tau_x','\tau_y','\tau_z');
title('SMC Torque Command'); grid on;

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

sgtitle(sprintf('Hexarotor SMC Control (Dist: %s)', dist_preset));

%% Print final state
fprintf('\n=== SMC Final State ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Altitude: %.2f m (desired: %.2f m)\n', -X(3,end), alt_des);
fprintf('Roll: %.2f deg\n', rad2deg(euler(1,end)));
fprintf('Pitch: %.2f deg\n', rad2deg(euler(2,end)));
fprintf('Yaw: %.2f deg\n', rad2deg(euler(3,end)));
fprintf('XY drift: %.2f m\n', norm(X(1:2,end)));
fprintf('Max attitude error: %.2f deg\n', max(abs(rad2deg(euler(:)))));
fprintf('========================\n');

%% Performance comparison - settling time
% Find settling time (within 2 deg of desired)
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
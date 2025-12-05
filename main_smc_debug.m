%% main_smc_debug.m - SMC vs PID 비교 디버그
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Parameters
params = params_init('hexa');
dist_preset = 'nominal';
[params, dist_state] = dist_init(params, dist_preset);
params_true = params;

%% Simulation settings
dt = 0.001;
t_end = 10;
t = 0:dt:t_end;
N = length(t);
Q = zeros(9,9);

%% Test mode: 'smc' or 'pid'
test_mode = 'smc';  % 'smc' 또는 'pid'

%% Gains
% SMC (축별 게인: [X; Y; Z])
% PID보다 보수적으로 시작
gains_pos_smc.a = [10; 10; 10];          % 슬라이딩 면 게인
gains_pos_smc.lambda1 = [0.8; 0.8; 0.8];  % 선형 도달 게인
gains_pos_smc.lambda2 = [0.3; 0.3; 0.55];  % 분수차 게인 (작게!)
gains_pos_smc.r = 0.98;

gains_att_smc.a = 15;
gains_att_smc.b = 15;
gains_att_smc.lambda1 = 5.4;
gains_att_smc.lambda2 = 5.4;
gains_att_smc.r = 0.95;

% PID
gains_pos_pid.Kp = [2;2;4];
gains_pos_pid.Ki = [0.1;0.1;0.2];
gains_pos_pid.Kd = [1.5;1.5;2];
gains_pos_pid.int_limit = [2;2;3];

gains_att_pid.Kp = [8;8;6];
gains_att_pid.Ki = [0.1;0.1;0.1];
gains_att_pid.Kd = [2;2;1.5];
gains_att_pid.int_limit = [1;1;1];

%% Initial state
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;
omega_hover = sqrt(m * g / (n_motor * k_T));

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -0.1];
x0(7:10)  = [1; 0; 0; 0];
x0(14:19) = omega_hover * ones(6,1);

%% Target
pos_des = [5; 5; -10];  % 간단한 X방향 이동
yaw_des = 0;

%% Control state
ctrl_state.int_pos = zeros(3,1);
ctrl_state.int_att = zeros(3,1);
ctrl_state.dt = dt;

%% Storage
X = zeros(28, N);
A_DES = zeros(3, N);
Q_DES = zeros(4, N);
THRUST = zeros(1, N);
X(:,1) = x0;

%% Simulation
fprintf('Test mode: %s\n', test_mode);
fprintf('Target: [%.1f, %.1f, %.1f]\n', pos_des(1), pos_des(2), pos_des(3));

for k = 1:N-1
    x_k = X(:,k);
    
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    omega = x_k(11:13);
    
    R_b2n = GetDCM_QUAT(q);
    vel_ned = R_b2n * vel_b;
    
    %% Position control
    if strcmp(test_mode, 'smc')
        [q_des, thrust_cmd, ctrl_state] = position_smc(...
            pos_des, pos_ned, vel_ned, yaw_des, ctrl_state, gains_pos_smc, params);
    else
        [q_des, thrust_cmd, ctrl_state] = position_pid(...
            pos_des, pos_ned, vel_ned, yaw_des, ctrl_state, gains_pos_pid, params);
    end
    
    Q_DES(:,k) = q_des;
    THRUST(k) = thrust_cmd;
    
    %% Attitude control
    if strcmp(test_mode, 'smc')
        [tau_cmd, ctrl_state] = attitude_smc(q_des, q, omega, ctrl_state, gains_att_smc, params);
    else
        [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains_att_pid, dt);
    end
    
    %% Control allocation
    cmd_vec = [thrust_cmd; tau_cmd];
    omega_sq = control_allocator(cmd_vec, params, 'inverse');
    omega_sq = max(omega_sq, 0);
    u = sqrt(omega_sq);
    u = max(min(u, params.drone.motor.omega_b_max), params.drone.motor.omega_b_min);
    
    %% Integration
    [x_next, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x_k, u, Q, dt, params_true, k, dt, t(k), dist_state);
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
    
    %% Debug print
    if k == 1 || mod(k, 2000) == 0
        e_pos = pos_des - pos_ned;
        euler_des = Quat2Euler(q_des);
        fprintf('t=%.1fs: pos=[%.2f,%.2f,%.2f], e=[%.2f,%.2f,%.2f], T=%.1f, q_des=[%.3f,%.3f,%.3f,%.3f], euler_des=[%.1f,%.1f,%.1f]deg\n', ...
            t(k), pos_ned(1), pos_ned(2), pos_ned(3), ...
            e_pos(1), e_pos(2), e_pos(3), thrust_cmd, ...
            q_des(1), q_des(2), q_des(3), q_des(4), ...
            rad2deg(euler_des(1)), rad2deg(euler_des(2)), rad2deg(euler_des(3)));
    end
end

%% Plot
figure('Position', [100 100 1200 600]);

subplot(2,3,1);
plot(t, X(1,:), 'b', t, X(2,:), 'r', t, -X(3,:), 'g', 'LineWidth', 1.5);
hold on;
yline(pos_des(1), 'b--'); yline(pos_des(2), 'r--'); yline(-pos_des(3), 'g--');
xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y','alt','x_{des}','y_{des}','alt_{des}');
title('Position'); grid on;

subplot(2,3,2);
euler = zeros(3,N);
for k = 1:N
    euler(:,k) = Quat2Euler(X(7:10,k));
end
plot(t, rad2deg(euler));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi');
title('Euler Angles'); grid on;

subplot(2,3,3);
plot(t, THRUST, 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Thrust [N]');
title('Thrust Command'); grid on;
yline(m*g, 'r--', 'hover');

subplot(2,3,4);
euler_des = zeros(3,N);
for k = 1:N
    euler_des(:,k) = Quat2Euler(Q_DES(:,k));
end
plot(t, rad2deg(euler_des));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi_{des}','\theta_{des}','\psi_{des}');
title('Desired Euler (from pos ctrl)'); grid on;

subplot(2,3,5);
pos_err = [pos_des(1) - X(1,:); pos_des(2) - X(2,:); pos_des(3) - X(3,:)];
plot(t, pos_err');
xlabel('Time [s]'); ylabel('Error [m]');
legend('e_x','e_y','e_z');
title('Position Error'); grid on;

subplot(2,3,6);
plot(t, vecnorm(pos_err), 'k', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('|Error| [m]');
title('Position Error Norm'); grid on;

sgtitle(sprintf('Position Control Debug - Mode: %s', upper(test_mode)));

fprintf('\n=== Final ===\n');
fprintf('Final pos: [%.2f, %.2f, %.2f]\n', X(1,end), X(2,end), -X(3,end));
fprintf('Final error: %.3f m\n', norm(X(1:3,end) - pos_des));
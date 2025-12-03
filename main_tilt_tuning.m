%% test_tilt_simple.m - Simple Tilt Test
clear; clc; close all;
addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));

%% Parameters
params = params_tiltrotor_init();

%% Simulation settings
dt = 0.002;
t_end = 10;
t = 0:dt:t_end;
N = length(t);
Q = zeros(9,9);

%% Initial state
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
omega_hover = sqrt(m * g / (4 * k_T));

x0 = zeros(30, 1);
x0(1:3) = [0; 0; -20];
x0(7:10) = [1; 0; 0; 0];
x0(14:17) = omega_hover * ones(4,1);

%% Storage
X = zeros(30, N);
X(:,1) = x0;

%% Control gains
gains_att.Kp = [5; 5; 5];
gains_att.Ki = [0; 0; 0];
gains_att.Kd = [1; 1; 0.5];
gains_att.int_limit = [1; 1; 1];

ctrl_state.int_att = zeros(3,1);

%% Simple control: hover + ramp tilt at t=3s
THRUST_CMD = zeros(1, N);
OMEGA_SQ = zeros(4, N);

for k = 1:N-1
    x_k = X(:,k);
    t_k = t(k);
    tilt = x_k(27:28);
    q = x_k(7:10);
    omega = x_k(11:13);
    
    % Tilt command: ramp from 0 to 45 deg after t=3s
    if t_k < 3
        tilt_cmd = [0; 0];
    else
        tilt_target = min((t_k - 3) / 3 * 45, 45) * pi/180;
        tilt_cmd = [tilt_target; tilt_target];
    end
    
    % Hover thrust
    thrust_cmd = m * g;
    
    % Attitude control to stay level
    q_des = [1; 0; 0; 0];
    [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains_att, dt);
    
    THRUST_CMD(k) = thrust_cmd;
    
    % Allocate
    cmd_vec = [thrust_cmd; tau_cmd];
    [omega_sq, B] = tiltrotor_allocator(cmd_vec, tilt, params, 'inverse');
    
    OMEGA_SQ(:,k) = omega_sq;
    
    % Debug: check if negative
    if any(omega_sq < 0)
        fprintf('t=%.2f: negative omega_sq detected!\n', t_k);
    end
    
    omega_sq = max(omega_sq, 0);
    u_motor = sqrt(omega_sq);
    u_motor = max(min(u_motor, 1500), 0);
    
    % Input
    u = zeros(9,1);
    u(1:4) = u_motor;
    u(5:6) = tilt_cmd;
    
    % Integrate
    [x_next, ~, ~, ~, ~, ~, ~] = srk4(@tiltrotor_dynamics, x_k, u, Q, dt, params, k, dt);
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    X(:,k+1) = x_next;
end
OMEGA_SQ(:,end) = OMEGA_SQ(:,end-1);
THRUST_CMD(end) = THRUST_CMD(end-1);

%% Check allocator at tilt=45deg
fprintf('\n=== Allocator Check at tilt=45deg ===\n');
tilt_test = [45; 45] * pi/180;
cmd_test = [m*g; 0; 0; 0];
[omega_sq_test, B_test] = tiltrotor_allocator(cmd_test, tilt_test, params, 'inverse');
fprintf('Thrust cmd: %.2f N\n', m*g);
fprintf('omega_sq: [%.0f, %.0f, %.0f, %.0f]\n', omega_sq_test);
fprintf('omega: [%.0f, %.0f, %.0f, %.0f] rad/s\n', sqrt(max(omega_sq_test,0)));
fprintf('B matrix:\n');
disp(B_test);

%% Plot
figure('Position', [100 100 1200 600]);

subplot(2,3,1);
plot(t, -X(3,:)); ylabel('Alt [m]'); xlabel('Time [s]');
title('Altitude'); grid on;

subplot(2,3,2);
plot(t, rad2deg(X(27:28,:)));
ylabel('Tilt [deg]'); xlabel('Time [s]');
legend('Tilt0', 'Tilt2'); title('Tilt Angles'); grid on;

subplot(2,3,3);
plot(t, X(4,:)); ylabel('Vx [m/s]'); xlabel('Time [s]');
title('Forward Velocity (Body)'); grid on;

subplot(2,3,4);
euler = zeros(3,N);
for i = 1:N
    euler(:,i) = Quat2Euler(X(7:10,i));
end
plot(t, rad2deg(euler));
ylabel('Angle [deg]'); xlabel('Time [s]');
legend('\phi','\theta','\psi'); title('Euler Angles'); grid on;

subplot(2,3,5);
plot(t, X(14:17,:));
ylabel('\omega [rad/s]'); xlabel('Time [s]');
legend('M1','M2','M3','M4'); title('Motor Speeds (Actual)'); grid on;

subplot(2,3,6);
plot(t, OMEGA_SQ);
ylabel('\omega^2'); xlabel('Time [s]');
legend('M1','M2','M3','M4'); title('Allocator Output (omega\_sq)'); grid on;

sgtitle('Simple Tilt Test');
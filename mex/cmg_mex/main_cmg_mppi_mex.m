%% main_cmg_mppi_mex.m - CMG MPPI Control (CUDA MEX Version)
%
% 실행 전 컴파일 필요: run compile_mex.m

clear; clc; close all;

%% 컴파일 확인
if ~exist('cmg_mppi_mex', 'file')
    fprintf('MEX not compiled. Running compile_mex...\n');
    compile_mex;
end

%% Parameters
params = cmg_params_init();

%% MPPI Parameters
% mppi_params.K = 4096;
mppi_params.N = 100;
mppi_params.dt = 0.01;
mppi_params.lambda = 0.5;
mppi_params.nu = 10;
mppi_params.sigma = [0.5; 0.5];
mppi_params.R = [0.1; 0.1];
mppi_params.Q_att = [0; 100000; 100000];
mppi_params.Q_omega = [10; 10; 10];
mppi_params.S_weight = 10.5;
mppi_params.w_terminal = 1000;

%% Simulation
dt = params.dt;
t_end = 300;
N = round(t_end / dt) + 1;
t = (0:N-1) * dt;

%% Initial state [roll; pitch; yaw; p; q; r; g1; g2]
x = zeros(8, 1);
x(7) = deg2rad(0);
x(8) = deg2rad(180);

%% MPPI state
mppi_state.u_seq = zeros(2, mppi_params.N);
mppi_state.att_des = [0; 0; 0];

%% Preallocate
X = zeros(8, N, 'single');
U = zeros(2, N, 'single');
ATT_DES = zeros(3, N, 'single');
X(:, 1) = single(x);

%% Target attitudes
att_targets = zeros(3, 6);
att_targets(:,1) = [0; deg2rad(20); deg2rad(-30)];
att_targets(:,2) = [0; 0; 0];
att_targets(:,3) = [0; deg2rad(-15); deg2rad(20)];
att_targets(:,4) = [0; deg2rad(10); deg2rad(10)];
att_targets(:,5) = [0; deg2rad(-20); deg2rad(-15)];
att_targets(:,6) = [0; 0; 0];

target_idx = min(floor(t / 50) + 1, 6);

%% Simulation loop
fprintf('Running CMG MPPI (CUDA MEX)...\n');
fprintf('K=%d, N=%d, t_end=%.0fs\n', mppi_params.K, mppi_params.N, t_end);
tic;
mppi_time = 0;

for k = 1:N-1
    % Target
    mppi_state.att_des = att_targets(:, target_idx(k));
    ATT_DES(:, k) = single(mppi_state.att_des);
    
    % MPPI (CUDA MEX)
    t_mppi = tic;
    [u, mppi_state] = cmg_mppi_controller_mex(double(X(:,k)), mppi_state, mppi_params, params);
    mppi_time = mppi_time + toc(t_mppi);
    
    % Saturate
    u = max(min(u, params.gimbal_rate_max), -params.gimbal_rate_max);
    U(:, k) = single(u);
    
    % RK4 dynamics
    x_k = double(X(:, k));
    k1 = cmg_dynamics(x_k, u, params);
    k2 = cmg_dynamics(x_k + 0.5*dt*k1, u, params);
    k3 = cmg_dynamics(x_k + 0.5*dt*k2, u, params);
    k4 = cmg_dynamics(x_k + dt*k3, u, params);
    X(:, k+1) = single(x_k + (dt/6) * (k1 + 2*k2 + 2*k3 + k4));
    
    if mod(k, 5000) == 0
        fprintf('  t = %.0f / %.0f s\n', t(k), t_end);
    end
end

U(:, end) = U(:, end-1);
ATT_DES(:, end) = ATT_DES(:, end-1);
total_time = toc;

%% Results
fprintf('\n=== Results (CUDA MEX) ===\n');
fprintf('Total time: %.2f s\n', total_time);
fprintf('MPPI time: %.2f s (%.3f ms/step, %.1f Hz)\n', ...
    mppi_time, 1000*mppi_time/N, N/mppi_time);
fprintf('Real-time factor: %.1fx\n', t_end/total_time);

%% Plot
X = double(X); U = double(U); ATT_DES = double(ATT_DES);

figure('Position', [100 100 1400 800]);

subplot(2,3,1);
plot(t, rad2deg(X(2,:)), 'b', t, rad2deg(X(3,:)), 'r');
hold on;
plot(t, rad2deg(ATT_DES(2,:)), 'b--', t, rad2deg(ATT_DES(3,:)), 'r--');
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('Pitch','Yaw','Pitch Des','Yaw Des');
title('Attitude'); grid on;

subplot(2,3,2);
plot(t, rad2deg(X(4:6,:)));
xlabel('Time [s]'); ylabel('Rate [deg/s]');
legend('p','q','r'); title('Angular Velocity'); grid on;

subplot(2,3,3);
plot(t, rad2deg(X(7,:)), t, rad2deg(X(8,:)));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\gamma_1','\gamma_2'); title('Gimbal Angles'); grid on;

subplot(2,3,4);
plot(t, rad2deg(U));
xlabel('Time [s]'); ylabel('Rate [deg/s]');
legend('\gamma_1 dot','\gamma_2 dot'); title('Gimbal Rates'); grid on;

subplot(2,3,5);
plot(t, abs(sin(X(7,:) - X(8,:))));
xlabel('Time [s]'); ylabel('|sin(\gamma_1 - \gamma_2)|');
title('Singularity Index'); grid on; yline(0.1, 'r--');

subplot(2,3,6);
att_err = rad2deg(sqrt((X(2,:)-ATT_DES(2,:)).^2 + (X(3,:)-ATT_DES(3,:)).^2));
plot(t, att_err);
xlabel('Time [s]'); ylabel('Error [deg]');
title('Attitude Error'); grid on;

sgtitle(sprintf('CMG MPPI CUDA MEX (K=%d) - %.1f Hz', mppi_params.K, N/mppi_time));
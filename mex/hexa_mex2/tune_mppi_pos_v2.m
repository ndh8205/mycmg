%% tune_mppi_pos_v2.m - MPPI v2 Position Controller Auto-Tuning
% Surrogate Optimization for position control (waypoint following)
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Check MEX
if ~exist('hexa_mppi_mex_v2', 'file')
    error('hexa_mppi_mex_v2 not found. Run: mexcuda hexa_mppi_mex_v2.cu -lcurand');
end

%% Fixed parameters
params = params_init('hexa');

% Fixed MPPI settings
K = 4096;
N = 100;
dt_ctrl = 0.01;  % 50 Hz
nu = 10.0;
t_end = 100;      % 30초 평가 (waypoint 완주용)

omega_bar2RPM = 60 / (2 * pi);

%% v2 파라미터 (9개)
% [lambda, sigma(RPM), R, w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth]

% Position 제어에서는 w_pos > 0 필요
% 하한
lb = [1,   50,  1e-7, 0.5, 0.1, 10,  0.1, 1,   0.01];
% 상한
ub = [2000, 8000, 1e-2, 5000,  5000,  5000, 5000,  3000,  1000];

n_params = length(lb);

%% 초기점 (attitude 튜닝 결과 기반 + w_pos 활성화)
% attitude 튜닝 후 여기 업데이트 권장
% x0 = [10, 200, 1e-6, 5, 1, 100, 10, 5, 0.1];
x0 = [1.1889, 51.0, 1.00e-07, 0.5129, 0.3656, 10.8916, 0.1000, 1.0000, 0.0100];

%% Objective function
obj_fun = @(x) mppi_pos_objective_v2(x, params, K, N, dt_ctrl, nu, t_end, omega_bar2RPM);

%% Surrogate Optimization
fprintf('===========================================\n');
fprintf('MPPI v2 Position Auto-Tuning\n');
fprintf('===========================================\n');
fprintf('K=%d, N=%d, t_end=%.0fs\n', K, N, t_end);
fprintf('Waypoints: 5 (10m square path)\n');
fprintf('Parameters: %d\n', n_params);
fprintf('===========================================\n\n');

opts = optimoptions('surrogateopt', ...
    'MaxFunctionEvaluations', 500, ...
    'UseParallel', true, ...
    'MinSurrogatePoints', 20, ...
    'PlotFcn', 'surrogateoptplot', ...
    'Display', 'iter', ...
    'InitialPoints', x0);

[x_best, cost_best, exitflag, output] = surrogateopt(obj_fun, lb, ub, opts);

%% Results
fprintf('\n===========================================\n');
fprintf('OPTIMIZATION COMPLETE\n');
fprintf('===========================================\n');
fprintf('Best cost: %.4f\n', cost_best);
fprintf('Exit flag: %d\n', exitflag);
fprintf('Evaluations: %d\n', output.funccount);

%% Print best parameters (복붙용)
best = build_mppi_params_v2(x_best, K, N, dt_ctrl, nu, omega_bar2RPM);

fprintf('\n%% Core MPPI parameters\n');
fprintf('mppi_params.nu = %.1f;          %% Fixed\n', nu);
fprintf('mppi_params.K = %d;\n', K);
fprintf('mppi_params.N = %d;\n', N);
fprintf('mppi_params.dt = %.4f;\n', dt_ctrl);
fprintf('mppi_params.lambda = %.4f;\n', best.lambda);
fprintf('mppi_params.sigma = %.4f;\n', best.sigma);
fprintf('mppi_params.R = %.2e;\n', best.R);
fprintf('mppi_params.w_pos = %.4f;\n', best.w_pos);
fprintf('mppi_params.w_vel = %.4f;\n', best.w_vel);
fprintf('mppi_params.w_att = %.4f;\n', best.w_att);
fprintf('mppi_params.w_omega = %.4f;\n', best.w_omega);
fprintf('mppi_params.w_terminal = %.4f;\n', best.w_terminal);
fprintf('mppi_params.w_smooth = %.4f;\n', best.w_smooth);
fprintf('mppi_params.crash_cost = 10000;\n');
fprintf('mppi_params.crash_angle = deg2rad(80);\n');

%% Initial point for next optimization (복붙용)
fprintf('\n%% 다음 최적화 초기값 (position, surrogateopt용)\n');
fprintf('%% [lambda, sigma_rpm, R, w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth]\n');
fprintf('x0 = [%.4f, %.1f, %.2e, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f];\n', ...
    x_best(1), x_best(2), x_best(3), x_best(4), x_best(5), x_best(6), x_best(7), x_best(8), x_best(9));

%% Validate best result
fprintf('\n=== Validation Run ===\n');
results = run_mppi_pos_sim_v2(best, params, t_end, true);

%% Save
save('tune_mppi_pos_v2_results.mat', 'x_best', 'cost_best', 'output', 'best', 'results');
fprintf('\nSaved to: tune_mppi_pos_v2_results.mat\n');

%% ========== OBJECTIVE FUNCTION ==========
function cost = mppi_pos_objective_v2(x, params, K, N, dt_ctrl, nu, t_end, omega_bar2RPM)
    
    mppi_params = build_mppi_params_v2(x, K, N, dt_ctrl, nu, omega_bar2RPM);
    
    try
        results = run_mppi_pos_sim_v2(mppi_params, params, t_end, false);
        cost = compute_pos_cost_v2(results);
    catch ME
        fprintf('Sim failed: %s\n', ME.message);
        cost = 9999;
    end
end

%% ========== BUILD MPPI PARAMS (v2) ==========
function mppi_params = build_mppi_params_v2(x, K, N, dt_ctrl, nu, omega_bar2RPM)
    % x = [lambda, sigma_rpm, R, w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth]
    
    mppi_params.K = K;
    mppi_params.N = N;
    mppi_params.dt = dt_ctrl;
    mppi_params.nu = nu;
    
    mppi_params.lambda = x(1);
    mppi_params.sigma = x(2) / omega_bar2RPM;  % RPM -> rad/s
    mppi_params.R = x(3);
    mppi_params.w_pos = x(4);
    mppi_params.w_vel = x(5);
    mppi_params.w_att = x(6);
    mppi_params.w_omega = x(7);
    mppi_params.w_terminal = x(8);
    mppi_params.w_smooth = x(9);
    
    % Fixed crash parameters
    mppi_params.crash_cost = 10000;
    mppi_params.crash_angle = deg2rad(80);
end

%% ========== COMPUTE COST (Position Control) ==========
function cost = compute_pos_cost_v2(results)
    
    if results.diverged
        cost = 9999;
        return;
    end
    
    % Waypoint completion is critical
    wp_penalty = (results.wp_total - results.wp_reached) * 100;
    
    if results.wp_reached < results.wp_total
        % Did not complete path - heavy penalty
        cost = 5000 + wp_penalty + results.path_complete_time;
        return;
    end
    
    % Completed path - optimize quality
    w_time = 5;       % Path completion time [s]
    w_pos = 20;       % Position tracking RMSE [m]
    w_alt = 30;       % Altitude tracking [m]
    w_att = 5;        % Attitude stability [deg]
    w_osc = 2;        % Oscillation [deg/s]
    w_final = 10;     % Final position error [m]
    
    cost = w_time * results.path_complete_time + ...
           w_pos * results.rmse_pos + ...
           w_alt * results.rmse_alt + ...
           w_att * results.rmse_att + ...
           w_osc * results.oscillation + ...
           w_final * results.final_pos_err;
    
    % ESS penalty
    K = 4096;
    if results.ess_mean < 0.05 * K
        cost = cost + 0.1 * (0.05 * K - results.ess_mean);
    end
    
    % Saturation penalty
    if results.sat_mean > 0.3
        cost = cost + 50 * (results.sat_mean - 0.3);
    end
    
    % Max attitude penalty (safety)
    if results.max_att > 45
        cost = cost + 10 * (results.max_att - 45);
    end
end
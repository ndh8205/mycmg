%% tune_mppi_att_surrogate_v2.m - MPPI v2 Surrogate Optimization
% RBF-based surrogate optimization (faster than Bayesian for high-dim)
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

K = 4096;
N = 50;
dt_ctrl = 0.01;

omega_bar2RPM = 60 / (2 * pi);

%% Simulation settings
t_end = 20;

%% Parameter bounds (9 tunable)
% Order: [lambda, sigma, R, w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth]
lb = [1,    1e-8,   1e-8, 1e-8,  1e-8,  1e-8,    1e-8,  1e-8,   1e-10];
ub = [1e+6, 1e+6,  1e+6, 1e+6,  1e+6,  1e+6, 1e+6,  1e+6,   1e+3];

% Log-transform indices (for better search in log space)
log_idx = [1, 3, 4, 5, 6, 7, 8, 9];  % lambda, R, w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth

% Transform bounds to log space
lb_t = lb;
ub_t = ub;
lb_t(log_idx) = log10(lb(log_idx));
ub_t(log_idx) = log10(ub(log_idx));

%% Initial point
% [lambda, sigma, R, w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth]
% x0 = [100, 30, 1e-6, 5.0, 1.0, 100, 10, 5, 0.1];
x0 = [1095.7288, 80.6496, 8.10e-08, 8922.3325, 684.9382, 11978.3654, 1.5109, 0.4010, 0.0011];

x0_t = x0;
x0_t(log_idx) = log10(x0(log_idx));

%% Objective function (with log transform)
obj_func = @(x_t) mppi_objective_surrogate_v2(x_t, log_idx, params, K, N, dt_ctrl, t_end);

%% Surrogate Optimization
fprintf('===========================================\n');
fprintf('MPPI v2 Surrogate Optimization\n');
fprintf('===========================================\n');
fprintf('K=%d, N=%d, t_end=%.0fs\n', K, N, t_end);
fprintf('Parameters: 9\n');
fprintf('===========================================\n\n');

options = optimoptions('surrogateopt', ...
    'MaxFunctionEvaluations', 100, ...
    'MinSurrogatePoints', 20, ...
    'UseParallel', true, ...
    'Display', 'iter', ...
    'PlotFcn', 'surrogateoptplot', ...
    'InitialPoints', x0_t);

[x_best_t, cost_best, exitflag, output] = surrogateopt(obj_func, lb_t, ub_t, options);

%% Transform back
x_best = x_best_t;
x_best(log_idx) = 10.^(x_best_t(log_idx));

%% Display results
fprintf('\n===========================================\n');
fprintf('OPTIMIZATION COMPLETE\n');
fprintf('===========================================\n');
fprintf('Best cost: %.4f\n', cost_best);
fprintf('Exit flag: %d\n', exitflag);
fprintf('\nBest parameters:\n');
fprintf('  lambda     = %.4f\n', x_best(1));
fprintf('  sigma      = %.4f rad/s (%.1f RPM)\n', x_best(2), x_best(2) * omega_bar2RPM);
fprintf('  R          = %.2e\n', x_best(3));
fprintf('  w_pos      = %.4f\n', x_best(4));
fprintf('  w_vel      = %.4f\n', x_best(5));
fprintf('  w_att      = %.4f\n', x_best(6));
fprintf('  w_omega    = %.4f\n', x_best(7));
fprintf('  w_terminal = %.4f\n', x_best(8));
fprintf('  w_smooth   = %.4f\n', x_best(9));
fprintf('===========================================\n');

%% Generate code snippet (복붙용)
fprintf('\n%% Core MPPI parameters\n');
fprintf('mppi_params.nu = %.1f;          %% Fixed\n', 10.0);
fprintf('mppi_params.K = %d;\n', K);
fprintf('mppi_params.N = %d;\n', N);
fprintf('mppi_params.dt = %.4f;\n', dt_ctrl);
fprintf('mppi_params.lambda = %.4f;\n', x_best(1));
fprintf('mppi_params.sigma = %.4f;\n', x_best(2));
fprintf('mppi_params.R = %.2e;\n', x_best(3));
fprintf('mppi_params.w_pos = %.4f;\n', x_best(4));
fprintf('mppi_params.w_vel = %.4f;\n', x_best(5));
fprintf('mppi_params.w_att = %.4f;\n', x_best(6));
fprintf('mppi_params.w_omega = %.4f;\n', x_best(7));
fprintf('mppi_params.w_terminal = %.4f;\n', x_best(8));
fprintf('mppi_params.w_smooth = %.4f;\n', x_best(9));
fprintf('mppi_params.crash_cost = 10000;\n');
fprintf('mppi_params.crash_angle = deg2rad(80);\n');

%% Initial point for next optimization (복붙용)
fprintf('\n%% 다음 최적화 초기값 (surrogateopt용)\n');
fprintf('%% [lambda, sigma, R, w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth]\n');
fprintf('x0 = [%.4f, %.4f, %.2e, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f];\n', ...
    x_best(1), x_best(2), x_best(3), x_best(4), x_best(5), x_best(6), x_best(7), x_best(8), x_best(9));

%% Save
save('tune_mppi_surrogate_v2_results.mat', 'x_best', 'cost_best', 'output', 'params', 'K', 'N', 't_end');
fprintf('\nResults saved to: tune_mppi_surrogate_v2_results.mat\n');


%% ==================== OBJECTIVE FUNCTION ====================
function cost = mppi_objective_surrogate_v2(x_t, log_idx, params, K, N, dt_ctrl, t_end)

% Transform back from log space
x = x_t;
x(log_idx) = 10.^(x_t(log_idx));

% Build mppi_params
mppi_params.K = K;
mppi_params.N = N;
mppi_params.dt = dt_ctrl;
mppi_params.lambda = x(1);
mppi_params.nu = 10.0;  % Fixed
mppi_params.sigma = x(2);
mppi_params.R = x(3);
mppi_params.w_pos = x(4);
mppi_params.w_vel = x(5);
mppi_params.w_att = x(6);
mppi_params.w_omega = x(7);
mppi_params.w_terminal = x(8);
mppi_params.w_smooth = x(9);
mppi_params.crash_cost = 10000;
mppi_params.crash_angle = deg2rad(80);

mppi_params.K_pid = 32;
mppi_params.Kp_pos = [1; 1; 2];
mppi_params.Kd_pos = [2; 2; 3];
mppi_params.Kp_att = [8; 8; 6];
mppi_params.Kd_att = [2; 2; 1.5];
mppi_params.sigma_pid = 0.2;

% Run simulation
try
    results = run_mppi_att_sim_v2(mppi_params, params, t_end, false);
catch ME
    fprintf('Simulation error: %s\n', ME.message);
    cost = 1000;
    return;
end

% Divergence - 즉시 실패
if results.diverged
    % 실패 유형별 패널티
    base_penalty = 1000;
    time_penalty = (t_end - results.diverge_time) * 100;  % 빨리 실패할수록 더 나쁨
    
    switch results.diverge_reason
        case 'ground'
            cost = base_penalty + 500 + time_penalty;  % 땅 충돌 최악
        case 'flip'
            cost = base_penalty + 400 + time_penalty;  % 뒤집힘
        case 'altitude_runaway'
            cost = base_penalty + 300 + time_penalty;  % 고도 폭주
        case 'xy_drift'
            cost = base_penalty + 200 + time_penalty;  % XY 드리프트
        otherwise
            cost = base_penalty + time_penalty;
    end
    return;
end

% Safety penalties (발산 안했지만 위험한 경우)
safety_penalty = 0;

% 큰 틸트 (60도 이상이면 위험)
if results.max_tilt > 60
    safety_penalty = safety_penalty + 100 * (results.max_tilt - 60);
elseif results.max_tilt > 45
    safety_penalty = safety_penalty + 20 * (results.max_tilt - 45);
end

% 고도 오차 (10m 기준, 3m 이상 벗어나면 위험)
alt_dev = abs(results.min_alt - 10);
if alt_dev > 5
    safety_penalty = safety_penalty + 100 * (alt_dev - 5);
elseif alt_dev > 3
    safety_penalty = safety_penalty + 30 * (alt_dev - 3);
end

% XY 드리프트 (10m 이상이면 위험)
if results.max_xy_drift > 10
    safety_penalty = safety_penalty + 50 * (results.max_xy_drift - 10);
elseif results.max_xy_drift > 5
    safety_penalty = safety_penalty + 10 * (results.max_xy_drift - 5);
end

% Metrics
settling_time = results.settling_time;
rmse_att = results.rmse_att;
max_att_ss = results.max_att_ss;
oscillation = results.oscillation;
xy_drift = results.xy_drift_ss;
alt_error = results.rmse_alt;

% ESS penalty
ess_threshold = 0.1 * K;
if results.ess_min < ess_threshold
    ess_penalty = 10 * (1 - results.ess_min / ess_threshold);
else
    ess_penalty = 0;
end

% Saturation penalty
if results.sat_max > 0.3
    sat_penalty = 20 * (results.sat_max - 0.3);
else
    sat_penalty = 0;
end

% Weighted cost
cost = 2.0 * settling_time ...
     + 5.0 * rmse_att ...
     + 1.0 * max_att_ss ...
     + 0.5 * oscillation ...
     + 5.0 * xy_drift ...
     + 10.0 * alt_error ...
     + safety_penalty ...
     + ess_penalty ...
     + sat_penalty;

cost = min(cost, 999);

end
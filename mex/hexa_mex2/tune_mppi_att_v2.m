%% tune_mppi_att_v2.m - MPPI v2 Attitude Controller Auto-Tuning
% Bayesian Optimization for improved cost function (v2)
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

%% Simulation settings for tuning
t_end = 8;  % seconds

%% Tunable parameters (9개 - v2 cost function)
% lambda:     [1, 1000]      - temperature
% sigma:      [10, 100]      - noise std (rad/s)
% R:          [1e-8, 1e-3]   - control cost
% w_pos:      [0, 100]       - position weight (disabled for attitude-only)
% w_vel:      [0.1, 100]     - velocity weight
% w_att:      [1, 1000]      - attitude weight (primary)
% w_omega:    [0.1, 100]     - angular rate weight
% w_terminal: [0.1, 100]     - terminal cost multiplier
% w_smooth:   [0.001, 10]    - control smoothness weight

vars = [
    optimizableVariable('lambda',     [1, 1000],    'Transform', 'log')
    optimizableVariable('sigma',      [10, 100])
    optimizableVariable('R',          [1e-8, 1e-3], 'Transform', 'log')
    optimizableVariable('w_vel',      [0.1, 100],   'Transform', 'log')
    optimizableVariable('w_att',      [1, 1000],    'Transform', 'log')
    optimizableVariable('w_omega',    [0.1, 100],   'Transform', 'log')
    optimizableVariable('w_terminal', [0.1, 100],   'Transform', 'log')
    optimizableVariable('w_smooth',   [0.001, 10],  'Transform', 'log')
];

%% Initial point
init_point = table(10, 30, 1e-6, 1.0, 100, 10, 5, 0.1, ...
    'VariableNames', {'lambda','sigma','R','w_vel','w_att','w_omega','w_terminal','w_smooth'});

%% Objective function
obj_func = @(x) mppi_objective_v2(x, params, K, N, dt_ctrl, t_end);

%% Bayesian Optimization
fprintf('===========================================\n');
fprintf('MPPI v2 Attitude Auto-Tuning\n');
fprintf('===========================================\n');
fprintf('K=%d, N=%d, t_end=%.0fs\n', K, N, t_end);
fprintf('Parameters: 8 (w_pos fixed at 0)\n');
fprintf('Evaluations: 100\n');
fprintf('===========================================\n\n');

results = bayesopt(obj_func, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'NumSeedPoints', 20, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'ExplorationRatio', 0.5, ...
    'IsObjectiveDeterministic', false, ...
    'UseParallel', true, ...
    'ParallelMethod', 'clipped-model-prediction', ...
    'GPActiveSetSize', 300, ...
    'Verbose', 2, ...
    'PlotFcn', {@plotObjectiveModel, @plotMinObjective, @plotElapsedTime}, ...
    'InitialX', init_point);

%% Extract best result
x_best = results.XAtMinObjective;
cost_best = results.MinObjective;

fprintf('\n===========================================\n');
fprintf('OPTIMIZATION COMPLETE\n');
fprintf('===========================================\n');
fprintf('Best cost: %.4f\n', cost_best);
fprintf('\nBest parameters:\n');
fprintf('  lambda     = %.4f\n', x_best.lambda);
fprintf('  sigma      = %.4f rad/s (%.1f RPM)\n', x_best.sigma, x_best.sigma * omega_bar2RPM);
fprintf('  R          = %.2e\n', x_best.R);
fprintf('  w_vel      = %.4f\n', x_best.w_vel);
fprintf('  w_att      = %.4f\n', x_best.w_att);
fprintf('  w_omega    = %.4f\n', x_best.w_omega);
fprintf('  w_terminal = %.4f\n', x_best.w_terminal);
fprintf('  w_smooth   = %.4f\n', x_best.w_smooth);
fprintf('===========================================\n');

%% Generate code snippet (복붙용)
fprintf('\n%% Core MPPI parameters\n');
fprintf('mppi_params.nu = %.1f;          %% Fixed\n', 10.0);
fprintf('mppi_params.K = %d;\n', K);
fprintf('mppi_params.N = %d;\n', N);
fprintf('mppi_params.dt = %.4f;\n', dt_ctrl);
fprintf('mppi_params.lambda = %.4f;\n', x_best.lambda);
fprintf('mppi_params.sigma = %.4f;\n', x_best.sigma);
fprintf('mppi_params.R = %.2e;\n', x_best.R);
fprintf('mppi_params.w_pos = 0.0;\n');
fprintf('mppi_params.w_vel = %.4f;\n', x_best.w_vel);
fprintf('mppi_params.w_att = %.4f;\n', x_best.w_att);
fprintf('mppi_params.w_omega = %.4f;\n', x_best.w_omega);
fprintf('mppi_params.w_terminal = %.4f;\n', x_best.w_terminal);
fprintf('mppi_params.w_smooth = %.4f;\n', x_best.w_smooth);
fprintf('mppi_params.crash_cost = 10000;\n');
fprintf('mppi_params.crash_angle = deg2rad(80);\n');

%% Initial point for next optimization (복붙용)
fprintf('\n%% 다음 최적화 초기값 (attitude, bayesopt용)\n');
fprintf('init_point = table(%.4f, %.4f, %.2e, %.4f, %.4f, %.4f, %.4f, %.4f, ...\n', ...
    x_best.lambda, x_best.sigma, x_best.R, x_best.w_vel, x_best.w_att, x_best.w_omega, x_best.w_terminal, x_best.w_smooth);
fprintf('    ''VariableNames'', {''lambda'',''sigma'',''R'',''w_vel'',''w_att'',''w_omega'',''w_terminal'',''w_smooth''});\n');

fprintf('\n%% 다음 최적화 초기값 (position, surrogateopt용)\n');
fprintf('%% [lambda, sigma_rpm, R, w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth]\n');
fprintf('x0 = [%.4f, %.1f, %.2e, 5.0, %.4f, %.4f, %.4f, %.4f, %.4f];\n', ...
    x_best.lambda, x_best.sigma * omega_bar2RPM, x_best.R, x_best.w_vel, x_best.w_att, x_best.w_omega, x_best.w_terminal, x_best.w_smooth);

%% Save results
save('tune_mppi_v2_results.mat', 'results', 'x_best', 'cost_best', 'params', 'K', 'N', 't_end');
fprintf('\nResults saved to: tune_mppi_v2_results.mat\n');


%% ==================== OBJECTIVE FUNCTION ====================
function cost = mppi_objective_v2(x, params, K, N, dt_ctrl, t_end)
% Objective function for MPPI v2 tuning

%% Build mppi_params struct
mppi_params.K = K;
mppi_params.N = N;
mppi_params.dt = dt_ctrl;
mppi_params.lambda = x.lambda;
mppi_params.nu = 10.0;  % Fixed importance sampling parameter
mppi_params.sigma = x.sigma;
mppi_params.R = x.R;
mppi_params.w_pos = 0.0;  % Disabled for attitude-only tuning
mppi_params.w_vel = x.w_vel;
mppi_params.w_att = x.w_att;
mppi_params.w_omega = x.w_omega;
mppi_params.w_terminal = x.w_terminal;
mppi_params.w_smooth = x.w_smooth;
mppi_params.crash_cost = 10000;
mppi_params.crash_angle = deg2rad(80);

%% Run simulation
try
    results = run_mppi_att_sim_v2(mppi_params, params, t_end, false);
catch ME
    fprintf('Simulation error: %s\n', ME.message);
    cost = 1000;
    return;
end

%% Compute cost from results

% Divergence penalty
if results.diverged
    cost = 500 + (t_end - results.diverge_time) * 50;
    return;
end

% Primary metrics
settling_time = results.settling_time;
rmse_att = results.rmse_att;
max_att_ss = results.max_att_ss;
oscillation = results.oscillation;
xy_drift = results.xy_drift_ss;
alt_error = results.rmse_alt;

% ESS penalty (want > 10% of K)
ess_threshold = 0.1 * K;
if results.ess_min < ess_threshold
    ess_penalty = 10 * (1 - results.ess_min / ess_threshold);
else
    ess_penalty = 0;
end

% Saturation penalty (want < 30%)
if results.sat_max > 0.3
    sat_penalty = 20 * (results.sat_max - 0.3);
else
    sat_penalty = 0;
end

%% Weighted cost
% Focus on attitude control quality
cost = 2.0 * settling_time ...      % Fast convergence
     + 5.0 * rmse_att ...           % Steady-state accuracy
     + 1.0 * max_att_ss ...         % Max steady-state error
     + 0.5 * oscillation ...        % Smoothness
     + 2.0 * xy_drift ...           % Position drift
     + 1.0 * alt_error ...          % Altitude error
     + ess_penalty ...              % Sampling quality
     + sat_penalty;                 % Control saturation

% Clamp
cost = min(cost, 500);

end
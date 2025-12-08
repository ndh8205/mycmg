%% tune_mppi_att.m - MPPI Attitude Controller Auto-Tuning
% Bayesian Optimization with parallel evaluation
% Focus: Steady-state tracking performance
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Check MEX
if ~exist('hexa_mppi_mex', 'file')
    error('hexa_mppi_mex not found. Run: mexcuda hexa_mppi_mex.cu -lcurand');
end

%% Fixed parameters
params = params_init('hexa');

% Fixed MPPI settings
K = 4096;
N = 100;
dt_ctrl = 0.01;  % 100 Hz
nu = 10.0;       % Fixed
t_end = 20;      % 20초 평가

omega_bar2RPM = 60 / (2 * pi);

%% Define optimization variables (12개) - NARROWED RANGE
% Based on user's working params as center

% Core MPPI
lambda_var   = optimizableVariable('lambda',   [5.00e-02, 1.00e+04]);
sigma_var    = optimizableVariable('sigma',    [1.65e+02, 5.55e+02]);   % RPM
R_var        = optimizableVariable('R',        [2.50e-06, 1.00e+01], 'Transform', 'log');
w_terminal_var = optimizableVariable('w_terminal', [8.01e+01, 1.52e+02]);

% Position weights
w_pos_xy_var = optimizableVariable('w_pos_xy', [1.30e-02, 2.37e+02]);
w_pos_z_var  = optimizableVariable('w_pos_z',  [5.48e-01, 1.63e+02]);

% Velocity weights
w_vel_xy_var = optimizableVariable('w_vel_xy', [0.05, 2.15e+03]);
w_vel_z_var  = optimizableVariable('w_vel_z',  [0.05, 5.47e+02]);

% Attitude weights
w_att_var    = optimizableVariable('w_att',    [0.2, 3.00e+03]);
w_yaw_var    = optimizableVariable('w_yaw',    [0.001, 3.15e+03]);

% Angular velocity weights
w_omega_rp_var  = optimizableVariable('w_omega_rp',  [0.005, 9.83e+02]);
w_omega_yaw_var = optimizableVariable('w_omega_yaw', [0.001, 1.63e+03]);

vars = [lambda_var, sigma_var, R_var, w_terminal_var, ...
        w_pos_xy_var, w_pos_z_var, w_vel_xy_var, w_vel_z_var, ...
        w_att_var, w_yaw_var, w_omega_rp_var, w_omega_yaw_var];

%% Initial point (user's working parameters)
init_point = table(717.64, 363.9, 2.50e-06, 116.07, 181.10, 108.29, 1319.31, 386.42, 37.35, 2496.58, 633.88, 1011.16, ...
    'VariableNames', {'lambda','sigma','R','w_terminal',...
    'w_pos_xy','w_pos_z','w_vel_xy','w_vel_z',...
    'w_att','w_yaw','w_omega_rp','w_omega_yaw'});

%% Objective function
obj_fun = @(x) mppi_att_objective(x, params, K, N, dt_ctrl, nu, t_end, omega_bar2RPM);

%% Bayesian Optimization with early stopping
fprintf('===========================================\n');
fprintf('MPPI Attitude Auto-Tuning (SIMPLE RMSE)\n');
fprintf('===========================================\n');
fprintf('K=%d, N=%d, t_end=%.0fs, nu=%.1f\n', K, N, t_end, nu);
fprintf('Parameters: 12\n');
fprintf('Evaluations: 100\n');
fprintf('Parallel: ON\n');
fprintf('Cost = RMSE(roll,pitch,yaw,alt) + settling + drift\n');
fprintf('===========================================\n\n');

opt_results = bayesopt(obj_fun, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'UseParallel', true, ...
    'InitialX', init_point, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'ExplorationRatio', 0.3, ...
    'NumSeedPoints', 30, ...
    'IsObjectiveDeterministic', false, ...
    'PlotFcn', {@plotObjectiveModel, @plotMinObjective}, ...
    'Verbose', 1);

%% Best result
fprintf('\n===========================================\n');
fprintf('OPTIMIZATION COMPLETE\n');
fprintf('===========================================\n');

all_costs = opt_results.ObjectiveTrace;
n_valid = sum(all_costs < 9999);  % diverged 아닌 것
n_total = length(all_costs);

fprintf('\nValid (not diverged): %d / %d (%.1f%%)\n', n_valid, n_total, 100*n_valid/n_total);

best = opt_results.XAtMinObjective;
best_cost = opt_results.MinObjective;

if best_cost >= 9999
    fprintf('\n!!! ALL EVALUATIONS DIVERGED !!!\n');
    fprintf('===========================================\n');
    save('tune_mppi_att_FAILED.mat', 'opt_results', 'all_costs');
    fprintf('Results saved to: tune_mppi_att_FAILED.mat\n');
    return;
end

fprintf('\nBest Parameters (cost=%.2f):\n', best_cost);
fprintf('  lambda      = %.2f\n', best.lambda);
fprintf('  sigma       = %.1f RPM (%.3f rad/s)\n', best.sigma, best.sigma / omega_bar2RPM);
fprintf('  R           = %.2e\n', best.R);
fprintf('  w_terminal  = %.2f\n', best.w_terminal);
fprintf('  w_pos_xy    = %.2f\n', best.w_pos_xy);
fprintf('  w_pos_z     = %.2f\n', best.w_pos_z);
fprintf('  w_vel_xy    = %.2f\n', best.w_vel_xy);
fprintf('  w_vel_z     = %.2f\n', best.w_vel_z);
fprintf('  w_att       = %.2f\n', best.w_att);
fprintf('  w_yaw       = %.2f\n', best.w_yaw);
fprintf('  w_omega_rp  = %.2f\n', best.w_omega_rp);
fprintf('  w_omega_yaw = %.2f\n', best.w_omega_yaw);
fprintf('\nBest Objective: %.4f\n', opt_results.MinObjective);

%% Validate best result
fprintf('\n=== Validation ===\n');
mppi_params_best = build_mppi_params(best, K, N, dt_ctrl, nu, omega_bar2RPM);
results_val = run_mppi_att_sim(mppi_params_best, params, 20, true);

%% Save results
save('tune_mppi_att_results.mat', 'opt_results', 'best', 'results_val', 'K', 'N');
fprintf('\nResults saved to: tune_mppi_att_results.mat\n');

%% Print code snippet
fprintf('\n=== Copy to main_att_mppi_mex.m ===\n');
fprintf('mppi_params.K = %d;\n', K);
fprintf('mppi_params.N = %d;\n', N);
fprintf('mppi_params.dt = dt_ctrl;\n');
fprintf('mppi_params.lambda = %.2f;\n', best.lambda);
fprintf('mppi_params.nu = %.1f;\n', nu);
fprintf('mppi_params.sigma = %.4f;  %% %.1f RPM\n', best.sigma / omega_bar2RPM, best.sigma);
fprintf('mppi_params.w_pos_xy = %.2f;\n', best.w_pos_xy);
fprintf('mppi_params.w_pos_z = %.2f;\n', best.w_pos_z);
fprintf('mppi_params.w_vel_xy = %.2f;\n', best.w_vel_xy);
fprintf('mppi_params.w_vel_z = %.2f;\n', best.w_vel_z);
fprintf('mppi_params.w_att = %.2f;\n', best.w_att);
fprintf('mppi_params.w_yaw = %.2f;\n', best.w_yaw);
fprintf('mppi_params.w_omega_rp = %.2f;\n', best.w_omega_rp);
fprintf('mppi_params.w_omega_yaw = %.2f;\n', best.w_omega_yaw);
fprintf('mppi_params.w_terminal = %.2f;\n', best.w_terminal);
fprintf('mppi_params.R = %.2e;\n', best.R);

%% ========== OBJECTIVE FUNCTION ==========
function cost = mppi_att_objective(x, params, K, N, dt_ctrl, nu, t_end, omega_bar2RPM)
    
    mppi_params = build_mppi_params(x, K, N, dt_ctrl, nu, omega_bar2RPM);
    
    try
        results = run_mppi_att_sim(mppi_params, params, t_end, false);
        cost = compute_att_cost(results, K);
    catch ME
        fprintf('Sim failed: %s\n', ME.message);
        cost = 1000;
    end
end

%% ========== BUILD MPPI PARAMS ==========
function mppi_params = build_mppi_params(x, K, N, dt_ctrl, nu, omega_bar2RPM)
    mppi_params.K = K;
    mppi_params.N = N;
    mppi_params.dt = dt_ctrl;
    mppi_params.nu = nu;
    
    % Tuned parameters
    mppi_params.lambda = x.lambda;
    mppi_params.sigma = x.sigma / omega_bar2RPM;
    mppi_params.R = x.R;
    mppi_params.w_terminal = x.w_terminal;
    mppi_params.w_pos_xy = x.w_pos_xy;
    mppi_params.w_pos_z = x.w_pos_z;
    mppi_params.w_vel_xy = x.w_vel_xy;
    mppi_params.w_vel_z = x.w_vel_z;
    mppi_params.w_att = x.w_att;
    mppi_params.w_yaw = x.w_yaw;
    mppi_params.w_omega_rp = x.w_omega_rp;
    mppi_params.w_omega_yaw = x.w_omega_yaw;
end

%% ========== COMPUTE COST (Simple RMSE-based) ==========
function cost = compute_att_cost(results, K)
    
    % Check divergence first
    if results.diverged
        cost = 9999;
        return;
    end
    
    % Simple cost: weighted sum of tracking errors
    % Lower is better
    
    % Steady-state RMSE (deg)
    cost_roll = results.rmse_roll;
    cost_pitch = results.rmse_pitch;
    cost_yaw = results.rmse_yaw;
    cost_alt = results.rmse_alt * 10;  % m -> scale up
    
    % Settling time penalty
    cost_settle = results.settling_time_avg;
    
    % Oscillation penalty
    cost_osc = results.oscillation;
    
    % XY drift penalty
    cost_drift = results.xy_drift_final;
    
    % Weights
    w_att = 10;
    w_yaw = 5;
    w_alt = 20;
    w_settle = 5;
    w_osc = 2;
    w_drift = 10;
    
    cost = w_att * (cost_roll + cost_pitch) + ...
           w_yaw * cost_yaw + ...
           w_alt * cost_alt + ...
           w_settle * cost_settle + ...
           w_osc * cost_osc + ...
           w_drift * cost_drift;
end
%% ========== (Helper functions removed - using simple RMSE cost) ==========
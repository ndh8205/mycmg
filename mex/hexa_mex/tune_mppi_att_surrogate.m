%% tune_mppi_att_surrogate.m - MPPI Attitude Controller Auto-Tuning
% Surrogate Optimization (RBF-based)
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
nu = 10.0;
t_end = 20;

omega_bar2RPM = 60 / (2 * pi);

%% Parameter bounds (12 variables)
% Order: [lambda, sigma, R, w_terminal, w_pos_xy, w_pos_z, 
%         w_vel_xy, w_vel_z, w_att, w_yaw, w_omega_rp, w_omega_yaw]

% lb = [1e-1,   300,   1e-5,  0.2,   0.5,   0.5,   0.01,  0.05,  100.0,  500.0,  0.005,   0.001];
% ub = [1e+4,   800,   10.0,  200,   300,   300,   1500,  1000,  3000,   5000,   2000,    1000 ]; 
lb = [5.00e-02, 1.65e+02, 2.50e-06, 8.01e+01, 1.30e+02, 5.48e+01, 3.47e+02, 1.87e+02, 2.50e+01, 1.53e+03, 2.66e+02, 4.33e+02];
ub = [1.00e+04, 5.55e+02, 1.00e+01, 1.52e+02, 2.37e+02, 1.63e+02, 2.15e+03, 5.47e+02, 3.00e+03, 3.15e+03, 9.83e+02, 1.63e+03];

% lb_fine = [9.85e+02, 2.52e+02, 5.00e-06, 8.11e+01, 1.28e+02, 7.61e+01, 8.73e+02, 2.57e+02, 5.08e+01, 1.64e+03, 4.37e+02, 7.23e+02];
% ub_fine = [1.83e+03, 4.69e+02, 6.50e-06, 1.51e+02, 2.39e+02, 1.41e+02, 1.62e+03, 4.77e+02, 9.43e+01, 3.04e+03, 8.12e+02, 1.34e+03];

n_vars = length(lb);
var_names = {'lambda', 'sigma', 'R', 'w_terminal', 'w_pos_xy', 'w_pos_z', ...
             'w_vel_xy', 'w_vel_z', 'w_att', 'w_yaw', 'w_omega_rp', 'w_omega_yaw'};

%% Initial point (working parameters)
x0 = [1.4064e+03, 3.6046e+02, 5.0000e-06, 1.1591e+02, 1.8352e+02, 1.0870e+02, 1.2475e+03, 3.6658e+02, 7.2521e+01, 2.3394e+03, 6.2437e+02, 1.0332e+03];

%% Objective function wrapper
obj_fun = @(x) mppi_att_objective_vec(x, params, K, N, dt_ctrl, nu, t_end, omega_bar2RPM);

%% Surrogate optimization options
opts = optimoptions('surrogateopt', ...
    'MaxFunctionEvaluations', 100, ...
    'UseParallel', true, ...
    'InitialPoints', x0, ...
    'MinSurrogatePoints', 30, ...
    'PlotFcn', 'surrogateoptplot', ...
    'Display', 'iter');

%% Run optimization
fprintf('===========================================\n');
fprintf('MPPI Attitude Auto-Tuning (surrogateopt)\n');
fprintf('===========================================\n');
fprintf('K=%d, N=%d, t_end=%.0fs, nu=%.1f\n', K, N, t_end, nu);
fprintf('Parameters: %d\n', n_vars);
fprintf('MaxEvaluations: %d\n', opts.MaxFunctionEvaluations);
fprintf('Parallel: ON\n');
fprintf('===========================================\n\n');

[x_best, cost_best, exitflag, output] = surrogateopt(obj_fun, lb, ub, opts);

%% Results
fprintf('\n===========================================\n');
fprintf('OPTIMIZATION COMPLETE\n');
fprintf('===========================================\n');
fprintf('Exit flag: %d\n', exitflag);
fprintf('Function evaluations: %d\n', output.funccount);

if cost_best >= 9999
    fprintf('\n!!! BEST RESULT DIVERGED !!!\n');
    save('tune_mppi_att_surrogate_FAILED.mat', 'x_best', 'cost_best', 'output');
    return;
end

fprintf('\nBest Parameters (cost=%.2f):\n', cost_best);
fprintf('  lambda      = %.2f\n', x_best(1));
fprintf('  sigma       = %.1f RPM (%.3f rad/s)\n', x_best(2), x_best(2) / omega_bar2RPM);
fprintf('  R           = %.2e\n', x_best(3));
fprintf('  w_terminal  = %.2f\n', x_best(4));
fprintf('  w_pos_xy    = %.2f\n', x_best(5));
fprintf('  w_pos_z     = %.2f\n', x_best(6));
fprintf('  w_vel_xy    = %.2f\n', x_best(7));
fprintf('  w_vel_z     = %.2f\n', x_best(8));
fprintf('  w_att       = %.2f\n', x_best(9));
fprintf('  w_yaw       = %.2f\n', x_best(10));
fprintf('  w_omega_rp  = %.2f\n', x_best(11));
fprintf('  w_omega_yaw = %.2f\n', x_best(12));

%% Validate best result
fprintf('\n=== Validation ===\n');
mppi_params_best = build_mppi_params_vec(x_best, K, N, dt_ctrl, nu, omega_bar2RPM);
results_val = run_mppi_att_sim(mppi_params_best, params, 20, true);

%% Save results
best.lambda = x_best(1);
best.sigma = x_best(2);
best.R = x_best(3);
best.w_terminal = x_best(4);
best.w_pos_xy = x_best(5);
best.w_pos_z = x_best(6);
best.w_vel_xy = x_best(7);
best.w_vel_z = x_best(8);
best.w_att = x_best(9);
best.w_yaw = x_best(10);
best.w_omega_rp = x_best(11);
best.w_omega_yaw = x_best(12);

save('tune_mppi_att_surrogate_results.mat', 'x_best', 'cost_best', 'best', 'results_val', 'output', 'K', 'N', 'lb', 'ub');
fprintf('\nResults saved to: tune_mppi_att_surrogate_results.mat\n');

%% Print code snippet
fprintf('\n=== Copy to main_att_mppi_mex.m ===\n');
fprintf('mppi_params.K = %d;\n', K);
fprintf('mppi_params.N = %d;\n', N);
fprintf('mppi_params.dt = dt_ctrl;\n');
fprintf('mppi_params.lambda = %.2f;\n', x_best(1));
fprintf('mppi_params.nu = %.1f;\n', nu);
fprintf('mppi_params.sigma = %.4f;  %% %.1f RPM\n', x_best(2) / omega_bar2RPM, x_best(2));
fprintf('mppi_params.w_pos_xy = %.2f;\n', x_best(5));
fprintf('mppi_params.w_pos_z = %.2f;\n', x_best(6));
fprintf('mppi_params.w_vel_xy = %.2f;\n', x_best(7));
fprintf('mppi_params.w_vel_z = %.2f;\n', x_best(8));
fprintf('mppi_params.w_att = %.2f;\n', x_best(9));
fprintf('mppi_params.w_yaw = %.2f;\n', x_best(10));
fprintf('mppi_params.w_omega_rp = %.2f;\n', x_best(11));
fprintf('mppi_params.w_omega_yaw = %.2f;\n', x_best(12));
fprintf('mppi_params.w_terminal = %.2f;\n', x_best(4));
fprintf('mppi_params.R = %.2e;\n', x_best(3));

%% ==================== NEXT SEARCH RECOMMENDATION ====================
fprintf('\n');
fprintf('============================================================\n');
fprintf('=== NEXT SEARCH RECOMMENDATION ===\n');
fprintf('============================================================\n');

% Analyze boundary proximity
boundary_margin = 0.1;  % 10% margin
lb_ratio = (x_best - lb) ./ (ub - lb);
ub_ratio = (ub - x_best) ./ (ub - lb);

hit_lb = lb_ratio < boundary_margin;
hit_ub = ub_ratio < boundary_margin;

fprintf('\n--- Boundary Analysis ---\n');
for i = 1:n_vars
    status = '';
    if hit_lb(i)
        status = ' << HIT LOWER BOUND';
    elseif hit_ub(i)
        status = ' << HIT UPPER BOUND';
    end
    fprintf('%12s: %.4e  (range: [%.2e, %.2e], pos: %.0f%%)%s\n', ...
        var_names{i}, x_best(i), lb(i), ub(i), lb_ratio(i)*100, status);
end

% Recommend new bounds
fprintf('\n--- Recommended New Bounds ---\n');
lb_new = lb;
ub_new = ub;
expand_factor = 2.0;
shrink_factor = 0.5;

for i = 1:n_vars
    range_i = ub(i) - lb(i);
    
    if hit_lb(i)
        % Expand lower bound
        lb_new(i) = max(lb(i) / expand_factor, 1e-10);
        fprintf('%12s: lb %.2e -> %.2e (EXPAND)\n', var_names{i}, lb(i), lb_new(i));
    elseif hit_ub(i)
        % Expand upper bound
        ub_new(i) = ub(i) * expand_factor;
        fprintf('%12s: ub %.2e -> %.2e (EXPAND)\n', var_names{i}, ub(i), ub_new(i));
    elseif lb_ratio(i) > 0.3 && ub_ratio(i) > 0.3
        % Both bounds far from best -> shrink around best
        margin = range_i * 0.3;
        lb_new(i) = max(x_best(i) - margin, lb(i) * shrink_factor);
        ub_new(i) = min(x_best(i) + margin, ub(i) * (1 + shrink_factor));
        fprintf('%12s: [%.2e, %.2e] -> [%.2e, %.2e] (SHRINK)\n', ...
            var_names{i}, lb(i), ub(i), lb_new(i), ub_new(i));
    end
end

% Print copy-paste ready code
fprintf('\n--- Copy for Next Run ---\n');
fprintf('%% Initial point (from this run)\n');
fprintf('x0 = [');
for i = 1:n_vars
    if i < n_vars
        fprintf('%.4e, ', x_best(i));
    else
        fprintf('%.4e];\n', x_best(i));
    end
end

fprintf('\n%% Recommended bounds\n');
fprintf('lb = [');
for i = 1:n_vars
    if i < n_vars
        fprintf('%.2e, ', lb_new(i));
    else
        fprintf('%.2e];\n', lb_new(i));
    end
end
fprintf('ub = [');
for i = 1:n_vars
    if i < n_vars
        fprintf('%.2e, ', ub_new(i));
    else
        fprintf('%.2e];\n', ub_new(i));
    end
end

fprintf('\n%% Alternative: Fine-tuning bounds (±30%% around best)\n');
fprintf('lb_fine = [');
for i = 1:n_vars
    val = max(x_best(i) * 0.7, lb(i));
    if i < n_vars
        fprintf('%.2e, ', val);
    else
        fprintf('%.2e];\n', val);
    end
end
fprintf('ub_fine = [');
for i = 1:n_vars
    val = min(x_best(i) * 1.3, ub(i));
    if i < n_vars
        fprintf('%.2e, ', val);
    else
        fprintf('%.2e];\n', val);
    end
end

fprintf('\n============================================================\n');

%% ========== OBJECTIVE FUNCTION (vector input) ==========
function cost = mppi_att_objective_vec(x, params, K, N, dt_ctrl, nu, t_end, omega_bar2RPM)
    
    mppi_params = build_mppi_params_vec(x, K, N, dt_ctrl, nu, omega_bar2RPM);
    
    try
        results = run_mppi_att_sim(mppi_params, params, t_end, false);
        cost = compute_att_cost(results);
    catch ME
        fprintf('Sim failed: %s\n', ME.message);
        cost = 10000;
    end
end

%% ========== BUILD MPPI PARAMS (vector input) ==========
function mppi_params = build_mppi_params_vec(x, K, N, dt_ctrl, nu, omega_bar2RPM)
    mppi_params.K = K;
    mppi_params.N = N;
    mppi_params.dt = dt_ctrl;
    mppi_params.nu = nu;
    
    mppi_params.lambda = x(1);
    mppi_params.sigma = x(2) / omega_bar2RPM;
    mppi_params.R = x(3);
    mppi_params.w_terminal = x(4);
    mppi_params.w_pos_xy = x(5);
    mppi_params.w_pos_z = x(6);
    mppi_params.w_vel_xy = x(7);
    mppi_params.w_vel_z = x(8);
    mppi_params.w_att = x(9);
    mppi_params.w_yaw = x(10);
    mppi_params.w_omega_rp = x(11);
    mppi_params.w_omega_yaw = x(12);
end

%% ========== COMPUTE COST ==========
function cost = compute_att_cost(results)
    
    if results.diverged
        cost = 9999;
        return;
    end
    
    w_att = 10;
    w_yaw = 5;
    w_alt = 20;
    w_settle = 5;
    w_osc = 2;
    w_drift = 10;
    
    cost = w_att * (results.rmse_roll + results.rmse_pitch) + ...
           w_yaw * results.rmse_yaw + ...
           w_alt * results.rmse_alt * 10 + ...
           w_settle * results.settling_time_avg + ...
           w_osc * results.oscillation + ...
           w_drift * results.xy_drift_final;
end
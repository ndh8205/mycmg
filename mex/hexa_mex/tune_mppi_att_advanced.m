%% tune_mppi_att_advanced.m - MPPI Advanced Auto-Tuning Framework
% Multi-fidelity Bayesian Optimization with Adaptive Refinement
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

K = 4096;
N = 100;
dt_ctrl = 0.01;
nu = 10.0;

omega_bar2RPM = 60 / (2 * pi);

%% ==================== PHASE 1: COARSE SEARCH ====================
% Short simulation, wide bounds, many samples
fprintf('==============================================\n');
fprintf('PHASE 1: COARSE SEARCH (Multi-fidelity Low)\n');
fprintf('==============================================\n');

t_end_coarse = 5;  % 짧은 시뮬레이션 (5초)
max_eval_coarse = 150;

vars_coarse = [
    optimizableVariable('lambda',      [1e-1, 1e5],   'Transform', 'log')
    optimizableVariable('sigma_rpm',   [100, 1000])
    optimizableVariable('R',           [1e-7, 100],   'Transform', 'log')
    optimizableVariable('w_terminal',  [0.01, 500],   'Transform', 'log')
    optimizableVariable('w_pos_xy',    [0.01, 500],   'Transform', 'log')
    optimizableVariable('w_pos_z',     [0.01, 500],   'Transform', 'log')
    optimizableVariable('w_vel_xy',    [0.001, 2000], 'Transform', 'log')
    optimizableVariable('w_vel_z',     [0.001, 2000], 'Transform', 'log')
    optimizableVariable('w_att',       [1, 5000],     'Transform', 'log')
    optimizableVariable('w_yaw',       [1, 5000],     'Transform', 'log')
    optimizableVariable('w_omega_rp',  [0.0001, 5000],'Transform', 'log')
    optimizableVariable('w_omega_yaw', [0.0001, 2000],'Transform', 'log')
];

obj_coarse = @(x) mppi_objective_bayesopt(x, params, K, N, dt_ctrl, nu, t_end_coarse, omega_bar2RPM);

results_coarse = bayesopt(obj_coarse, vars_coarse, ...
    'MaxObjectiveEvaluations', max_eval_coarse, ...
    'NumSeedPoints', 30, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'ExplorationRatio', 0.7, ...  % 높은 탐색 비율
    'UseParallel', true, ...
    'IsObjectiveDeterministic', false, ...
    'Verbose', 1, ...
    'PlotFcn', []);

x_coarse = results_coarse.XAtMinObjective;
fprintf('\nPhase 1 Best: cost=%.4f\n', results_coarse.MinObjective);

%% Variable Importance Analysis (Phase 1)
fprintf('\n--- Variable Importance (Phase 1) ---\n');
importance = analyze_variable_importance(results_coarse);

%% ==================== PHASE 2: REFINED SEARCH ====================
fprintf('\n==============================================\n');
fprintf('PHASE 2: REFINED SEARCH (Multi-fidelity High)\n');
fprintf('==============================================\n');

t_end_fine = 20;  % 긴 시뮬레이션 (20초)
max_eval_fine = 150;

% Adaptive bounds: ±factor around Phase 1 best (tighter for important vars)
bounds_factor_important = 3.0;  % 중요 변수: ±3x
bounds_factor_unimportant = 5.0;  % 덜 중요한 변수: ±5x (더 넓게)

var_names = {'lambda', 'sigma_rpm', 'R', 'w_terminal', 'w_pos_xy', 'w_pos_z', ...
             'w_vel_xy', 'w_vel_z', 'w_att', 'w_yaw', 'w_omega_rp', 'w_omega_yaw'};

% Extract Phase 1 best values
x_coarse_vals = table2array(x_coarse);

% Build refined bounds
vars_fine = [];
for i = 1:length(var_names)
    val = x_coarse_vals(i);
    is_log = any(strcmp(var_names{i}, {'lambda', 'R', 'w_terminal', 'w_pos_xy', ...
        'w_pos_z', 'w_vel_xy', 'w_vel_z', 'w_att', 'w_yaw', 'w_omega_rp', 'w_omega_yaw'}));
    
    % Importance-based factor
    if importance(i) > median(importance)
        factor = bounds_factor_important;
    else
        factor = bounds_factor_unimportant;
    end
    
    if is_log
        lb_i = val / factor;
        ub_i = val * factor;
        vars_fine = [vars_fine; optimizableVariable(var_names{i}, [lb_i, ub_i], 'Transform', 'log')];
    else
        range_i = val * (factor - 1);
        lb_i = max(val - range_i, 1);
        ub_i = val + range_i;
        vars_fine = [vars_fine; optimizableVariable(var_names{i}, [lb_i, ub_i])];
    end
    
    fprintf('%12s: [%.2e, %.2e] (importance: %.3f)\n', var_names{i}, lb_i, ub_i, importance(i));
end

obj_fine = @(x) mppi_objective_bayesopt(x, params, K, N, dt_ctrl, nu, t_end_fine, omega_bar2RPM);

% Warm start with Phase 1 top candidates
top_n = min(10, height(results_coarse.XTrace));
[~, sort_idx] = sort(results_coarse.ObjectiveTrace);
init_points = results_coarse.XTrace(sort_idx(1:top_n), :);

% Clamp to new bounds
for i = 1:height(init_points)
    for j = 1:length(var_names)
        val = init_points{i, var_names{j}};
        val = max(val, vars_fine(j).Range(1));
        val = min(val, vars_fine(j).Range(2));
        init_points{i, var_names{j}} = val;
    end
end

results_fine = bayesopt(obj_fine, vars_fine, ...
    'MaxObjectiveEvaluations', max_eval_fine, ...
    'NumSeedPoints', 10, ...
    'AcquisitionFunctionName', 'expected-improvement-per-second-plus', ...
    'ExplorationRatio', 0.3, ...  % 낮은 탐색, 높은 활용
    'UseParallel', true, ...
    'IsObjectiveDeterministic', false, ...
    'Verbose', 1, ...
    'PlotFcn', {@plotObjectiveModel, @plotMinObjective}, ...
    'InitialX', init_points);

%% ==================== FINAL RESULTS ====================
x_best = results_fine.XAtMinObjective;
cost_best = results_fine.MinObjective;

fprintf('\n==============================================\n');
fprintf('OPTIMIZATION COMPLETE\n');
fprintf('==============================================\n');
fprintf('Phase 1 evals: %d (t_sim=%.0fs)\n', results_coarse.NumObjectiveEvaluations, t_end_coarse);
fprintf('Phase 2 evals: %d (t_sim=%.0fs)\n', results_fine.NumObjectiveEvaluations, t_end_fine);
fprintf('Total evals: %d\n', results_coarse.NumObjectiveEvaluations + results_fine.NumObjectiveEvaluations);
fprintf('Best cost: %.4f\n', cost_best);

fprintf('\nBest Parameters:\n');
disp(x_best);

%% Final Validation (longer simulation)
fprintf('\n=== Final Validation (t=30s) ===\n');
mppi_params_best = build_mppi_params_table(x_best, K, N, dt_ctrl, nu, omega_bar2RPM);
results_val = run_mppi_att_sim(mppi_params_best, params, 30, true);

%% Variable Importance Analysis (Final)
fprintf('\n=== Variable Importance Analysis ===\n');
importance_final = analyze_variable_importance(results_fine);
[~, rank_idx] = sort(importance_final, 'descend');
fprintf('Ranking (most to least important):\n');
for i = 1:length(var_names)
    idx = rank_idx(i);
    fprintf('  %2d. %12s: %.4f\n', i, var_names{idx}, importance_final(idx));
end

%% Sensitivity Analysis
fprintf('\n=== Local Sensitivity Analysis ===\n');
sensitivity = compute_sensitivity(x_best, obj_fine, vars_fine);
fprintf('%12s | %10s | %10s\n', 'Variable', 'Sens(+10%)', 'Sens(-10%)');
fprintf('%s\n', repmat('-', 1, 38));
for i = 1:length(var_names)
    fprintf('%12s | %+10.4f | %+10.4f\n', var_names{i}, sensitivity.plus(i), sensitivity.minus(i));
end

%% Save results
save('tune_mppi_att_advanced_results.mat', ...
    'results_coarse', 'results_fine', 'x_best', 'cost_best', ...
    'importance', 'importance_final', 'sensitivity', 'results_val', 'K', 'N');
fprintf('\nResults saved to: tune_mppi_att_advanced_results.mat\n');

%% Print code snippets
print_code_snippet(x_best, K, N, nu, omega_bar2RPM);
print_next_search_recommendation(x_best, results_fine, vars_fine, importance_final, var_names);

%% ==================== HELPER FUNCTIONS ====================

function cost = mppi_objective_bayesopt(x, params, K, N, dt_ctrl, nu, t_end, omega_bar2RPM)
    mppi_params = build_mppi_params_table(x, K, N, dt_ctrl, nu, omega_bar2RPM);
    
    try
        results = run_mppi_att_sim(mppi_params, params, t_end, false);
        
        if results.diverged
            cost = 9999 + (t_end - results.diverge_time) * 100;  % 빨리 발산할수록 더 나쁨
            return;
        end
        
        % Multi-objective cost
        w_att = 10;
        w_yaw = 5;
        w_alt = 20;
        w_settle = 5;
        w_osc = 2;
        w_drift = 10;
        w_ess = 0.01;  % ESS 보너스
        
        cost = w_att * (results.rmse_roll + results.rmse_pitch) + ...
               w_yaw * results.rmse_yaw + ...
               w_alt * results.rmse_alt * 10 + ...
               w_settle * results.settling_time_avg + ...
               w_osc * results.oscillation + ...
               w_drift * results.xy_drift_final - ...
               w_ess * log(results.ess_mean + 1);  % ESS 높을수록 좋음
        
    catch ME
        fprintf('Sim failed: %s\n', ME.message);
        cost = 10000;
    end
end

function mppi_params = build_mppi_params_table(x, K, N, dt_ctrl, nu, omega_bar2RPM)
    mppi_params.K = K;
    mppi_params.N = N;
    mppi_params.dt = dt_ctrl;
    mppi_params.nu = nu;
    
    mppi_params.lambda = x.lambda;
    mppi_params.sigma = x.sigma_rpm / omega_bar2RPM;
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

function importance = analyze_variable_importance(results)
    % GP 모델 기반 변수 중요도 분석
    % ARD (Automatic Relevance Determination) length scale 기반
    
    try
        model = results.ObjectiveMinimumTrace;
        n_vars = width(results.XTrace);
        
        % Correlation-based importance (fallback)
        X = table2array(results.XTrace);
        Y = results.ObjectiveTrace;
        
        importance = zeros(1, n_vars);
        for i = 1:n_vars
            % Spearman correlation (monotonic relationship)
            importance(i) = abs(corr(X(:,i), Y, 'Type', 'Spearman'));
        end
        
        % Normalize
        importance = importance / max(importance);
        
    catch
        n_vars = width(results.XTrace);
        importance = ones(1, n_vars) / n_vars;
    end
end

function sensitivity = compute_sensitivity(x_best, obj_fun, vars)
    % Local sensitivity: ±10% perturbation
    
    var_names = {vars.Name};
    n_vars = length(var_names);
    
    % Baseline cost
    cost_base = obj_fun(x_best);
    
    sensitivity.plus = zeros(1, n_vars);
    sensitivity.minus = zeros(1, n_vars);
    
    for i = 1:n_vars
        x_plus = x_best;
        x_minus = x_best;
        
        val = x_best{1, var_names{i}};
        
        % +10%
        x_plus{1, var_names{i}} = min(val * 1.1, vars(i).Range(2));
        cost_plus = obj_fun(x_plus);
        sensitivity.plus(i) = (cost_plus - cost_base) / cost_base * 100;
        
        % -10%
        x_minus{1, var_names{i}} = max(val * 0.9, vars(i).Range(1));
        cost_minus = obj_fun(x_minus);
        sensitivity.minus(i) = (cost_minus - cost_base) / cost_base * 100;
    end
end

function print_code_snippet(x_best, K, N, nu, omega_bar2RPM)
    fprintf('\n=== Copy to main_att_mppi_mex.m ===\n');
    fprintf('mppi_params.K = %d;\n', K);
    fprintf('mppi_params.N = %d;\n', N);
    fprintf('mppi_params.dt = dt_ctrl;\n');
    fprintf('mppi_params.lambda = %.4f;\n', x_best.lambda);
    fprintf('mppi_params.nu = %.1f;\n', nu);
    fprintf('mppi_params.sigma = %.4f;  %% %.1f RPM\n', x_best.sigma_rpm / omega_bar2RPM, x_best.sigma_rpm);
    fprintf('mppi_params.w_pos_xy = %.4f;\n', x_best.w_pos_xy);
    fprintf('mppi_params.w_pos_z = %.4f;\n', x_best.w_pos_z);
    fprintf('mppi_params.w_vel_xy = %.4f;\n', x_best.w_vel_xy);
    fprintf('mppi_params.w_vel_z = %.4f;\n', x_best.w_vel_z);
    fprintf('mppi_params.w_att = %.4f;\n', x_best.w_att);
    fprintf('mppi_params.w_yaw = %.4f;\n', x_best.w_yaw);
    fprintf('mppi_params.w_omega_rp = %.4f;\n', x_best.w_omega_rp);
    fprintf('mppi_params.w_omega_yaw = %.4f;\n', x_best.w_omega_yaw);
    fprintf('mppi_params.w_terminal = %.4f;\n', x_best.w_terminal);
    fprintf('mppi_params.R = %.2e;\n', x_best.R);
end

function print_next_search_recommendation(x_best, results, vars, importance, var_names)
    fprintf('\n============================================================\n');
    fprintf('=== NEXT SEARCH RECOMMENDATION ===\n');
    fprintf('============================================================\n');
    
    % Analyze convergence
    obj_trace = results.ObjectiveTrace;
    n_eval = length(obj_trace);
    
    % Check if still improving
    if n_eval > 20
        recent_best = min(obj_trace(end-19:end));
        early_best = min(obj_trace(1:20));
        improvement = (early_best - recent_best) / early_best * 100;
        fprintf('\nConvergence: %.1f%% improvement in last 20 evals\n', improvement);
        
        if improvement < 1
            fprintf('  → Converged. Consider:\n');
            fprintf('    1. Different initial conditions\n');
            fprintf('    2. Multi-objective optimization\n');
            fprintf('    3. Robustness tuning (with disturbance)\n');
        else
            fprintf('  → Still improving. Recommend more evaluations.\n');
        end
    end
    
    % Recommend focusing on important variables
    [~, rank_idx] = sort(importance, 'descend');
    top_3 = rank_idx(1:3);
    fprintf('\nMost influential variables (focus here):\n');
    for i = 1:3
        idx = top_3(i);
        fprintf('  %d. %s (importance: %.3f)\n', i, var_names{idx}, importance(idx));
    end
    
    % Suggest fixed variables (least important)
    bottom_3 = rank_idx(end-2:end);
    fprintf('\nConsider fixing (least influential):\n');
    for i = 1:3
        idx = bottom_3(i);
        val = x_best{1, var_names{idx}};
        fprintf('  %s = %.4e (importance: %.3f)\n', var_names{idx}, val, importance(idx));
    end
    
    % Print focused search bounds
    fprintf('\n--- Focused Search (top 6 variables) ---\n');
    fprintf('vars_focused = [\n');
    for i = 1:6
        idx = rank_idx(i);
        val = x_best{1, var_names{idx}};
        lb_i = val / 2;
        ub_i = val * 2;
        is_log = ~strcmp(var_names{idx}, 'sigma_rpm');
        if is_log
            fprintf('    optimizableVariable(''%s'', [%.2e, %.2e], ''Transform'', ''log'')\n', ...
                var_names{idx}, lb_i, ub_i);
        else
            fprintf('    optimizableVariable(''%s'', [%.1f, %.1f])\n', var_names{idx}, lb_i, ub_i);
        end
    end
    fprintf('];\n');
    
    fprintf('\n============================================================\n');
end
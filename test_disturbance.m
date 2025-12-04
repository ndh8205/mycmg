%% test_disturbance.m
% Test script for disturbance system
% Usage example for dist_init, dist_torque, dist_wind, apply_uncertainty

clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% 1. Initialize parameters
params = params_init('hexa');

%% 2. Choose disturbance preset
% Options: 'nominal', 'level1', 'level2', 'level3', 'custom'
preset = 'level3';
[params, dist_state] = dist_init(params, preset);

%% 3. Apply parameter uncertainty (for level3)
if params.dist.uncertainty.enable
    params_true = apply_uncertainty(params);
else
    params_true = params;
end

%% 4. Test disturbance generation
dt = 0.01;
t_end = 30;
t_vec = 0:dt:t_end;
N = length(t_vec);

% Preallocate
tau_dist_log = zeros(3, N);
F_wind_log = zeros(3, N);

% Dummy states for testing
vel_b = [1; 0; 0];  % m/s body velocity
R_b2n = eye(3);     % Level flight

fprintf('Testing disturbance generation...\n');
for i = 1:N
    t = t_vec(i);
    
    % Torque disturbance
    [tau_dist, dist_state] = dist_torque(t, params, dist_state);
    tau_dist_log(:,i) = tau_dist;
    
    % Wind disturbance
    [F_wind, dist_state] = dist_wind(vel_b, R_b2n, t, dt, params, dist_state);
    F_wind_log(:,i) = F_wind;
end

%% 5. Plot results
figure('Name', 'Disturbance Test', 'Position', [100 100 1200 600]);

% Torque disturbance
subplot(2,2,1);
plot(t_vec, tau_dist_log(1,:), 'r', ...
     t_vec, tau_dist_log(2,:), 'g', ...
     t_vec, tau_dist_log(3,:), 'b', 'LineWidth', 1);
xlabel('Time [s]');
ylabel('Torque [Nm]');
title('Torque Disturbance');
legend('\tau_x', '\tau_y', '\tau_z');
grid on;

% Wind force
subplot(2,2,2);
plot(t_vec, F_wind_log(1,:), 'r', ...
     t_vec, F_wind_log(2,:), 'g', ...
     t_vec, F_wind_log(3,:), 'b', 'LineWidth', 1);
xlabel('Time [s]');
ylabel('Force [N]');
title('Wind Force (Body Frame)');
legend('F_x', 'F_y', 'F_z');
grid on;

% Torque magnitude
subplot(2,2,3);
tau_mag = vecnorm(tau_dist_log);
plot(t_vec, tau_mag, 'k', 'LineWidth', 1);
xlabel('Time [s]');
ylabel('|Torque| [Nm]');
title('Torque Disturbance Magnitude');
grid on;

% Wind force magnitude
subplot(2,2,4);
F_mag = vecnorm(F_wind_log);
plot(t_vec, F_mag, 'k', 'LineWidth', 1);
xlabel('Time [s]');
ylabel('|Force| [N]');
title('Wind Force Magnitude');
grid on;

sgtitle(sprintf('Disturbance Test - Preset: %s', preset));

%% 6. Print statistics
fprintf('\n=== Disturbance Statistics ===\n');
fprintf('Torque Disturbance:\n');
fprintf('  Max |tau|:  %.4f Nm\n', max(tau_mag));
fprintf('  Mean |tau|: %.4f Nm\n', mean(tau_mag));
fprintf('  Std |tau|:  %.4f Nm\n', std(tau_mag));
fprintf('\nWind Force:\n');
fprintf('  Max |F|:    %.4f N\n', max(F_mag));
fprintf('  Mean |F|:   %.4f N\n', mean(F_mag));
fprintf('  Std |F|:    %.4f N\n', std(F_mag));
fprintf('==============================\n');

%% 7. Test different presets comparison
figure('Name', 'Preset Comparison', 'Position', [100 100 1000 400]);

presets = {'level1', 'level2', 'level3'};
colors = {'b', 'r', 'k'};

for p = 1:length(presets)
    % Reset
    params_test = params_init('hexa');
    [params_test, dist_state_test] = dist_init(params_test, presets{p});
    
    tau_test = zeros(1, N);
    for i = 1:N
        [tau, dist_state_test] = dist_torque(t_vec(i), params_test, dist_state_test);
        tau_test(i) = norm(tau);
    end
    
    subplot(1,2,1); hold on;
    plot(t_vec, tau_test, colors{p}, 'LineWidth', 1);
end
xlabel('Time [s]');
ylabel('|Torque| [Nm]');
title('Torque Disturbance Comparison');
legend(presets);
grid on;

% Wind comparison
for p = 1:length(presets)
    params_test = params_init('hexa');
    [params_test, dist_state_test] = dist_init(params_test, presets{p});
    
    F_test = zeros(1, N);
    for i = 1:N
        [F, dist_state_test] = dist_wind(vel_b, R_b2n, t_vec(i), dt, params_test, dist_state_test);
        F_test(i) = norm(F);
    end
    
    subplot(1,2,2); hold on;
    plot(t_vec, F_test, colors{p}, 'LineWidth', 1);
end
xlabel('Time [s]');
ylabel('|Force| [N]');
title('Wind Force Comparison');
legend(presets);
grid on;

fprintf('\nTest complete.\n');
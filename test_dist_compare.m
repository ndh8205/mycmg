%% test_dist_compare.m
% Compare 'paper' vs 'random_sine' torque disturbance
clear; clc; close all;

addpath(genpath('model'));
addpath(genpath('disturbance'));

%% Simulation settings
dt = 0.01;
t_end = 40;
t = 0:dt:t_end;
N = length(t);

%% Initialize both types
params1 = params_init('hexa');
[params1, dist_state1] = dist_init(params1, 'paper');

params2 = params_init('hexa');
[params2, dist_state2] = dist_init(params2, 'level1');  % random_sine

%% Generate disturbances
tau_paper = zeros(3, N);
tau_random = zeros(3, N);

for k = 1:N
    [tau_paper(:,k), dist_state1] = dist_torque(t(k), params1, dist_state1);
    [tau_random(:,k), dist_state2] = dist_torque(t(k), params2, dist_state2);
end

%% Plot comparison
figure('Position', [100 100 1200 800]);

% Paper type - 3 axes
subplot(3,2,1);
plot(t, tau_paper(1,:), 'r', t, tau_paper(2,:), 'g', t, tau_paper(3,:), 'b', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Torque [Nm]');
title('Paper Type (Integrated Random Sine)');
legend('\tau_x', '\tau_y', '\tau_z'); grid on;
ylim([-0.03 0.03]);

% Random sine type - 3 axes
subplot(3,2,2);
plot(t, tau_random(1,:), 'r', t, tau_random(2,:), 'g', t, tau_random(3,:), 'b', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Torque [Nm]');
title('Random Sine Type');
legend('\tau_x', '\tau_y', '\tau_z'); grid on;
ylim([-0.03 0.03]);

% Paper - magnitude
subplot(3,2,3);
plot(t, vecnorm(tau_paper), 'k', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('|Torque| [Nm]');
title('Paper - Magnitude'); grid on;

% Random sine - magnitude
subplot(3,2,4);
plot(t, vecnorm(tau_random), 'k', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('|Torque| [Nm]');
title('Random Sine - Magnitude'); grid on;

% FFT comparison - X axis
subplot(3,2,5);
Fs = 1/dt;
L = N;
f = Fs*(0:floor(L/2))/L;

Y1 = fft(tau_paper(1,:));
P1 = abs(Y1/L);
P1 = P1(1:floor(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);

Y2 = fft(tau_random(1,:));
P2 = abs(Y2/L);
P2 = P2(1:floor(L/2)+1);
P2(2:end-1) = 2*P2(2:end-1);

plot(f, P1, 'b', f, P2, 'r', 'LineWidth', 1);
xlabel('Frequency [Hz]'); ylabel('Amplitude');
title('FFT Comparison (\tau_x)');
legend('Paper', 'Random Sine'); grid on;
xlim([0 5]);

% Overlay comparison
subplot(3,2,6);
plot(t, tau_paper(1,:), 'b', t, tau_random(1,:), 'r', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('\tau_x [Nm]');
title('Overlay Comparison (\tau_x)');
legend('Paper', 'Random Sine'); grid on;

sgtitle('Torque Disturbance Comparison: Paper vs Random Sine');

%% Statistics
fprintf('\n=== Disturbance Statistics ===\n');
fprintf('\nPaper Type:\n');
fprintf('  Max:  [%.4f, %.4f, %.4f] Nm\n', max(tau_paper,[],2));
fprintf('  Min:  [%.4f, %.4f, %.4f] Nm\n', min(tau_paper,[],2));
fprintf('  Std:  [%.4f, %.4f, %.4f] Nm\n', std(tau_paper,[],2));
fprintf('  RMS:  [%.4f, %.4f, %.4f] Nm\n', rms(tau_paper,2));

fprintf('\nRandom Sine Type:\n');
fprintf('  Max:  [%.4f, %.4f, %.4f] Nm\n', max(tau_random,[],2));
fprintf('  Min:  [%.4f, %.4f, %.4f] Nm\n', min(tau_random,[],2));
fprintf('  Std:  [%.4f, %.4f, %.4f] Nm\n', std(tau_random,[],2));
fprintf('  RMS:  [%.4f, %.4f, %.4f] Nm\n', rms(tau_random,2));
fprintf('==============================\n');
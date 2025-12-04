function [params, dist_state] = dist_init(params, preset)
% dist_init: Initialize disturbance parameters and states
%
% Inputs:
%   params - system parameter struct
%   preset - 'nominal'|'level1'|'level2'|'level3'|'custom' (default: 'nominal')
%
% Outputs:
%   params     - updated params with dist field
%   dist_state - disturbance state struct (for Dryden filter, etc.)

if nargin < 2
    preset = 'nominal';
end

%% Default disturbance parameters
params.dist.enable = false;

% Torque disturbance
params.dist.torque.type = 'none';           % 'none'|'random_sine'|'step'|'impulse'|'sine'|'combined'|'paper'
params.dist.torque.magnitude = [0.02; 0.02; 0.02];  % [Nm]
params.dist.torque.freq = 0.5;              % [Hz] for sine
params.dist.torque.step_time = 5;           % [s]
params.dist.torque.step_duration = inf;     % [s]
params.dist.torque.impulse_time = 3;        % [s]
params.dist.torque.impulse_duration = 0.1;  % [s]

% Paper-style torque disturbance (integrated random sine)
params.dist.torque.paper.accel_size = [0.02; 0.02; 0.02];  % Base amplitude
params.dist.torque.paper.freq_mult = 2;                     % pi_size in paper
params.dist.torque.paper.scale = 5;                         % Divider
params.dist.torque.paper.max_torque = 0.02;                 % Max bound [Nm]

% Wind disturbance
params.dist.wind.type = 'none';             % 'none'|'constant'|'gust'|'dryden'
params.dist.wind.velocity_ned = [2; 0; 0];  % [m/s] constant wind in NED
params.dist.wind.gust_magnitude = 5;        % [m/s]
params.dist.wind.gust_start = 10;           % [s]
params.dist.wind.gust_duration = 2;         % [s]
params.dist.wind.dryden_intensity = 'light';% 'light'|'moderate'|'severe'
params.dist.wind.altitude = 50;             % [m] for Dryden model

% Aerodynamic parameters for wind effect
params.dist.aero.Cd = 1.0;                  % Drag coefficient
params.dist.aero.A = 0.1;                   % Reference area [m^2]
params.dist.aero.rho = 1.225;               % Air density [kg/m^3]

% Parameter uncertainty
params.dist.uncertainty.enable = false;
params.dist.uncertainty.mass = 0;           % ±fraction (0.1 = ±10%)
params.dist.uncertainty.inertia = 0;        % ±fraction
params.dist.uncertainty.k_T = 0;            % ±fraction

%% Apply presets
switch preset
    case 'nominal'
        params.dist.enable = false;
        
    case 'level1'
        params.dist.enable = true;
        params.dist.torque.type = 'random_sine';
        params.dist.torque.magnitude = [0.02; 0.02; 0.02];
        params.dist.wind.type = 'constant';
        params.dist.wind.velocity_ned = [2; 1; 0];
        
    case 'level2'
        params.dist.enable = true;
        params.dist.torque.type = 'combined';
        params.dist.torque.magnitude = [0.03; 0.03; 0.03];
        params.dist.wind.type = 'gust';
        params.dist.wind.gust_magnitude = 5;
        params.dist.wind.gust_start = 10;
        params.dist.wind.gust_duration = 2;
        
    case 'level3'
        params.dist.enable = true;
        params.dist.torque.type = 'combined';
        params.dist.torque.magnitude = [0.03; 0.03; 0.03];
        params.dist.wind.type = 'dryden';
        params.dist.wind.dryden_intensity = 'moderate';
        params.dist.uncertainty.enable = true;
        params.dist.uncertainty.mass = 0.1;
        params.dist.uncertainty.inertia = 0.1;
        params.dist.uncertainty.k_T = 0.05;
        
    case 'custom'
        % Use current params.dist values
        params.dist.enable = true;
        
    case 'paper'
        % Paper-style disturbance (integrated random sine, torque only)
        params.dist.enable = true;
        params.dist.torque.type = 'paper';
        params.dist.torque.paper.accel_size = [0.02; 0.02; 0.02];
        params.dist.torque.paper.freq_mult = 2;
        params.dist.torque.paper.scale = 5;
        params.dist.torque.paper.max_torque = 0.02;
        params.dist.wind.type = 'none';

    case 'level_hell'
        params.dist.enable = true;
        
        params.dist.wind.type = 'dryden';
        params.dist.wind.velocity_ned = [10; 5; 2]; 
        
        params.dist.wind.dryden_intensity = 'severe'; 
        
        params.dist.torque.type = 'combined';
        params.dist.torque.magnitude = [0.05; 0.05; 0.05]; % 기존보다 2.5배 강화
        
        params.dist.uncertainty.enable = true;
        params.dist.uncertainty.mass = 0.2;    % 실제 무게가 알고 있는 것보다 20% 무거움/가벼움
        params.dist.uncertainty.inertia = 0.2; % 관성 모멘트 20% 오차
        params.dist.uncertainty.k_T = 0.1;     % 모터 추력 10% 감소/증가
end

%% Initialize disturbance state
dist_state.t_prev = 0;

% Random sine state (pre-generated phases and frequencies)
rng('shuffle');
dist_state.random_sine.phase = 2*pi*rand(3,5);      % 5 harmonics per axis
dist_state.random_sine.freq_mult = 0.5 + rand(3,5); % Frequency multipliers

% Paper-style disturbance state (integrated random sine)
% Multiple sinusoids per axis for richer spectrum (like paper's time-varying random)
n_harmonics = 10;
dist_state.paper.integral = zeros(3,1);
dist_state.paper.freq = 0.1 + 2*rand(3, n_harmonics);      % Random frequencies [0.1-2.1] Hz
dist_state.paper.phase = 2*pi*rand(3, n_harmonics);        % Random phases
dist_state.paper.amp = rand(3, n_harmonics);               % Random amplitudes
% Normalize amplitudes so sum = 1
dist_state.paper.amp = dist_state.paper.amp ./ sum(dist_state.paper.amp, 2);

% Dryden filter state (6 states: 2 per axis for 2nd order filter)
dist_state.dryden.x = zeros(6,1);
[dist_state.dryden.A, dist_state.dryden.B, dist_state.dryden.C] = ...
    dryden_matrices(params.dist.wind.altitude, params.dist.wind.dryden_intensity);

% Gust state
dist_state.gust.active = false;
dist_state.gust.direction = [1; 0; 0];  % Will be randomized at gust start

end

%% Local function: Dryden state-space matrices
function [A, B, C] = dryden_matrices(h, intensity)
% Simplified Dryden model for low altitude
% Returns state-space matrices for 3-axis turbulence

% Turbulence intensity based on preset
switch intensity
    case 'light'
        sigma_u = 1.5;  sigma_v = 1.5;  sigma_w = 1.0;
    case 'moderate'
        sigma_u = 3.0;  sigma_v = 3.0;  sigma_w = 2.0;
    case 'severe'
        sigma_u = 6.0;  sigma_v = 6.0;  sigma_w = 4.0;
    otherwise
        sigma_u = 1.5;  sigma_v = 1.5;  sigma_w = 1.0;
end

% Scale lengths (low altitude approximation)
L_u = h / (0.177 + 0.000823*h)^1.2;
L_v = L_u;
L_w = h;

% Airspeed (nominal)
V = 10;  % [m/s]

% Time constants
tau_u = L_u / V;
tau_v = L_v / V;
tau_w = L_w / V;

% First-order approximation for each axis
% dx/dt = -1/tau * x + sqrt(2/tau) * sigma * noise
A = diag([-1/tau_u, -1/tau_u, -1/tau_v, -1/tau_v, -1/tau_w, -1/tau_w]);
B = [sqrt(2/tau_u)*sigma_u; 0; sqrt(2/tau_v)*sigma_v; 0; sqrt(2/tau_w)*sigma_w; 0];
C = [1 0 0 0 0 0;
     0 0 1 0 0 0;
     0 0 0 0 1 0];

end
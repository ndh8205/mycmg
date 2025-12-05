function [thrust_cmd, ctrl_state] = altitude_smc(alt_des, alt, vel_z, q, ctrl_state, gains, dt, params)
% altitude_smc: Fractional-order Sliding Mode Controller for altitude
%
% Based on paper's Alti_controller.m
%
% Inputs:
%   alt_des    - desired altitude [m] (positive up)
%   alt        - current altitude [m] (positive up)
%   vel_z      - vertical velocity [m/s] (positive up)
%   q          - 4x1 current quaternion [qw; qx; qy; qz]
%   ctrl_state - struct (kept for interface compatibility)
%   gains      - struct with SMC gains:
%                .a       - sliding surface gain (default: 1)
%                .lambda1 - reaching law gain 1 (default: 1)
%                .lambda2 - reaching law gain 2 (default: 0.8)
%                .r       - fractional power (default: 0.98)
%   dt         - time step [s]
%   params     - system parameters
%
% Outputs:
%   thrust_cmd - total thrust command [N]
%   ctrl_state - unchanged
%
% SMC Structure:
%   Sliding surface: s = -ż + a*|e|^r * sign(e)
%   Reaching law:    ṡ = -λ₁*s - λ₂*|s|^r * sign(s)

%% Extract gains (with defaults)
if isfield(gains, 'a')
    a = gains.a;
else
    a = 1;
end

if isfield(gains, 'lambda1')
    lambda1 = gains.lambda1;
else
    lambda1 = 1;
end

if isfield(gains, 'lambda2')
    lambda2 = gains.lambda2;
else
    lambda2 = 0.8;
end

if isfield(gains, 'r')
    r = gains.r;
else
    r = 0.98;
end

%% Parameters
m = params.drone.body.m;
g = params.env.g;

%% Attitude compensation
% Get euler angles for thrust direction compensation
euler = Quat2Euler(q);
phi = euler(1);    % roll
theta = euler(2);  % pitch

% Avoid division by zero
cos_phi_theta = cos(phi) * cos(theta);
if abs(cos_phi_theta) < 0.1
    cos_phi_theta = sign(cos_phi_theta) * 0.1;
end

%% Altitude error
e = alt_des - alt;

%% Sliding surface (fractional-order)
% s = -ż + a*|e|^r * sign(e)
% Handle singularity when e ≈ 0
eps_val = 1e-6;
e_safe = max(abs(e), eps_val);
s = -vel_z + a * e_safe^r * sign(e);

%% Reaching law (fractional-order)
% ṡ = -λ₁*s - λ₂*|s|^r * sign(s)
s_safe = max(abs(s), eps_val);
s_dot = -lambda1 * s - lambda2 * s_safe^r * sign(s);

%% Control input
% T = (a*(-ż) + g - ṡ) * m / (cosφ*cosθ)
%
% Note: Paper uses a*(-ż) which provides velocity damping
% Combined with the fractional sliding surface
thrust_cmd = (a * (-vel_z) + g - s_dot) * m / cos_phi_theta;

%% Thrust saturation
n_motor = params.drone.body.n_motor;
thrust_max = n_motor * params.drone.motor.k_T * params.drone.motor.omega_b_max^2;
thrust_min = 0;
thrust_cmd = max(min(thrust_cmd, thrust_max), thrust_min);

end
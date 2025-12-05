function [tau_cmd, ctrl_state] = attitude_smc(q_des, q, omega, ctrl_state, gains, params)
% attitude_smc: Fractional-order Sliding Mode Controller for attitude
%
% Based on paper's Atti_controller.m
% Adapted for scalar-first quaternion convention [qw; qx; qy; qz]
%
% Inputs:
%   q_des      - 4x1 desired quaternion [qw; qx; qy; qz] (scalar-first)
%   q          - 4x1 current quaternion
%   omega      - 3x1 angular velocity [p; q; r] [rad/s] (body frame)
%   ctrl_state - struct (kept for interface compatibility)
%   gains      - struct with SMC gains:
%                .a       - sliding surface gain (default: 5)
%                .b       - fractional term gain (default: 5)
%                .lambda1 - reaching law gain 1 (default: 0.4)
%                .lambda2 - reaching law gain 2 (default: 0.4)
%                .r       - fractional power (default: 0.95)
%   params     - system parameters (for inertia J)
%
% Outputs:
%   tau_cmd    - 3x1 torque command [Nm]
%   ctrl_state - unchanged
%
% SMC Structure:
%   Sliding surface: s = ω + a*e_v + b*|e_v|^r * sign(e_v)
%   Reaching law:    ṡ = -λ₁*s - λ₂*|s|^r * sign(s)
%   where e_v is quaternion error vector part

%% Ensure column vectors
q_des = q_des(:);
q = q(:);
omega = omega(:);

%% Extract gains (with defaults)
if isfield(gains, 'a')
    a = gains.a;
else
    a = 5;
end

if isfield(gains, 'b')
    b_gain = gains.b;
else
    b_gain = 5;
end

if isfield(gains, 'lambda1')
    lambda1 = gains.lambda1;
else
    lambda1 = 0.4;
end

if isfield(gains, 'lambda2')
    lambda2 = gains.lambda2;
else
    lambda2 = 0.4;
end

if isfield(gains, 'r')
    r = gains.r;
else
    r = 0.95;
end

%% Get inertia
J = params.drone.body.J;

%% Quaternion error
% Paper uses vector-first [qv; q0], we use scalar-first [q0; qv]
% q_err = conj(q_des) ⊗ q  (rotation from desired to current)
%
% Paper's Q_e = [Q_d(4)*I - skew(Q_d(1:3)), -Q_d(1:3); Q_d(1:3)', Q_d(4)] * Q
% is equivalent to q_err = conj(q_des) ⊗ q in our convention

q_des_conj = [q_des(1); -q_des(2:4)];
q_err = q2q_mult(q_des_conj, q);

% Ensure shortest path (positive scalar part)
if q_err(1) < 0
    q_err = -q_err;
end

%% Extract quaternion error components
% scalar-first: q_err = [q_err_0; q_err_v]
% Paper's Q_e(4) corresponds to our q_err(1)
% Paper's Q_e(1:3) corresponds to our q_err(2:4)
q_err_0 = q_err(1);      % scalar part
q_err_v = q_err(2:4);    % vector part

%% Sliding surface (fractional-order)
% s = ω + a*q_err_v + b*|q_err_v|^r * sign(q_err_v)
s = omega + a * q_err_v + b_gain * abs(q_err_v).^r .* sign(q_err_v);

%% Reaching law (fractional-order)
% ṡ = -λ₁*s - λ₂*|s|^r * sign(s)
s_dot = -lambda1 * s - lambda2 * abs(s).^r .* sign(s);

%% Quaternion derivative contribution
% From paper: coeff .* (skew(q_err_v) + q_err_0*I) * ω / 2
% where coeff = a + b*r*|q_err_v|^(r-1)
%
% Handle singularity when q_err_v ≈ 0
eps_val = 1e-6;
q_err_v_safe = max(abs(q_err_v), eps_val);
coeff = a + b_gain * r * q_err_v_safe.^(r-1);

% skew(q_err_v) + q_err_0*I
M = skew3(q_err_v) + q_err_0 * eye(3);

% Quaternion error derivative term
q_dot_term = coeff .* (M * omega / 2);

%% Control input
% τ = ω×(Jω) - J*sign(q_err_0)*q_dot_term + J*ṡ
tau_cmd = cross(omega, J*omega) - J * sign(q_err_0) * q_dot_term + J * s_dot;

end
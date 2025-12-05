function [q_des, thrust_cmd, ctrl_state] = position_smc(pos_des, pos, vel, yaw_des, ctrl_state, gains, params)
% position_smc: Fractional-order SMC Position Controller
%
% Inputs:
%   pos_des    - 3x1 desired position [m] (NED frame)
%   pos        - 3x1 current position [m] (NED frame)
%   vel        - 3x1 current velocity [m/s] (NED frame)
%   yaw_des    - desired yaw angle [rad]
%   ctrl_state - struct (kept for interface compatibility)
%   gains      - struct with SMC gains:
%                .a       - sliding surface gain (default: 2)
%                .lambda1 - reaching law gain 1 (default: 1)
%                .lambda2 - reaching law gain 2 (default: 0.5)
%                .r       - fractional power (default: 0.9)
%                .a_max   - acceleration saturation [m/s²] (default: [4;4;6])
%   params     - system parameters
%
% Outputs:
%   q_des      - 4x1 desired quaternion
%   thrust_cmd - total thrust command [N]
%   ctrl_state - unchanged
%
% SMC Structure (per axis):
%   Sliding surface: s = -vel + a*e_pos
%   Control law:     a_des = -a*vel + λ₁*s + λ₂*|s|^r*sign(s)

%% Ensure column vectors
pos_des = pos_des(:);
pos = pos(:);
vel = vel(:);

%% Extract gains (with defaults)
% 축별 게인 지원: 스칼라 또는 3x1 벡터
if isfield(gains, 'a')
    a = gains.a(:);
    if length(a) == 1, a = a * ones(3,1); end
else
    a = [2; 2; 4];
end

if isfield(gains, 'lambda1')
    lambda1 = gains.lambda1(:);
    if length(lambda1) == 1, lambda1 = lambda1 * ones(3,1); end
else
    lambda1 = [1; 1; 2];
end

if isfield(gains, 'lambda2')
    lambda2 = gains.lambda2(:);
    if length(lambda2) == 1, lambda2 = lambda2 * ones(3,1); end
else
    lambda2 = [0.3; 0.3; 0.5];
end

if isfield(gains, 'r')
    r = gains.r;
else
    r = 0.9;
end

%% Parameters
m = params.drone.body.m;
g = params.env.g;

%% Position error (NED)
e_pos = pos_des - pos;

%% Sliding surface (축별)
% s = -vel + a.*e_pos
s = -vel + a .* e_pos;

%% Control law (축별)
% a_des = -a.*vel + λ₁.*s + λ₂.*|s|^r.*sign(s)
eps_val = 1e-6;
s_safe = max(abs(s), eps_val);
a_des = -a .* vel + lambda1 .* s + lambda2 .* s_safe.^r .* sign(s);

%% Acceleration saturation (발산 방지)
% 물리적으로 실현 가능한 가속도로 제한
% 틸트 각도 제한 (~20deg) 고려: a_xy_max ≈ g*tan(20°) ≈ 3.5 m/s²
if isfield(gains, 'a_max')
    a_max = gains.a_max(:);
    if length(a_max) == 1, a_max = a_max * ones(3,1); end
else
    a_max = [4; 4; 6];  % [m/s²] XY는 틸트 제한, Z는 추력 여유 고려
end
a_des = max(min(a_des, a_max), -a_max);

%% Acceleration to thrust and attitude
% F_des = m * (a_des - g_ned)  where g_ned = [0; 0; g]
g_ned = [0; 0; g];
F_des = m * (a_des - g_ned);

% 멀티로터는 추력 반전 불가 → F_des_z는 음수여야 함 (위로 힘)
% F_des_z > 0이면 드론이 뒤집히려 함 → 추력 최소로 제한
if F_des(3) > -0.1 * m * g  % 최소 hover 추력의 10%
    F_des(3) = -0.1 * m * g;
end

% Total thrust magnitude
thrust_cmd = norm(F_des);

% Desired body z-axis in NED frame
if thrust_cmd > 1e-6
    z_b_des = -F_des / thrust_cmd;
else
    z_b_des = [0; 0; -1];  % Default: level
end

%% Construct desired rotation matrix
% Given z_b_des and yaw_des, construct R_b2n_des

% Desired x-axis direction (from yaw)
x_c = [cos(yaw_des); sin(yaw_des); 0];

% Desired y-axis: z × x (then normalize)
y_b_des = cross(z_b_des, x_c);
y_norm = norm(y_b_des);
if y_norm > 1e-6
    y_b_des = y_b_des / y_norm;
else
    % Singularity: z_b parallel to x_c
    y_b_des = [0; 1; 0];
end

% Recompute x-axis to ensure orthogonality
x_b_des = cross(y_b_des, z_b_des);
x_b_des = x_b_des / norm(x_b_des);

% Rotation matrix (body to NED)
R_b2n_des = [x_b_des, y_b_des, z_b_des];

%% Convert to quaternion
q_des = DCM2Quat(R_b2n_des);
q_des = q_des(:);

% Ensure positive scalar part
if q_des(1) < 0
    q_des = -q_des;
end

end
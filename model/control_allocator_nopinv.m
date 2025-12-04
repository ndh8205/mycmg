function [output, B] = control_allocator_nopinv(input, params, mode)
% control_allocator: Quadrotor/Hexarotor control allocation
%
% Usage:
%   [omega_sq, B] = control_allocator(cmd_vec, params, 'inverse')
%   [cmd_vec, B]  = control_allocator(omega_sq, params, 'forward')
%
% Inputs:
%   input  - Nx1 vector (N=4 for quad, N=6 for hexa)
%            'inverse': [T; τx; τy; τz] → omega_sq
%            'forward': [ω²] → [T; τx; τy; τz]
%   params - parameter struct
%   mode   - 'inverse' (default) or 'forward'
%
% Outputs:
%   output - Nx1 vector
%   B      - 4xN allocation matrix

if nargin < 3
    mode = 'inverse';
end

%% Determine drone type
if isfield(params.drone, 'type') && strcmp(params.drone.type, 'hexa')
    [output, B] = hexa_allocator(input, params, mode);
else
    [output, B] = quad_allocator(input, params, mode);
end

end

%% ========== QUADROTOR ==========
function [output, B] = quad_allocator(input, params, mode)

%% Parameters
k_T = params.drone.motor.k_T;   % Thrust coeff [N/(rad/s)^2]
k_M = params.drone.motor.k_M;   % Moment coeff [m]: τ = k_M * T

%% Lever Arms (absolute values)
l_x  = 0.13;   % [m] longitudinal arm
l_y1 = 0.22;   % [m] lateral arm (rotor 0, 2)
l_y2 = 0.20;   % [m] lateral arm (rotor 1, 3)

%% Allocation Matrix B
%  [T; τx; τy; τz] = B * [ω1²; ω2²; ω3²; ω4²]
%
%  Coordinate: X-Forward, Y-Right, Z-Down (NED)
%
%  Rotor layout (top view):
%
%           x (Front)
%           ^
%           |
%     1(CCW)|  3(CW)
%        \  |  /
%         \ | /
%          -+----> y (Right)
%         / | \
%        /  |  \
%     4(CW) |  2(CCW)
%           |
%        (Back)

%            ω1²        ω2²        ω3²        ω4²
B = [
% T:    all thrusts sum up
        k_T,        k_T,        k_T,        k_T;

% τx:   Left(+)    Right(-)   Right(-)   Left(+)
        k_T*l_y1,  -k_T*l_y2,  -k_T*l_y1,  k_T*l_y2;

% τy:   Front(+)   Back(-)    Front(+)   Back(-)
        k_T*l_x,   -k_T*l_x,   k_T*l_x,  -k_T*l_x;

% τz:   CCW(+)     CCW(+)     CW(-)      CW(-)
        k_T*k_M,    k_T*k_M,  -k_T*k_M,  -k_T*k_M
];

%% Compute output
if strcmp(mode, 'inverse')
    output = B \ input;
else
    output = B * input;
end

end

%% ========== HEXAROTOR ==========
function [output, B] = hexa_allocator(input, params, mode)

%% Parameters
k = params.drone.motor.k_T;     % Thrust coeff [N/(rad/s)^2]
b = k * params.drone.motor.k_M; % Drag coeff [Nm/(rad/s)^2]
L = params.drone.body.L;        % Arm length [m]

%% Allocation Matrix B
%  [T; τx; τy; τz] = B * [ω1²; ω2²; ω3²; ω4²; ω5²; ω6²]
%
%  Coordinate: X-Forward, Y-Right, Z-Down (NED)
%
%  Rotor layout (top view, looking down Z+):
%
%              X+ (Front)
%               ↑
%          M2      M1
%         (CW)   (CCW)
%            \    /
%     M3 -----+-----→ Y+ (Right)
%    (CCW)    |    M6 (CCW)
%            /    \
%         M4      M5
%        (CW)    (CW)
%
%  Motor positions (angle from X+, CCW positive):
%    M1: 30°,  (x,y) = (+L*√3/2, +L/2),   CCW
%    M2: 330°, (x,y) = (+L*√3/2, -L/2),   CW
%    M3: 270°, (x,y) = (0,       -L),     CCW
%    M4: 210°, (x,y) = (-L*√3/2, -L/2),   CW
%    M5: 150°, (x,y) = (-L*√3/2, +L/2),   CW
%    M6: 90°,  (x,y) = (0,       +L),     CCW
%
%  Torque equations (NED):
%    τx (roll)  = Σ(-y_i × k × ω_i²)
%    τy (pitch) = Σ(+x_i × k × ω_i²)
%    τz (yaw)   = Σ(±b × ω_i²)  CCW:+, CW:-

s3 = sqrt(3)/2;  % ≈ 0.866

%          M1        M2        M3        M4        M5        M6
%        (30°)    (330°)    (270°)    (210°)    (150°)     (90°)
%         CCW       CW       CCW        CW        CW        CCW

B = [
% T: total thrust
   k,        k,        k,        k,        k,        k;

% τx (roll): -y*k
  -k*L/2,   k*L/2,    k*L,     k*L/2,   -k*L/2,   -k*L;

% τy (pitch): +x*k
  k*L*s3,  k*L*s3,    0,      -k*L*s3, -k*L*s3,    0;

% τz (yaw): CCW(+b), CW(-b)
   b,       -b,        b,       -b,       -b,       b
];

%% Compute output
if strcmp(mode, 'inverse')
    output = B \ input;
else
    output = B * input;
end

end
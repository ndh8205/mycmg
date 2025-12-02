function [output, B] = control_allocator(input, params, mode)
% control_allocator: Quadrotor control allocation
%
% Usage:
%   [omega_sq, B] = control_allocator(cmd_vec, params, 'inverse')
%   [cmd_vec, B]  = control_allocator(omega_sq, params, 'forward')
%
% Inputs:
%   input  - 4x1 vector
%            'inverse': [T; τx; τy; τz] → omega_sq
%            'forward': [ω0²; ω1²; ω2²; ω3²] → [T; τx; τy; τz]
%   params - parameter struct
%   mode   - 'inverse' (default) or 'forward'
%
% Outputs:
%   output - 4x1 vector
%   B      - 4x4 allocation matrix

if nargin < 3
    mode = 'inverse';
end

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
%
%  Rotor 1: Front-Left,  CCW, (+l_x, -l_y1)
%  Rotor 2: Back-Right,  CCW, (-l_x, +l_y2)
%  Rotor 3: Front-Right, CW,  (+l_x, +l_y1)
%  Rotor 4: Back-Left,   CW,  (-l_x, -l_y2)
%
%  Torque convention (τ = r × F, F = [0;0;-T]):
%    τx = -y*T  (left rotor → +roll, right rotor → -roll)
%    τy = +x*T  (front rotor → +pitch, back rotor → -pitch)
%    τz = reaction torque (CCW rotor → +yaw, CW rotor → -yaw)

%            ω1²        ω2²        ω3²        ω4²
%           ─────      ─────      ─────      ─────
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
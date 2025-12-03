function [F_aero, M_aero] = aero_model(vel_air_b, omega_b, u_ctrl, params)
% aero_model: Aerodynamic force and moment calculation
%             Based on Gazebo LiftDragPlugin formulation
%
% Inputs:
%   vel_air_b - 3x1 airspeed in body frame [m/s]
%   omega_b   - 3x1 angular velocity in body frame [rad/s]
%   u_ctrl    - 3x1 control surface commands [rad]
%               [left_elevon; right_elevon; elevator]
%   params    - parameter struct
%
% Outputs:
%   F_aero    - 3x1 aerodynamic force in body frame [N]
%   M_aero    - 3x1 aerodynamic moment in body frame [N*m]
%
% Coordinate: NED body frame (X-fwd, Y-right, Z-down)

%% Parameters
rho = params.env.rho;

% Wing parameters
S_wing    = params.aero.wing.area;        % per side [m²]
a0        = params.aero.wing.a0;          % zero-lift AoA [rad]
CL_alpha  = params.aero.wing.CL_alpha;    % lift slope [1/rad]
CD_alpha  = params.aero.wing.CD_alpha;    % drag slope
CD_0      = params.aero.wing.CD_0;        % zero-lift drag
alpha_stall = params.aero.wing.alpha_stall;
CL_stall  = params.aero.wing.CL_stall;    % post-stall slope
CD_stall  = params.aero.wing.CD_stall;

cp_left   = params.aero.wing.cp_left;     % [m]
cp_right  = params.aero.wing.cp_right;

% Elevon parameters
CL_delta_elevon = params.aero.elevon.CL_delta;
elevon_max = params.aero.elevon.max;

% Elevator parameters
S_elev    = params.aero.elevator.area;
cp_elev   = params.aero.elevator.cp;
CL_delta_elev = params.aero.elevator.CL_delta;

%% Airspeed components
V = norm(vel_air_b);
if V < 1.0
    F_aero = zeros(3,1);
    M_aero = zeros(3,1);
    return;
end

u = vel_air_b(1);  % forward
v = vel_air_b(2);  % right
w = vel_air_b(3);  % down

%% Angle of attack and sideslip
% Guard against low forward speed
u_safe = max(abs(u), 0.1) * sign(u + 1e-6);
alpha = atan2(w, u_safe);          % AoA [rad]
beta  = atan2(v, sqrt(u^2 + w^2)); % sideslip [rad]

% Limit alpha for stability
alpha = max(min(alpha, 45*pi/180), -45*pi/180);

%% Dynamic pressure
q_bar = 0.5 * rho * V^2;

%% Control surface inputs
delta_elevon_L = u_ctrl(1);
delta_elevon_R = u_ctrl(2);
delta_elevator = u_ctrl(3);

% Saturate
delta_elevon_L = max(min(delta_elevon_L, elevon_max), -elevon_max);
delta_elevon_R = max(min(delta_elevon_R, elevon_max), -elevon_max);
delta_elevator = max(min(delta_elevator, elevon_max), -elevon_max);

%% ========== LEFT WING ==========
[F_L, M_L] = wing_forces(alpha, q_bar, S_wing, cp_left, ...
    a0, CL_alpha, CD_alpha, CD_0, alpha_stall, CL_stall, CD_stall, ...
    delta_elevon_L, CL_delta_elevon);

%% ========== RIGHT WING ==========
[F_R, M_R] = wing_forces(alpha, q_bar, S_wing, cp_right, ...
    a0, CL_alpha, CD_alpha, CD_0, alpha_stall, CL_stall, CD_stall, ...
    delta_elevon_R, CL_delta_elevon);

%% ========== ELEVATOR ==========
[F_E, M_E] = elevator_forces(alpha, q_bar, S_elev, cp_elev, ...
    delta_elevator, CL_delta_elev, CL_alpha, CD_alpha, CD_0);

%% ========== BODY DRAG ==========
% Simple fuselage drag
CD_body = 0.1;
A_body = 0.05;  % frontal area [m²]
D_body = q_bar * CD_body * A_body;
F_body_drag = -D_body * (vel_air_b / V);

%% ========== TOTAL ==========
F_aero = F_L + F_R + F_E + F_body_drag;
M_aero = M_L + M_R + M_E;

end

%% ========== LOCAL FUNCTIONS ==========

function [F, M] = wing_forces(alpha, q_bar, S, cp, ...
    a0, CL_alpha, CD_alpha, CD_0, alpha_stall, CL_stall, CD_stall, ...
    delta_ctrl, CL_delta)
% Wing lift and drag calculation with stall model
%
% Gazebo LiftDrag model:
%   Pre-stall:  CL = CL_alpha * (alpha - a0)
%   Post-stall: CL = CL_stall * (alpha - alpha_stall) + CL_max

    % Effective AoA with control surface
    alpha_eff = alpha - a0 + delta_ctrl * CL_delta / CL_alpha;
    
    % Lift coefficient
    if abs(alpha) < alpha_stall
        % Pre-stall (linear region)
        CL = CL_alpha * alpha_eff;
        CD = CD_0 + CD_alpha * alpha_eff^2;
    else
        % Post-stall
        % CL_max at stall
        CL_max = CL_alpha * (sign(alpha) * alpha_stall - a0);
        CL = CL_max + CL_stall * (alpha - sign(alpha) * alpha_stall);
        
        % Increased drag in stall
        CD_max = CD_0 + CD_alpha * alpha_stall^2;
        CD = CD_max + CD_stall * (abs(alpha) - alpha_stall);
    end
    
    % Control surface drag increment
    CD = CD + 0.01 * abs(delta_ctrl);
    
    % Lift and drag forces (wind frame)
    L = q_bar * S * CL;
    D = q_bar * S * CD;
    
    % Convert to body frame
    % Lift: perpendicular to velocity, in XZ plane
    % Drag: opposite to velocity
    ca = cos(alpha);
    sa = sin(alpha);
    
    % Force in body frame
    % F_x = -D*cos(alpha) + L*sin(alpha)
    % F_z = -D*sin(alpha) - L*cos(alpha)
    F = zeros(3,1);
    F(1) = -D * ca + L * sa;
    F(3) = -D * sa - L * ca;
    
    % Moment about CG
    M = cross(cp, F);
end

function [F, M] = elevator_forces(alpha, q_bar, S, cp, ...
    delta_elev, CL_delta, CL_alpha, CD_alpha, CD_0)
% Elevator (horizontal stabilizer) forces

    % Effective AoA
    alpha_eff = alpha + delta_elev * CL_delta / CL_alpha;
    
    % Coefficients (simplified)
    CL = CL_alpha * alpha_eff * 0.5;  % reduced effectiveness
    CD = CD_0 + 0.05 * alpha_eff^2;
    
    % Forces
    L = q_bar * S * CL;
    D = q_bar * S * CD;
    
    ca = cos(alpha);
    sa = sin(alpha);
    
    F = zeros(3,1);
    F(1) = -D * ca + L * sa;
    F(3) = -D * sa - L * ca;
    
    M = cross(cp, F);
end
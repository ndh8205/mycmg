function [F_wind, dist_state] = dist_wind(vel_b, R_b2n, t, dt, params, dist_state)
% dist_wind: Calculate wind disturbance force in body frame
%
% Inputs:
%   vel_b      - 3x1 body velocity [m/s]
%   R_b2n      - 3x3 rotation matrix (body to NED)
%   t          - current time [s]
%   dt         - time step [s]
%   params     - parameter struct with dist field
%   dist_state - disturbance state struct
%
% Outputs:
%   F_wind     - 3x1 wind force [N] (body frame)
%   dist_state - updated state

F_wind = zeros(3,1);

if ~params.dist.enable
    return;
end

R_n2b = R_b2n';
wind_type = params.dist.wind.type;

% Get wind velocity in NED frame
switch wind_type
    case 'none'
        V_wind_ned = zeros(3,1);
        
    case 'constant'
        V_wind_ned = params.dist.wind.velocity_ned(:);
        
    case 'gust'
        % Base constant wind
        V_wind_ned = params.dist.wind.velocity_ned(:);
        
        % Add gust component
        t_start = params.dist.wind.gust_start;
        t_dur = params.dist.wind.gust_duration;
        
        if t >= t_start && t < t_start + t_dur
            % Gust active
            if ~dist_state.gust.active
                % Initialize gust direction (random in horizontal plane)
                theta_gust = 2*pi*rand();
                dist_state.gust.direction = [cos(theta_gust); sin(theta_gust); 0.1*randn()];
                dist_state.gust.direction = dist_state.gust.direction / norm(dist_state.gust.direction);
                dist_state.gust.active = true;
            end
            
            % 1-cosine gust profile
            t_rel = t - t_start;
            gust_profile = 0.5 * (1 - cos(2*pi*t_rel/t_dur));
            V_gust = params.dist.wind.gust_magnitude * gust_profile * dist_state.gust.direction;
            V_wind_ned = V_wind_ned + V_gust;
        else
            dist_state.gust.active = false;
        end
        
    case 'dryden'
        % Dryden turbulence model
        [V_turb, dist_state] = dryden_filter(dt, params, dist_state);
        
        % Add to base wind (small constant component)
        % V_base = 0.5 * params.dist.wind.velocity_ned(:);
        V_base = params.dist.wind.velocity_ned(:);
        V_wind_ned = V_base + V_turb;
        
    otherwise
        V_wind_ned = zeros(3,1);
end

% Transform wind to body frame
V_wind_body = R_n2b * V_wind_ned;

% Relative airspeed in body frame
V_air = vel_b - V_wind_body;

% Aerodynamic drag force
Cd = params.dist.aero.Cd;
A = params.dist.aero.A;
rho = params.dist.aero.rho;

V_air_mag = norm(V_air);
if V_air_mag > 0.01
    F_wind = -0.5 * rho * Cd * A * V_air_mag * V_air;
else
    F_wind = zeros(3,1);
end

end

%% Local function: Dryden filter update
function [V_turb, dist_state] = dryden_filter(dt, params, dist_state)
% Update Dryden turbulence filter state

A = dist_state.dryden.A;
B = dist_state.dryden.B;
C = dist_state.dryden.C;
x = dist_state.dryden.x;

% White noise input
noise = randn();

% Euler integration of state-space model
x_dot = A * x + B * noise;
x = x + x_dot * dt;

% Output: turbulence velocity in NED
V_turb = C * x;

% Update state
dist_state.dryden.x = x;

end
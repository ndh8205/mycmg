function [x, Thrust_mass, alpha_tot, phi_A, F_aero, M_aero, dist_state] = srk4(Func, x, u, Q, dt, params, i, delt, t, dist_state)
% srk4: Stochastic Runge-Kutta 4th order integrator with disturbance support
%
% Inputs:
%   Func   - dynamics function handle (@drone_dynamics)
%   x      - state vector
%   u      - control input
%   Q      - process noise covariance matrix
%   dt     - time step for dynamics
%   params - parameter struct
%   i      - step index
%   delt   - integration step size
%   t      - current time [s] (optional)
%   dist_state - disturbance state struct (optional)
%
% Outputs:
%   x          - updated state
%   Thrust_mass - total thrust
%   alpha_tot  - [] (reserved)
%   phi_A      - [] (reserved)
%   F_aero     - total force (body)
%   M_aero     - total moment (body)
%   dist_state - updated disturbance state

    % Handle optional inputs
    if nargin < 9
        t = i * delt;
    end
    if nargin < 10
        dist_state = [];
    end
    
    % Scaling factor for stochastic integration
    alpha = [1.0/6.0; 2.0/6.0; 2.0/6.0; 1.0/6.0];
    beta = 1 / (alpha' * alpha);
    sigma_square = diag(Q);
    ScaledQ = beta * sigma_square / delt;
    
    % Generate random noise vectors
    w1 = sqrt(ScaledQ) .* randn(length(ScaledQ), 1);
    w2 = sqrt(ScaledQ) .* randn(length(ScaledQ), 1);
    w3 = sqrt(ScaledQ) .* randn(length(ScaledQ), 1);
    w4 = sqrt(ScaledQ) .* randn(length(ScaledQ), 1);
    
    % Time at each RK4 stage
    t1 = t;
    t2 = t + 0.5 * delt;
    t3 = t + 0.5 * delt;
    t4 = t + delt;
    
    % Compute k1
    [k1, ~, ~, ~, ~, ~, ~] = Func(x, u, w1, dt, params, i, t1, dist_state);
    k1 = k1 * delt;
    
    % Compute k2
    [k2, ~, ~, ~, ~, ~, ~] = Func(x + 0.5 * k1, u, w2, dt, params, i, t2, dist_state);
    k2 = k2 * delt;
    
    % Compute k3
    [k3, ~, ~, ~, ~, ~, ~] = Func(x + 0.5 * k2, u, w3, dt, params, i, t3, dist_state);
    k3 = k3 * delt;
    
    % Compute k4 and capture diagnostics + updated dist_state
    [k4, Thrust_mass, alpha_tot, phi_A, F_aero, M_aero, dist_state] = ...
        Func(x + k3, u, w4, dt, params, i, t4, dist_state);
    k4 = k4 * delt;
    
    % Update state
    x = x + (k1 + 2*(k2 + k3) + k4) / 6.0;
end
function [x, Thrust_mass, alpha_tot, phi_A, F_aero, M_aero, moment_coupling] = srk4(Func, x, u, Q, dt, params, i, delt)
    % Scaling factor for stochastic integration
    alpha = [1.0/6.0; 2.0/6.0; 2.0/6.0; 1.0/6.0];
    beta = 1 / (alpha' * alpha);
    sigma_square = diag(Q); % Compute diagonal terms only for simplicity
    ScaledQ = beta * sigma_square / delt;
    
    % Generate random noise vectors
    w1 = sqrt(ScaledQ) .* randn(length(ScaledQ), 1);
    w2 = sqrt(ScaledQ) .* randn(length(ScaledQ), 1);
    w3 = sqrt(ScaledQ) .* randn(length(ScaledQ), 1);
    w4 = sqrt(ScaledQ) .* randn(length(ScaledQ), 1);
    
    % Compute k1 and capture Thrust_mass if needed
    [k1, ~] = Func(x, u, w1, dt, params, i);
    k1 = k1 * delt;
    
    % Compute k2
    [k2, ~] = Func(x + 0.5 * k1, u, w2, dt, params, i);
    k2 = k2 * delt;
    
    % Compute k3
    [k3, ~] = Func(x + 0.5 * k2, u, w3, dt, params, i);
    k3 = k3 * delt;
    
    % Compute k4 and capture all diagnostic data
    [k4, Thrust_mass, alpha_tot, phi_A, F_aero, M_aero, moment_coupling] = Func(x + k3, u, w4, dt, params, i);
    k4 = k4 * delt;
    
    % Update state
    x = x + (k1 + 2*(k2 + k3) + k4) / 6.0;
end
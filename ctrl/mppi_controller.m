function [u_opt, mppi_state] = mppi_controller(x_current, mppi_state, mppi_gains, params)
% mppi_controller: Model Predictive Path Integral Controller (GPU Optimized)
%
% Inputs:
%   x_current  - 28x1 or 13x1 current state
%   mppi_state - struct containing:
%                .u_seq    - 6 x N current control sequence
%                .pos_des  - 3x1 desired position
%                .yaw_des  - desired yaw [rad]
%   mppi_gains - struct containing MPPI parameters
%   params     - system parameters
%
% Outputs:
%   u_opt      - 6x1 optimal motor speed command [rad/s]
%   mppi_state - updated state (shifted control sequence)

%% Persistent GPU arrays (avoid repeated allocation)
persistent gpu_initialized
persistent K_prev N_prev
persistent m_p Jxx_p Jyy_p Jzz_p g_p k_T_p k_M_p L_p omega_max_p omega_min_p
persistent delta_u_gpu x_all S weights u_seq_gpu pos_des_gpu R_diag_gpu
persistent rand_buffer rand_idx RAND_BUFFER_SIZE

%% Extract MPPI parameters
K = mppi_gains.K;
N = mppi_gains.N;
dt = mppi_gains.dt;
lambda = mppi_gains.lambda;
nu = mppi_gains.nu;
sigma = mppi_gains.sigma;

% Cost weights (convert to single once)
w_pos = single(mppi_gains.w_pos);
w_vel = single(mppi_gains.w_vel);
w_att = single(mppi_gains.w_att);
w_yaw = single(mppi_gains.w_yaw);
w_omega = single(mppi_gains.w_omega);
w_terminal = single(mppi_gains.w_terminal);
R_diag = single(diag(mppi_gains.R));
dt_s = single(dt);
lambda_s = single(lambda);
nu_inv = single(1 - 1/nu);

%% Initialize GPU cached parameters (only once or when size changes)
if isempty(gpu_initialized) || K ~= K_prev || N ~= N_prev
    fprintf('Initializing MPPI GPU arrays (K=%d, N=%d)...\n', K, N);
    
    % Cache parameters as single precision
    m_p = single(params.drone.body.m);
    J = params.drone.body.J;
    Jxx_p = single(J(1,1));
    Jyy_p = single(J(2,2));
    Jzz_p = single(J(3,3));
    g_p = single(params.env.g);
    k_T_p = single(params.drone.motor.k_T);
    k_M_p = single(params.drone.motor.k_M);
    L_p = single(params.drone.body.L);
    omega_max_p = single(params.drone.motor.omega_b_max);
    omega_min_p = single(params.drone.motor.omega_b_min);
    
    % Preallocate GPU arrays
    delta_u_gpu = gpuArray.zeros(6, N, K, 'single');
    x_all = gpuArray.zeros(13, K, 'single');
    S = gpuArray.zeros(1, K, 'single');
    weights = gpuArray.zeros(1, K, 'single');
    u_seq_gpu = gpuArray.zeros(6, N, 'single');
    pos_des_gpu = gpuArray.zeros(3, 1, 'single');
    R_diag_gpu = gpuArray.zeros(6, 1, 'single');
    
    % Pre-generate random buffer (avoid randn every call)
    RAND_BUFFER_SIZE = 50;  % 50 calls worth
    fprintf('Pre-generating random buffer (%d samples)...\n', RAND_BUFFER_SIZE);
    rand_buffer = gpuArray.randn(6, N, K, RAND_BUFFER_SIZE, 'single');
    rand_idx = 1;
    
    K_prev = K;
    N_prev = N;
    gpu_initialized = true;
end

%% Extract current state (reduce to 13 states)
if length(x_current) >= 13
    x0 = single(x_current(1:13));
else
    x0 = single(x_current);
end

%% Desired state
pos_des = single(mppi_state.pos_des(:));
yaw_des = single(mppi_state.yaw_des);

%% Update GPU arrays (copy values, no new allocation)
% Get pre-generated random perturbations from buffer
delta_u_gpu(:) = sigma * rand_buffer(:,:,:,rand_idx);
rand_idx = mod(rand_idx, RAND_BUFFER_SIZE) + 1;

% Copy values to preallocated arrays
u_seq_gpu(:) = single(mppi_state.u_seq);
pos_des_gpu(:) = pos_des;
R_diag_gpu(:) = R_diag;

%% Initialize state array (broadcast x0 to all K rollouts)
x0_gpu = gpuArray(x0);
x_all(:,:) = repmat(x0_gpu, 1, K);

%% Reset cost accumulator
S(:) = 0;

%% Precompute constants for dynamics
k = k_T_p;
b = k_T_p * k_M_p;
s3 = single(sqrt(3)/2);

%% Rollout all trajectories
for i = 1:N
    % Get control perturbation for this timestep
    delta_u_i = reshape(delta_u_gpu(:, i, :), 6, K);
    u_nom_i = u_seq_gpu(:, i);
    u_i = u_nom_i + delta_u_i;
    
    % Saturate
    u_i = max(min(u_i, omega_max_p), omega_min_p);
    
    % ===== Running Cost =====
    % Position cost
    pos_err = x_all(1:3,:) - pos_des_gpu;
    cost_pos = w_pos * sum(pos_err.^2, 1);
    
    % Velocity cost
    cost_vel = w_vel * sum(x_all(4:6,:).^2, 1);
    
    % Attitude cost
    qw = x_all(7,:);
    cost_att = w_att * (1 - qw.^2);
    
    % Yaw cost
    qx = x_all(8,:); qy = x_all(9,:); qz = x_all(10,:);
    yaw_curr = atan2(2*(qw.*qz + qx.*qy), 1 - 2*(qy.^2 + qz.^2));
    yaw_err = atan2(sin(yaw_curr - yaw_des), cos(yaw_curr - yaw_des));
    cost_yaw = w_yaw * yaw_err.^2;
    
    % Angular velocity cost
    cost_omega = w_omega * sum(x_all(11:13,:).^2, 1);
    
    % Control cost (importance sampling)
    delta_u_R = sum(delta_u_i.^2 .* R_diag_gpu, 1);
    u_nom_R_delta = sum((u_nom_i .* R_diag_gpu) .* delta_u_i, 1);
    u_nom_R = sum((u_nom_i.^2) .* R_diag_gpu, 1);
    cost_ctrl = nu_inv/2 * delta_u_R + u_nom_R_delta + 0.5 * u_nom_R;
    
    % Accumulate
    S = S + (cost_pos + cost_vel + cost_att + cost_yaw + cost_omega + cost_ctrl) * dt_s;
    
    % ===== Dynamics Step (inline) =====
    % Motor forces
    omega_sq = u_i.^2;
    T = k * sum(omega_sq, 1);
    tau_x = k * L_p * (-0.5*omega_sq(1,:) + 0.5*omega_sq(2,:) + omega_sq(3,:) ...
                       + 0.5*omega_sq(4,:) - 0.5*omega_sq(5,:) - omega_sq(6,:));
    tau_y = k * L_p * s3 * (omega_sq(1,:) + omega_sq(2,:) - omega_sq(4,:) - omega_sq(5,:));
    tau_z = b * (omega_sq(1,:) - omega_sq(2,:) + omega_sq(3,:) ...
                - omega_sq(4,:) + omega_sq(5,:) - omega_sq(6,:));
    
    % Extract states
    vel = x_all(4:6,:);
    wx = x_all(11,:); wy = x_all(12,:); wz = x_all(13,:);
    
    % Rotation matrix elements
    R11 = 1 - 2*(qy.^2 + qz.^2);
    R12 = 2*(qx.*qy - qz.*qw);
    R13 = 2*(qx.*qz + qy.*qw);
    R21 = 2*(qx.*qy + qz.*qw);
    R22 = 1 - 2*(qx.^2 + qz.^2);
    R23 = 2*(qy.*qz - qx.*qw);
    R31 = 2*(qx.*qz - qy.*qw);
    R32 = 2*(qy.*qz + qx.*qw);
    R33 = 1 - 2*(qx.^2 + qy.^2);
    
    % Position derivative: R_b2n * vel_body
    pos_dot_1 = R11.*vel(1,:) + R12.*vel(2,:) + R13.*vel(3,:);
    pos_dot_2 = R21.*vel(1,:) + R22.*vel(2,:) + R23.*vel(3,:);
    pos_dot_3 = R31.*vel(1,:) + R32.*vel(2,:) + R33.*vel(3,:);
    
    % Velocity derivative: F/m + R'*g - omega x vel
    % F_body = [0; 0; -T], g_ned = [0; 0; g]
    % R_n2b * g_ned = [R31; R32; R33] * g
    gx = R31 * g_p;
    gy = R32 * g_p;
    gz = R33 * g_p;
    
    % omega x vel
    cross_x = wy.*vel(3,:) - wz.*vel(2,:);
    cross_y = wz.*vel(1,:) - wx.*vel(3,:);
    cross_z = wx.*vel(2,:) - wy.*vel(1,:);
    
    vel_dot_1 = gx - cross_x;
    vel_dot_2 = gy - cross_y;
    vel_dot_3 = -T/m_p + gz - cross_z;
    
    % Quaternion derivative
    quat_dot_0 = 0.5 * (-wx.*qx - wy.*qy - wz.*qz);
    quat_dot_1 = 0.5 * ( wx.*qw + wz.*qy - wy.*qz);
    quat_dot_2 = 0.5 * ( wy.*qw - wz.*qx + wx.*qz);
    quat_dot_3 = 0.5 * ( wz.*qw + wy.*qx - wx.*qy);
    
    % Angular velocity derivative
    Jw_x = Jxx_p * wx;
    Jw_y = Jyy_p * wy;
    Jw_z = Jzz_p * wz;
    
    omega_dot_x = (tau_x - (wy.*Jw_z - wz.*Jw_y)) / Jxx_p;
    omega_dot_y = (tau_y - (wz.*Jw_x - wx.*Jw_z)) / Jyy_p;
    omega_dot_z = (tau_z - (wx.*Jw_y - wy.*Jw_x)) / Jzz_p;
    
    % Euler integration
    x_all(1,:) = x_all(1,:) + pos_dot_1 * dt_s;
    x_all(2,:) = x_all(2,:) + pos_dot_2 * dt_s;
    x_all(3,:) = x_all(3,:) + pos_dot_3 * dt_s;
    x_all(4,:) = x_all(4,:) + vel_dot_1 * dt_s;
    x_all(5,:) = x_all(5,:) + vel_dot_2 * dt_s;
    x_all(6,:) = x_all(6,:) + vel_dot_3 * dt_s;
    x_all(7,:) = x_all(7,:) + quat_dot_0 * dt_s;
    x_all(8,:) = x_all(8,:) + quat_dot_1 * dt_s;
    x_all(9,:) = x_all(9,:) + quat_dot_2 * dt_s;
    x_all(10,:) = x_all(10,:) + quat_dot_3 * dt_s;
    x_all(11,:) = x_all(11,:) + omega_dot_x * dt_s;
    x_all(12,:) = x_all(12,:) + omega_dot_y * dt_s;
    x_all(13,:) = x_all(13,:) + omega_dot_z * dt_s;
    
    % Normalize quaternion
    qnorm = sqrt(x_all(7,:).^2 + x_all(8,:).^2 + x_all(9,:).^2 + x_all(10,:).^2);
    x_all(7:10,:) = x_all(7:10,:) ./ qnorm;
    
    % Update qw for next cost calculation
    qw = x_all(7,:);
    qx = x_all(8,:);
    qy = x_all(9,:);
    qz = x_all(10,:);
end

%% Terminal cost
pos_err_T = x_all(1:3,:) - pos_des_gpu;
cost_pos_T = w_terminal * sum(pos_err_T.^2, 1);
cost_att_T = w_att * (1 - x_all(7,:).^2);
cost_vel_T = 0.5 * w_terminal * sum(x_all(4:6,:).^2, 1);
S = S + cost_pos_T + cost_att_T + cost_vel_T;

%% Compute importance weights
S_min = min(S);
S_shifted = S - S_min;
weights(:) = exp(-S_shifted / lambda_s);
weights_sum = sum(weights);

if weights_sum > 1e-10
    weights(:) = weights / weights_sum;
else
    weights(:) = 1 / single(K);
end

%% Weighted average of control perturbations
weights_3d = reshape(weights, 1, 1, K);
weighted_delta = sum(delta_u_gpu .* weights_3d, 3);

%% Update control sequence
u_seq_new = u_seq_gpu + weighted_delta;
u_seq_new = max(min(u_seq_new, omega_max_p), omega_min_p);

%% Gather result (single GPU->CPU transfer)
u_seq_cpu = double(gather(u_seq_new));

%% Output
u_opt = u_seq_cpu(:, 1);

%% Shift control sequence
mppi_state.u_seq(:, 1:N-1) = u_seq_cpu(:, 2:N);
mppi_state.u_seq(:, N) = u_seq_cpu(:, N);

end
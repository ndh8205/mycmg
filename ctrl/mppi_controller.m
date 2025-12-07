function [u_opt, mppi_state] = mppi_controller(x_current, mppi_state, mppi_params, params)
% mppi_controller: MPPI with Generalized Importance Sampling
% Based on Williams et al. 2017, Section III.C
%
% Key equations from paper:
%   δu = (1/√ρ)(ε/√Δt)  - control perturbation
%   q̃ = q + (1-ν⁻¹)/2 δu'Rδu + u'Rδu + (1/2)u'Ru  - augmented running cost
%   u* = u + E_q[exp(-S̃/λ)δu] / E_q[exp(-S̃/λ)]  - optimal update

%% ========== PERSISTENT GPU MEMORY ==========
persistent gpu_init K_c N_c batch_c
persistent S_all weights_all
persistent u_seq_gpu pos_des_gpu
persistent rand_buffer rand_idx
persistent m_g Jxx_g Jyy_g Jzz_g g_g k_g b_g L_g s3_g
persistent omega_max_g omega_min_g
persistent sigma_gpu R_gpu

%% ========== PARAMETERS ==========
K = mppi_params.K;
N = mppi_params.N;
dt = single(mppi_params.dt);
lambda = single(mppi_params.lambda);
nu = single(mppi_params.nu);  % Importance sampling variance scaling

% Importance sampling coefficient: (1 - 1/ν)
nu_coeff = single(1 - 1/nu);

%% ========== GPU INITIALIZATION ==========
if isempty(gpu_init) || K ~= K_c || N ~= N_c
    fprintf('[MPPI] GPU Init (K=%d, N=%d, ν=%.1f)...\n', K, N, nu);
    
    % Drone parameters (single precision scalars)
    m_g = single(params.drone.body.m);
    J = params.drone.body.J;
    Jxx_g = single(J(1,1));
    Jyy_g = single(J(2,2));
    Jzz_g = single(J(3,3));
    g_g = single(params.env.g);
    k_g = single(params.drone.motor.k_T);
    b_g = single(params.drone.motor.k_T * params.drone.motor.k_M);
    L_g = single(params.drone.body.L);
    s3_g = single(sqrt(3)/2);
    omega_max_g = single(params.drone.motor.omega_b_max);
    omega_min_g = single(params.drone.motor.omega_b_min);
    
    % Batch size (GPU memory adaptive)
    gpu_info = gpuDevice();
    mem_per_sample = 13 * N * 4 * 3;
    batch_c = min(K, max(128, floor(gpu_info.AvailableMemory * 0.15 / mem_per_sample)));
    fprintf('[MPPI] Batch size: %d\n', batch_c);
    
    % Preallocate
    S_all = gpuArray.zeros(1, K, 'single');
    weights_all = gpuArray.zeros(1, K, 'single');
    u_seq_gpu = gpuArray.zeros(6, N, 'single');
    pos_des_gpu = gpuArray.zeros(3, 1, 'single');
    
    % Control parameters
    sigma_gpu = gpuArray(single(mppi_params.sigma(:)));
    R_gpu = gpuArray(single(mppi_params.R(:)));
    
    % Random buffer (standard normal)
    rand_buffer = gpuArray.randn(6, N, K, 'single');
    rand_idx = 0;
    
    K_c = K; N_c = N;
    gpu_init = true;
    fprintf('[MPPI] Init complete.\n');
end

%% ========== PREPARE INPUTS ==========
x0 = single(x_current(1:13));
pos_des_gpu(:) = single(mppi_state.pos_des(:));
yaw_des = single(mppi_state.yaw_des);
u_seq_gpu(:) = single(mppi_state.u_seq);

% Cost weights
w_pos = single(mppi_params.w_pos);
w_vel = single(mppi_params.w_vel);
w_att = single(mppi_params.w_att);
w_yaw = single(mppi_params.w_yaw);
w_omega = single(mppi_params.w_omega);
w_term = single(mppi_params.w_terminal);

%% ========== GENERATE NOISE ==========
% Paper Eq: δu = (1/√ρ)(ε/√Δt), here sigma absorbs 1/√ρ
% Scale by 1/√Δt for proper discrete-time noise
rand_idx = rand_idx + 1;
if rand_idx > 50
    rand_buffer = gpuArray.randn(6, N, K, 'single');
    rand_idx = 1;
end

% Control perturbation: δu = sigma * ε / √Δt
sqrt_dt = single(sqrt(dt));
noise_all = (sigma_gpu / sqrt_dt) .* rand_buffer;

%% ========== BATCH ROLLOUT ==========
num_batches = ceil(K / batch_c);

for bi = 1:num_batches
    i1 = (bi-1) * batch_c + 1;
    i2 = min(bi * batch_c, K);
    B = i2 - i1 + 1;
    
    % Batch noise (δu)
    dU = noise_all(:, :, i1:i2);
    
    % Initialize state
    X = repmat(gpuArray(x0(:)), 1, B);
    S = gpuArray.zeros(1, B, 'single');
    
    % Rollout
    for t = 1:N
        % Nominal control and perturbation
        u_nom = u_seq_gpu(:, t);
        du = reshape(dU(:, t, :), 6, B);
        
        % Actual control: u = u_nom + δu
        u = max(min(u_nom + du, omega_max_g), omega_min_g);
        
        % Extract states
        vel = X(4:6, :);
        qw = X(7, :); qx = X(8, :); qy = X(9, :); qz = X(10, :);
        wx = X(11, :); wy = X(12, :); wz = X(13, :);
        
        % ===== FORCES & TORQUES (matches control_allocator.m) =====
        w2 = u.^2;
        T = k_g * sum(w2, 1);
        
        % τx (roll): coefficients from B matrix row 2
        tau_x = k_g * L_g * (-0.5*w2(1,:) + 0.5*w2(2,:) + w2(3,:) ...
                            + 0.5*w2(4,:) - 0.5*w2(5,:) - w2(6,:));
        
        % τy (pitch): coefficients from B matrix row 3
        tau_y = k_g * L_g * s3_g * (w2(1,:) + w2(2,:) - w2(4,:) - w2(5,:));
        
        % τz (yaw): CCW(+), CW(-)
        tau_z = b_g * (w2(1,:) - w2(2,:) + w2(3,:) - w2(4,:) + w2(5,:) - w2(6,:));
        
        % ===== DCM ELEMENTS =====
        R11 = 1 - 2*(qy.^2 + qz.^2);
        R12 = 2*(qx.*qy - qz.*qw);
        R13 = 2*(qx.*qz + qy.*qw);
        R21 = 2*(qx.*qy + qz.*qw);
        R22 = 1 - 2*(qx.^2 + qz.^2);
        R23 = 2*(qy.*qz - qx.*qw);
        R31 = 2*(qx.*qz - qy.*qw);
        R32 = 2*(qy.*qz + qx.*qw);
        R33 = 1 - 2*(qx.^2 + qy.^2);
        
        % ===== DERIVATIVES =====
        % Position: ṗ = R * v_body
        pos_dot = [R11.*vel(1,:) + R12.*vel(2,:) + R13.*vel(3,:);
                   R21.*vel(1,:) + R22.*vel(2,:) + R23.*vel(3,:);
                   R31.*vel(1,:) + R32.*vel(2,:) + R33.*vel(3,:)];
        
        % Velocity: v̇ = F/m + R'*g - ω×v
        gx = R31 * g_g; gy = R32 * g_g; gz = R33 * g_g;
        vel_dot = [gx - (wy.*vel(3,:) - wz.*vel(2,:));
                   gy - (wz.*vel(1,:) - wx.*vel(3,:));
                   -T/m_g + gz - (wx.*vel(2,:) - wy.*vel(1,:))];
        
        % Quaternion derivative
        qw_dot = 0.5 * (-wx.*qx - wy.*qy - wz.*qz);
        qx_dot = 0.5 * ( wx.*qw + wz.*qy - wy.*qz);
        qy_dot = 0.5 * ( wy.*qw - wz.*qx + wx.*qz);
        qz_dot = 0.5 * ( wz.*qw + wy.*qx - wx.*qy);
        
        % Angular velocity derivative: ω̇ = J⁻¹(τ - ω×Jω)
        omega_dot_x = (tau_x - (wy.*(Jzz_g*wz) - wz.*(Jyy_g*wy))) / Jxx_g;
        omega_dot_y = (tau_y - (wz.*(Jxx_g*wx) - wx.*(Jzz_g*wz))) / Jyy_g;
        omega_dot_z = (tau_z - (wx.*(Jyy_g*wy) - wy.*(Jxx_g*wx))) / Jzz_g;
        
        % ===== EULER INTEGRATION =====
        X(1:3, :) = X(1:3, :) + pos_dot * dt;
        X(4:6, :) = X(4:6, :) + vel_dot * dt;
        X(7, :) = qw + qw_dot * dt;
        X(8, :) = qx + qx_dot * dt;
        X(9, :) = qy + qy_dot * dt;
        X(10, :) = qz + qz_dot * dt;
        X(11:13, :) = X(11:13, :) + [omega_dot_x; omega_dot_y; omega_dot_z] * dt;
        
        % Normalize quaternion
        qn = sqrt(X(7,:).^2 + X(8,:).^2 + X(9,:).^2 + X(10,:).^2);
        X(7:10, :) = X(7:10, :) ./ qn;
        
        % ===== STATE-DEPENDENT COST q(x) =====
        pos_err = X(1:3, :) - pos_des_gpu;
        c_state = w_pos * sum(pos_err.^2, 1) + ...
                  w_vel * sum(X(4:6, :).^2, 1) + ...
                  w_att * (1 - X(7, :).^2) + ...
                  w_omega * sum(X(11:13, :).^2, 1);
        
        % Yaw cost
        yaw_c = atan2(2*(X(7,:).*X(10,:) + X(8,:).*X(9,:)), ...
                      1 - 2*(X(9,:).^2 + X(10,:).^2));
        c_state = c_state + w_yaw * atan2(sin(yaw_c - yaw_des), cos(yaw_c - yaw_des)).^2;
        
        % ===== IMPORTANCE SAMPLING CONTROL COST (Paper Eq. III.C) =====
        % q̃ = q + (1-ν⁻¹)/2 * δu'Rδu + u'Rδu + (1/2)u'Ru
        
        % Term 1: (1-ν⁻¹)/2 * δu'Rδu
        c_ctrl_1 = nu_coeff * 0.5 * sum(du.^2 .* R_gpu, 1);
        
        % Term 2: u_nom'R*δu (cross term)
        c_ctrl_2 = sum((u_nom .* R_gpu) .* du, 1);
        
        % Term 3: (1/2)u_nom'R*u_nom
        c_ctrl_3 = 0.5 * sum(u_nom.^2 .* R_gpu, 1);
        
        c_ctrl = c_ctrl_1 + c_ctrl_2 + c_ctrl_3;
        
        % Total running cost
        S = S + (c_state + c_ctrl) * dt;
    end
    
    % ===== TERMINAL COST =====
    pos_err_T = X(1:3, :) - pos_des_gpu;
    c_term = w_term * (sum(pos_err_T.^2, 1) + (1 - X(7,:).^2) + 0.5*sum(X(4:6,:).^2, 1));
    S = S + c_term;
    
    S_all(1, i1:i2) = S;
end

%% ========== COMPUTE WEIGHTS ==========
% w(k) = exp(-S̃(k)/λ) / Σ exp(-S̃/λ)
S_min = min(S_all);
weights_all(:) = exp(-(S_all - S_min) / lambda);
w_sum = sum(weights_all);
if w_sum > 1e-10
    weights_all(:) = weights_all / w_sum;
else
    weights_all(:) = single(1/K);
end

%% ========== WEIGHTED CONTROL UPDATE ==========
% u* = u + Σ w(k) * δu(k)
w_3d = reshape(weights_all, 1, 1, K);
dU_weighted = sum(noise_all .* w_3d, 3);

u_seq_new = u_seq_gpu + dU_weighted;
u_seq_new = max(min(u_seq_new, omega_max_g), omega_min_g);

%% ========== OUTPUT ==========
u_seq_cpu = double(gather(u_seq_new));
u_opt = u_seq_cpu(:, 1);

% Shift sequence for next iteration (warm start)
mppi_state.u_seq(:, 1:N-1) = u_seq_cpu(:, 2:N);
mppi_state.u_seq(:, N) = u_seq_cpu(:, N);

end
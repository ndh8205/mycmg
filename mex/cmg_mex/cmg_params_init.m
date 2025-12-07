function params = cmg_params_init()
% cmg_params_init: CMG 시스템 파라미터

%% Body inertia (대각)
params.J = diag([3.431, 1.265, 4.494]);  % [kg·m²]
params.J_inv = diag([1/3.431, 1/1.265, 1/4.494]);

%% CMG
params.h1 = 0.25;  % [Nms] 휠 각운동량
params.h2 = 0.25;

%% Gimbal limits
params.gimbal_rate_max = 2.0;  % [rad/s]

%% Simulation
params.dt = 0.01;  % [s]

end
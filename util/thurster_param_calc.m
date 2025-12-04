clear all
close all
clc

%% --- 1. 물리 상수 ---
params.physical.g = 9.81;            % 중력 가속도 [m/s^2]
params.physical.rho = 1.225;         % 공기 밀도 (해수면 기준) [kg/m^3]

%% --- 2. 배터리 설정 (6S LiPo) ---
params.battery.n_cells = 6;          % 배터리 셀 수 (6S)
params.battery.v_cell_nominal = 3.7; % 셀당 공칭 전압 [V] (완충 시 4.2V 사용 가능)
params.battery.v_cell_max = 4.2;     % 셀당 최대 전압 [V]

% 전체 배터리 전압 계산 (시뮬레이션 기준 전압)
params.battery.voltage = params.battery.n_cells * params.battery.v_cell_nominal; % 22.2V

%% --- 3. 모터 및 ESC 설정 ---
params.motor.KV = 335;               % 모터 KV [RPM/V]
params.motor.efficiency = 0.90;      % 모터+ESC 효율 (부하 시 RPM 감소 고려, 보통 0.8~0.9)

% 최대 회전수 계산 (rad/s 변환)
% 이론상 Max RPM = KV * Voltage * Efficiency
rpm_max = params.motor.KV * params.battery.voltage * params.motor.efficiency;
params.motor.max_omega = rpm_max * (2 * pi / 60); % [rad/s]

%% --- 4. 프로펠러 설정 (9~11인치) ---
params.prop.diameter_inch = 17;      % 프로펠러 지름 [inch] (9, 10, 11 변경)
params.prop.diameter_m = params.prop.diameter_inch * 0.0254; % [m]로 변환

% 추력 상수 계수 (Thrust Coefficient, C_T)
% 일반적 형상: 0.10 ~ 0.12, 고효율 프롭: 0.13 ~ 0.15
params.prop.CT = 0.12; 

% 토크 상수 계수 (Drag/Moment Coefficient, C_M) - Yaw 제어용
% 보통 CT의 1/10 ~ 1/20 수준이나 프롭 형상에 따라 다름. APC 기준 약 0.05 근사
params.prop.CM = 0.05; 

%% --- 5. 최종 추력/토크 계수 계산 (자동 계산됨) ---
% 1) 추력 계수 k_T [N / (rad/s)^2]
% 공식: T = k_T * omega^2  <==>  T = CT * rho * n^2 * D^4
params.drone.motor.k_T = (params.prop.CT * params.physical.rho * params.prop.diameter_m^4) / (4 * pi^2);

% 2) 토크 계수 k_Q (또는 d) [N*m / (rad/s)^2]
% 모터가 돌 때 반작용 토크: Q = k_Q * omega^2
params.drone.motor.k_Q = (params.prop.CM * params.physical.rho * params.prop.diameter_m^5) / (4 * pi^2);


%% --- 결과 확인용 출력 ---
fprintf('--- Drone Configuration ---\n');
fprintf('Battery: %dS (%.1f V)\n', params.battery.n_cells, params.battery.voltage);
fprintf('Motor: %d KV, Max Omega: %.1f rad/s (approx %d RPM)\n', ...
    params.motor.KV, params.motor.max_omega, round(rpm_max));
fprintf('Propeller: %d inch\n', params.prop.diameter_inch);
fprintf('Calculated k_T: %.2e [N/(rad/s)^2]\n', params.drone.motor.k_T);
fprintf('Calculated k_Q: %.2e [Nm/(rad/s)^2]\n', params.drone.motor.k_Q);
fprintf('Calculated k_T: %.9f [N/(rad/s)^2]\n', params.drone.motor.k_T);
fprintf('Calculated k_Q: %.9f [Nm/(rad/s)^2]\n', params.drone.motor.k_Q);
fprintf('Est. Max Thrust per Motor: %.2f N (%.2f kg)\n', ...
    params.drone.motor.k_T * params.motor.max_omega^2, ...
    (params.drone.motor.k_T * params.motor.max_omega^2)/9.81);


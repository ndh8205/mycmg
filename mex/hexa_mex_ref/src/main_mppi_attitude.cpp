#include "../include/types.hpp"
#include "../include/params.hpp"
#include "../include/math_utils.hpp"
#include "../include/mppi_controller.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

// External functions
void control_allocator_inverse(const float* cmd_vec, float* omega_sq, const Params& params);
void srk4(State28& state, const float* u_cmd, const Params& params, float dt);

int main(int argc, char* argv[]) {
    std::cout << "====================================================\n";
    std::cout << "  Hexarotor MPPI Attitude Control\n";
    std::cout << "====================================================\n";

    // Parameters
    Params params;
    params.init_hexa();

    // Simulation settings
    float sim_hz = 1000.0f;  // Simulation rate
    float dt_sim = 1.0f / sim_hz;
    float t_end = 10.0f;
    int N_sim = (int)(t_end / dt_sim) + 1;

    float omega_bar2RPM = 60.0f / (2.0f * M_PI);

    // MPPI parameters (자세 제어 전용 - 논문 기반 개선 설정)
    MPPIParams mppi_params;
    mppi_params.K = 4096;          // 샘플 수 증가 (논문 권장: >4000)
    mppi_params.N = 40;            // Horizon: 40 steps (0.8초)
    mppi_params.dt = 0.02f;        // 20Hz control rate
    mppi_params.lambda = 10.0f;    // Temperature parameter (논문 권장: 10~20)

    // Exploration noise - 균형잡힌 값 (작지만 충분한 탐색)
    mppi_params.sigma_thrust = 1.0f;   // 1.0N (호버 추력 60N 대비 ~1.7%)
    mppi_params.sigma_tau = 0.2f;      // 0.2Nm

    mppi_params.thrust_min = 0.0f;
    mppi_params.thrust_max = 150.0f;
    mppi_params.tau_max = 5.0f;

    // Cost function parameters (자세 제어 집중 + 제어 스무딩)
    // 주의: quaternion error는 radian 단위! (20deg ≈ 0.35rad)
    // 주의: Control cost는 매우 작게! (thrust~60N, tau~1Nm)
    MPPICostParams cost_params;
    cost_params.w_pos_xy = 0.0f;        // XY 위치 무시 (자세 제어만)
    cost_params.w_pos_z = 10.0f;        // 고도 (10m 오차 시 cost=1000)
    cost_params.w_vel = 1.0f;           // 속도 (1m/s 오차 시 cost=1)
    cost_params.w_att_roll = 1000.0f;   // Roll 자세! (0.35rad 오차 시 cost=122)
    cost_params.w_att_pitch = 1000.0f;  // Pitch 자세! (0.35rad 오차 시 cost=122)
    cost_params.w_att_yaw = 100.0f;     // Yaw (덜 중요)
    cost_params.w_omega = 10.0f;        // 각속도 (1rad/s 오차 시 cost=10)
    cost_params.R_thrust = 0.001f;      // Control cost! (60N 시 cost=1.8)
    cost_params.R_tau = 0.0001f;        // Control cost! (1Nm 시 cost=0.00005)
    cost_params.R_du_thrust = 0.0f;     // Control smoothness 비활성화 (CUDA와 동일하게)
    cost_params.R_du_tau = 0.0f;        // Control smoothness 비활성화 (CUDA와 동일하게)
    cost_params.terminal_factor = 10.0f; // 종단 비용 10배 증폭

    // Create MPPI controller
    std::cout << "\nInitializing MPPI controller...\n";
    MPPIController mppi(mppi_params, cost_params, params);

    // Reference (hovering at 10m altitude, level attitude)
    MPPIReference ref;
    ref.pos_des = Vector3(0, 0, -10.0f);  // 10m altitude (NED)
    ref.vel_des = Vector3(0, 0, 0);
    ref.q_des = Quaternion(1, 0, 0, 0);   // Level attitude
    ref.omega_des = Vector3(0, 0, 0);

    mppi.set_reference(ref);

    // Initial state: perturbed attitude, same as SMC test
    float m = params.drone.m;
    float g = params.env.g;
    float k_T = params.drone.k_T;
    int n_motor = params.drone.n_motor;

    float omega_hover = sqrtf(m * g / (n_motor * k_T));
    std::cout << "Hover motor speed: " << omega_hover * omega_bar2RPM << " RPM\n";

    // Initial state: same as SMC test (20/15/10 deg)
    float roll0 = deg2rad(20.0f);
    float pitch0 = deg2rad(15.0f);
    float yaw0 = deg2rad(10.0f);
    Quaternion q0 = euler_to_quat(yaw0, pitch0, roll0);

    State28 state;
    state.pos = Vector3(0, 0, -10.0f);  // Start at 10m altitude
    state.vel = Vector3(0, 0, 0);
    state.quat = q0;  // Initial attitude: 20/15/10 deg
    state.omega = Vector3(0, 0, 0);
    for (int i = 0; i < 6; i++) state.omega_m[i] = omega_hover;
    state.bias_gyro = Vector3(0, 0, 0);
    state.bias_accel = Vector3(0, 0, 0);
    state.bias_mag = Vector3(0, 0, 0);

    // CSV files for logging (SMC 호환 형식)
    std::ofstream att_file("attitude_mppi.csv");
    std::ofstream motor_file("motor_mppi.csv");
    std::ofstream control_file("control_mppi.csv");

    // Headers (SMC 형식 맞춤)
    att_file << "time,alt,pos_x,pos_y,roll,pitch,yaw\n";
    motor_file << "time,m1,m2,m3,m4,m5,m6\n";
    control_file << "t,thrust,tau_x,tau_y,tau_z,min_cost,avg_cost\n";

    // Control rate
    float dt_control = mppi_params.dt;
    int control_decimation = (int)(dt_control / dt_sim);
    std::cout << "Control rate: " << 1.0f/dt_control << " Hz (every " << control_decimation << " sim steps)\n";

    // Timer
    auto t_start = std::chrono::high_resolution_clock::now();

    // Current control
    ControlInput u_current(m * g, 0, 0, 0);  // Start with hover thrust

    // MPPI timing statistics
    std::vector<double> mppi_times;
    mppi_times.reserve(N_sim / control_decimation);

    // Log initial state
    Vector3 euler_init = quat_to_euler(state.quat);
    att_file << "0," << -state.pos.z << "," << state.pos.x << "," << state.pos.y << ","
             << rad2deg(euler_init.x) << "," << rad2deg(euler_init.y) << "," << rad2deg(euler_init.z) << "\n";
    motor_file << "0," << state.omega_m[0] * omega_bar2RPM << ","
               << state.omega_m[1] * omega_bar2RPM << ","
               << state.omega_m[2] * omega_bar2RPM << ","
               << state.omega_m[3] * omega_bar2RPM << ","
               << state.omega_m[4] * omega_bar2RPM << ","
               << state.omega_m[5] * omega_bar2RPM << "\n";

    // Simulation loop
    std::cout << "\nRunning MPPI attitude control simulation...\n";
    for (int k = 0; k < N_sim; k++) {
        float t = k * dt_sim;

        // Update MPPI control at lower rate
        if (k % control_decimation == 0) {
            // Measure MPPI computation time
            auto t_mppi_start = std::chrono::high_resolution_clock::now();
            u_current = mppi.compute_control(state, ref, t);
            auto t_mppi_end = std::chrono::high_resolution_clock::now();

            double mppi_time_ms = std::chrono::duration<double, std::milli>(t_mppi_end - t_mppi_start).count();
            mppi_times.push_back(mppi_time_ms);

            if (k % 1000 == 0) {  // Print every 1 second
                Vector3 euler = quat_to_euler(state.quat);
                std::cout << std::fixed << std::setprecision(3)
                         << "t=" << t << "s: "
                         << "alt=" << -state.pos.z << "m, "
                         << "att=[" << rad2deg(euler.x) << ", "
                         << rad2deg(euler.y) << ", "
                         << rad2deg(euler.z) << "]deg, "
                         << "cost=[" << mppi.get_last_min_cost() << ", "
                         << mppi.get_last_avg_cost() << "], "
                         << "mppi_t=" << mppi_time_ms << "ms\n";
            }
        }

        // Control allocation
        float cmd_vec[4];
        u_current.to_array(cmd_vec);
        float omega_sq[6];
        control_allocator_inverse(cmd_vec, omega_sq, params);

        // Motor commands (saturate)
        float omega_cmd[6];
        for (int i = 0; i < 6; i++) {
            float omega = sqrtf(std::max(0.0f, omega_sq[i]));
            omega_cmd[i] = std::max(params.drone.omega_b_min,
                                   std::min(params.drone.omega_b_max, omega));
        }

        // Log state and control (every timestep for compatibility)
        Vector3 euler = quat_to_euler(state.quat);
        att_file << t << "," << -state.pos.z << "," << state.pos.x << "," << state.pos.y << ","
                 << rad2deg(euler.x) << "," << rad2deg(euler.y) << "," << rad2deg(euler.z) << "\n";

        motor_file << t << "," << state.omega_m[0] * omega_bar2RPM << ","
                   << state.omega_m[1] * omega_bar2RPM << ","
                   << state.omega_m[2] * omega_bar2RPM << ","
                   << state.omega_m[3] * omega_bar2RPM << ","
                   << state.omega_m[4] * omega_bar2RPM << ","
                   << state.omega_m[5] * omega_bar2RPM << "\n";

        if (k % control_decimation == 0) {
            control_file << t << ","
                        << u_current.thrust << ","
                        << u_current.tau.x << "," << u_current.tau.y << "," << u_current.tau.z << ","
                        << mppi.get_last_min_cost() << "," << mppi.get_last_avg_cost() << "\n";
        }

        // Integrate dynamics (high rate)
        srk4(state, omega_cmd, params, dt_sim);

        // Ensure quaternion continuity
        Quaternion q_prev = state.quat;
        if (k > 0) {
            state.quat = ensure_quat_cont(state.quat, q_prev);
        }
    }

    // Timer
    auto t_end_chrono = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end_chrono - t_start).count();

    // MPPI timing statistics
    double mppi_avg = 0.0, mppi_min = 1e9, mppi_max = 0.0;
    double mppi_total = 0.0;
    for (double t : mppi_times) {
        mppi_total += t;
        if (t < mppi_min) mppi_min = t;
        if (t > mppi_max) mppi_max = t;
    }
    if (!mppi_times.empty()) {
        mppi_avg = mppi_total / mppi_times.size();
    }

    // Median
    double mppi_median = 0.0;
    if (!mppi_times.empty()) {
        std::vector<double> sorted_times = mppi_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        size_t mid = sorted_times.size() / 2;
        if (sorted_times.size() % 2 == 0) {
            mppi_median = (sorted_times[mid-1] + sorted_times[mid]) / 2.0;
        } else {
            mppi_median = sorted_times[mid];
        }
    }

    std::cout << "\n====================================================\n";
    std::cout << "Simulation complete!\n";
    std::cout << "  Simulation time: " << t_end << " s\n";
    std::cout << "  Total computation time: " << elapsed_ms / 1000.0 << " s\n";
    std::cout << "  Real-time factor: " << t_end / (elapsed_ms / 1000.0) << "x\n";
    std::cout << "====================================================\n";

    std::cout << "\n====================================================\n";
    std::cout << "MPPI Controller Timing Statistics\n";
    std::cout << "====================================================\n";
    std::cout << "  MPPI calls: " << mppi_times.size() << "\n";
    std::cout << "  Average:    " << std::fixed << std::setprecision(2) << mppi_avg << " ms\n";
    std::cout << "  Median:     " << mppi_median << " ms\n";
    std::cout << "  Min:        " << mppi_min << " ms\n";
    std::cout << "  Max:        " << mppi_max << " ms\n";
    std::cout << "  Total MPPI: " << mppi_total / 1000.0 << " s\n";
    std::cout << "  MPPI overhead: " << (mppi_total / elapsed_ms * 100.0) << " %\n";
    std::cout << "  Control frequency: " << 1000.0 / mppi_avg << " Hz (achievable)\n";
    std::cout << "====================================================\n";

    // Final state
    Vector3 euler_final = quat_to_euler(state.quat);
    std::cout << "\nFinal state:\n";
    std::cout << "  Position: [" << state.pos.x << ", " << state.pos.y << ", " << state.pos.z << "] m\n";
    std::cout << "  Altitude: " << -state.pos.z << " m (target: 10.0 m)\n";
    std::cout << "  Euler: [" << rad2deg(euler_final.x) << ", "
              << rad2deg(euler_final.y) << ", "
              << rad2deg(euler_final.z) << "] deg\n";
    std::cout << "  XY drift: " << sqrtf(state.pos.x*state.pos.x + state.pos.y*state.pos.y) << " m\n";

    att_file.close();
    motor_file.close();
    control_file.close();

    std::cout << "\nData saved to:\n";
    std::cout << "  attitude_mppi.csv\n";
    std::cout << "  motor_mppi.csv\n";
    std::cout << "  control_mppi.csv\n";

    return 0;
}

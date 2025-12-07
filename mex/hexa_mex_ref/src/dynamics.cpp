#include "../include/types.hpp"
#include "../include/params.hpp"
#include "../include/math_utils.hpp"
#include "../include/disturbance.hpp"
#include <cstring>

// External functions
void control_allocator_forward(const float* omega_sq, float* cmd_vec, const Params& params);

// Motor dynamics: first-order system with asymmetric time constants
void motor_dynamics(const float* omega_m, const float* u_cmd, const Params& params, float* omega_m_dot) {
    for (int i = 0; i < 6; i++) {
        // Saturate command
        float u_sat = std::max(std::min(u_cmd[i], params.drone.omega_b_max), params.drone.omega_b_min);

        // Choose time constant
        float tau = (u_sat >= omega_m[i]) ? params.drone.tau_up : params.drone.tau_down;

        // First-order dynamics
        omega_m_dot[i] = (u_sat - omega_m[i]) / tau;
    }
}

// Bias dynamics
void bias_dynamics(const Vector3& b_gyro, const Vector3& b_accel, const Vector3& b_mag,
                   const float* w, const Params& params,
                   Vector3& b_gyro_dot, Vector3& b_accel_dot, Vector3& b_mag_dot) {
    // Gyro bias: random walk
    b_gyro_dot = Vector3(w[0], w[1], w[2]);

    // Accel bias: random walk
    b_accel_dot = Vector3(w[3], w[4], w[5]);

    // Mag bias: FOGM (First-Order Gauss-Markov)
    float tau_mag = params.sensor.mag_bias_corr_time;
    b_mag_dot.x = -b_mag.x / tau_mag + w[6];
    b_mag_dot.y = -b_mag.y / tau_mag + w[7];
    b_mag_dot.z = -b_mag.z / tau_mag + w[8];
}

// Full drone dynamics (with disturbance support)
void drone_dynamics(const State28& state, const float* u_cmd, const float* w,
                    float dt, const Params& params, float t, DisturbanceState* dist_state,
                    State28& state_dot) {
    // Extract states
    Vector3 pos = state.pos;
    Vector3 vel_b = state.vel;
    Quaternion quat = state.quat;
    Vector3 omega = state.omega;
    float omega_m[6];
    for (int i = 0; i < 6; i++) omega_m[i] = state.omega_m[i];

    // Normalize quaternion
    quat = quat.normalized();

    // DCM (body to NED)
    Matrix3 R_b2n = quat_to_dcm(quat);
    Matrix3 R_n2b = R_b2n.transpose();

    // Force and moment calculation
    float omega_sq[6];
    for (int i = 0; i < 6; i++) omega_sq[i] = omega_m[i] * omega_m[i];

    float cmd_vec[4];
    control_allocator_forward(omega_sq, cmd_vec, params);

    float T_total = cmd_vec[0];
    Vector3 M_ctrl(cmd_vec[1], cmd_vec[2], cmd_vec[3]);

    // Disturbance forces and torques
    Vector3 tau_dist(0, 0, 0);
    Vector3 F_dist(0, 0, 0);

    if (params.dist.enable && dist_state != nullptr) {
        // Torque disturbance
        tau_dist = dist_torque(t, params, *dist_state);

        // Wind disturbance
        F_dist = dist_wind(vel_b, R_b2n, t, dt, params, *dist_state);
    }

    // Total force (body frame)
    Vector3 F_body(0, 0, -T_total);
    F_body = F_body + F_dist;

    // Total moment (body frame)
    Vector3 M_body = M_ctrl + tau_dist;

    // === Equations of Motion ===

    // Position derivative (NED = R_b2n * vel_body)
    Vector3 pos_dot = R_b2n * vel_b;

    // Velocity derivative (body frame)
    Vector3 gravity_body = R_n2b * Vector3(0, 0, params.env.g);
    Vector3 vel_dot = F_body / params.drone.m + gravity_body - omega.cross(vel_b);

    // Quaternion derivative
    Quaternion quat_dot = quat_derivative(quat, omega);

    // Angular velocity derivative
    // J * omega_dot = M_body - omega × (J * omega)
    Vector3 J_omega;
    J_omega.x = params.drone.J[0] * omega.x;
    J_omega.y = params.drone.J[4] * omega.y;
    J_omega.z = params.drone.J[8] * omega.z;

    Vector3 gyro_torque = omega.cross(J_omega);
    Vector3 total_torque = M_body - gyro_torque;

    Vector3 omega_dot;
    omega_dot.x = total_torque.x / params.drone.J[0];
    omega_dot.y = total_torque.y / params.drone.J[4];
    omega_dot.z = total_torque.z / params.drone.J[8];

    // Motor dynamics
    float omega_m_dot[6];
    motor_dynamics(omega_m, u_cmd, params, omega_m_dot);

    // Bias dynamics
    Vector3 b_gyro_dot, b_accel_dot, b_mag_dot;
    bias_dynamics(state.bias_gyro, state.bias_accel, state.bias_mag, w, params,
                  b_gyro_dot, b_accel_dot, b_mag_dot);

    // Ground collision (simple check)
    if (pos.z >= 0) {
        pos_dot = Vector3(0, 0, 0);
        vel_dot = Vector3(0, 0, 0);
        omega_dot = Vector3(0, 0, 0);
    }

    // Fill output
    state_dot.pos = pos_dot;
    state_dot.vel = vel_dot;
    state_dot.quat = quat_dot;
    state_dot.omega = omega_dot;
    for (int i = 0; i < 6; i++) state_dot.omega_m[i] = omega_m_dot[i];
    state_dot.bias_gyro = b_gyro_dot;
    state_dot.bias_accel = b_accel_dot;
    state_dot.bias_mag = b_mag_dot;
}

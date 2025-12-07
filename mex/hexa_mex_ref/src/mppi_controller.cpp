#include "../include/mppi_controller.hpp"
#include "../include/math_utils.hpp"
#include "../include/disturbance.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

// External functions
void drone_dynamics(const State28& state, const float* u_cmd, const float* w,
                    float dt, const Params& params, float t, DisturbanceState* dist_state,
                    State28& state_dot);
void control_allocator_inverse(const float* cmd_vec, float* omega_sq, const Params& params);

// Helper: quaternion error (returns rotation vector)
Vector3 quaternion_error(const Quaternion& q_des, const Quaternion& q) {
    // Error quaternion: q_err = q_des * q^{-1}
    Quaternion q_conj = q.conjugate();
    Quaternion q_err(
        q_des.w * q_conj.w - q_des.x * q_conj.x - q_des.y * q_conj.y - q_des.z * q_conj.z,
        q_des.w * q_conj.x + q_des.x * q_conj.w + q_des.y * q_conj.z - q_des.z * q_conj.y,
        q_des.w * q_conj.y - q_des.x * q_conj.z + q_des.y * q_conj.w + q_des.z * q_conj.x,
        q_des.w * q_conj.z + q_des.x * q_conj.y - q_des.y * q_conj.x + q_des.z * q_conj.w
    );

    // Small angle approximation: theta ≈ 2 * [qx, qy, qz]
    return Vector3(2.0f * q_err.x, 2.0f * q_err.y, 2.0f * q_err.z);
}

float quaternion_distance(const Quaternion& q1, const Quaternion& q2) {
    Vector3 err = quaternion_error(q1, q2);
    return err.norm();
}

// Helper: Add two states (for RK4)
static State28 state_add(const State28& a, const State28& b, float scale = 1.0f) {
    State28 result;
    result.pos = a.pos + b.pos * scale;
    result.vel = a.vel + b.vel * scale;
    result.quat.w = a.quat.w + b.quat.w * scale;
    result.quat.x = a.quat.x + b.quat.x * scale;
    result.quat.y = a.quat.y + b.quat.y * scale;
    result.quat.z = a.quat.z + b.quat.z * scale;
    result.omega = a.omega + b.omega * scale;
    for (int i = 0; i < 6; i++) result.omega_m[i] = a.omega_m[i] + b.omega_m[i] * scale;
    result.bias_gyro = a.bias_gyro + b.bias_gyro * scale;
    result.bias_accel = a.bias_accel + b.bias_accel * scale;
    result.bias_mag = a.bias_mag + b.bias_mag * scale;
    return result;
}

MPPIController::MPPIController(const MPPIParams& mppi_params,
                               const MPPICostParams& cost_params,
                               const Params& drone_params)
    : mppi_params_(mppi_params),
      cost_params_(cost_params),
      drone_params_(drone_params),
      rng_(std::random_device{}()),
      dist_normal_(0.0f, 1.0f),
      min_cost_(0.0f),
      avg_cost_(0.0f),
      iteration_count_(0)
{
    // Initialize control sequence with hover thrust
    float m = drone_params.drone.m;
    float g = drone_params.env.g;
    float hover_thrust = m * g;

    u_seq_.resize(mppi_params_.N);
    for (int i = 0; i < mppi_params_.N; i++) {
        u_seq_[i] = ControlInput(hover_thrust, 0, 0, 0);
    }

    // Initialize rollout storage
    costs_.resize(mppi_params_.K);
    weights_.resize(mppi_params_.K);
    du_seqs_.resize(mppi_params_.K);
    for (int k = 0; k < mppi_params_.K; k++) {
        du_seqs_[k].resize(mppi_params_.N);
    }

    std::cout << "MPPI Controller initialized:\n";
    std::cout << "  K = " << mppi_params_.K << " rollouts\n";
    std::cout << "  N = " << mppi_params_.N << " horizon\n";
    std::cout << "  dt = " << mppi_params_.dt << " s\n";
    std::cout << "  lambda = " << mppi_params_.lambda << "\n";
}

MPPIController::~MPPIController() {
}

void MPPIController::set_reference(const MPPIReference& ref) {
    ref_ = ref;
}

void MPPIController::set_mppi_params(const MPPIParams& params) {
    mppi_params_ = params;
    // Resize storage if needed
    if ((int)u_seq_.size() != mppi_params_.N) {
        u_seq_.resize(mppi_params_.N);
    }
    if ((int)costs_.size() != mppi_params_.K) {
        costs_.resize(mppi_params_.K);
        weights_.resize(mppi_params_.K);
        du_seqs_.resize(mppi_params_.K);
        for (int k = 0; k < mppi_params_.K; k++) {
            du_seqs_[k].resize(mppi_params_.N);
        }
    }
}

void MPPIController::set_cost_params(const MPPICostParams& params) {
    cost_params_ = params;
}

ControlInput MPPIController::compute_control(const State28& state,
                                             const MPPIReference& ref,
                                             float t) {
    // Update reference
    ref_ = ref;

    // Run MPPI iteration
    mppi_iteration(state, t);

    // Return first control input
    iteration_count_++;
    return u_seq_[0];
}

void MPPIController::mppi_iteration(const State28& state, float t) {
    // Step 1: Shift control sequence first (MPC receding horizon warm start)
    // This provides a good initial guess for the next iteration
    for (int i = 0; i < mppi_params_.N - 1; i++) {
        u_seq_[i] = u_seq_[i + 1];
    }
    // Last control: use hover thrust
    float m = drone_params_.drone.m;
    float g = drone_params_.env.g;
    u_seq_[mppi_params_.N - 1] = ControlInput(m * g, 0, 0, 0);

    // Step 2: Generate random control variations for all K rollouts
    for (int k = 0; k < mppi_params_.K; k++) {
        for (int i = 0; i < mppi_params_.N; i++) {
            du_seqs_[k][i] = sample_noise();
        }
    }

    // Step 3: Simulate K rollouts and compute costs
    #pragma omp parallel for
    for (int k = 0; k < mppi_params_.K; k++) {
        costs_[k] = simulate_rollout(state, u_seq_, du_seqs_[k], t);
    }

    // Step 4: Compute importance weights
    // Find minimum cost for numerical stability
    float min_cost = *std::min_element(costs_.begin(), costs_.end());

    min_cost_ = min_cost;
    avg_cost_ = std::accumulate(costs_.begin(), costs_.end(), 0.0f) / mppi_params_.K;

    // Compute weights: w_k = exp(-1/lambda * (S_k - min_S))
    float weight_sum = 0.0f;
    for (int k = 0; k < mppi_params_.K; k++) {
        weights_[k] = expf(-(costs_[k] - min_cost) / mppi_params_.lambda);
        weight_sum += weights_[k];
    }

    // Normalize weights
    if (weight_sum > 1e-8f) {
        for (int k = 0; k < mppi_params_.K; k++) {
            weights_[k] /= weight_sum;
        }
    } else {
        // Fallback: uniform weights
        for (int k = 0; k < mppi_params_.K; k++) {
            weights_[k] = 1.0f / mppi_params_.K;
        }
    }

    // Step 5: Update control sequence using weighted average
    for (int i = 0; i < mppi_params_.N; i++) {
        ControlInput du_weighted(0, 0, 0, 0);

        for (int k = 0; k < mppi_params_.K; k++) {
            du_weighted = du_weighted + du_seqs_[k][i] * weights_[k];
        }

        u_seq_[i] = u_seq_[i] + du_weighted;
        clamp_control(u_seq_[i]);
    }
}

float MPPIController::simulate_rollout(const State28& state_init,
                                       const std::vector<ControlInput>& u_seq,
                                       const std::vector<ControlInput>& du_seq,
                                       float t) {
    State28 state = state_init;
    float total_cost = 0.0f;

    // Hover thrust as initial previous control
    float m = drone_params_.drone.m;
    float g = drone_params_.env.g;
    ControlInput u_prev(m * g, 0, 0, 0);

    // Simulate forward for N steps
    for (int i = 0; i < mppi_params_.N; i++) {
        // Perturbed control
        ControlInput u = u_seq[i] + du_seq[i];
        clamp_control(u);

        // Compute running cost with control rate penalty
        bool is_terminal = (i == mppi_params_.N - 1);
        float cost = compute_running_cost(state, u, u_prev, ref_, is_terminal);
        total_cost += cost;

        // Integrate dynamics
        integrate_step(state, u, mppi_params_.dt, t + i * mppi_params_.dt);

        // Update previous control
        u_prev = u;
    }

    return total_cost;
}

float MPPIController::compute_running_cost(const State28& state,
                                           const ControlInput& u,
                                           const ControlInput& u_prev,
                                           const MPPIReference& ref,
                                           bool is_terminal) {
    float cost = 0.0f;

    // Position error (NED frame)
    Vector3 pos_err = state.pos - ref.pos_des;
    cost += cost_params_.w_pos_xy * (pos_err.x * pos_err.x + pos_err.y * pos_err.y);
    cost += cost_params_.w_pos_z * (pos_err.z * pos_err.z);

    // Velocity cost (body frame)
    Vector3 vel_ned = quat_to_dcm(state.quat) * state.vel;
    Vector3 vel_err = vel_ned - ref.vel_des;
    cost += cost_params_.w_vel * vel_err.norm2();

    // Attitude error
    Vector3 att_err = quaternion_error(ref.q_des, state.quat);
    cost += cost_params_.w_att_roll * att_err.x * att_err.x;
    cost += cost_params_.w_att_pitch * att_err.y * att_err.y;
    cost += cost_params_.w_att_yaw * att_err.z * att_err.z;

    // Angular velocity cost
    Vector3 omega_err = state.omega - ref.omega_des;
    cost += cost_params_.w_omega * omega_err.norm2();

    // Control cost: (1/2) * u^T * R * u
    if (!is_terminal) {
        cost += 0.5f * cost_params_.R_thrust * u.thrust * u.thrust;
        cost += 0.5f * cost_params_.R_tau * u.tau.norm2();

        // Control rate cost (smoothness penalty): (1/2) * du^T * R_du * du
        float du_thrust = u.thrust - u_prev.thrust;
        Vector3 du_tau = u.tau - u_prev.tau;
        cost += 0.5f * cost_params_.R_du_thrust * du_thrust * du_thrust;
        cost += 0.5f * cost_params_.R_du_tau * du_tau.norm2();
    }

    // Boundary costs (optional)
    if (cost_params_.enable_boundaries) {
        if (state.pos.x < cost_params_.pos_min.x || state.pos.x > cost_params_.pos_max.x ||
            state.pos.y < cost_params_.pos_min.y || state.pos.y > cost_params_.pos_max.y ||
            state.pos.z < cost_params_.pos_min.z || state.pos.z > cost_params_.pos_max.z) {
            cost += cost_params_.boundary_cost;
        }
    }

    // Terminal cost multiplier
    if (is_terminal) {
        cost *= cost_params_.terminal_factor;
    }

    return cost;
}

void MPPIController::clamp_control(ControlInput& u) {
    // Clamp thrust
    u.thrust = std::max(mppi_params_.thrust_min,
                       std::min(mppi_params_.thrust_max, u.thrust));

    // Clamp torque magnitude
    float tau_mag = u.tau.norm();
    if (tau_mag > mppi_params_.tau_max) {
        u.tau = u.tau * (mppi_params_.tau_max / tau_mag);
    }
}

ControlInput MPPIController::sample_noise() {
    float thrust_noise = mppi_params_.sigma_thrust * dist_normal_(rng_);
    float tau_x_noise = mppi_params_.sigma_tau * dist_normal_(rng_);
    float tau_y_noise = mppi_params_.sigma_tau * dist_normal_(rng_);
    float tau_z_noise = mppi_params_.sigma_tau * dist_normal_(rng_);

    return ControlInput(thrust_noise, tau_x_noise, tau_y_noise, tau_z_noise);
}

void MPPIController::integrate_step(State28& state, const ControlInput& u,
                                    float dt, float t) {
    // Convert high-level control [thrust, tau] to motor commands
    float cmd_vec[4];
    u.to_array(cmd_vec);

    float omega_sq[6];
    control_allocator_inverse(cmd_vec, omega_sq, drone_params_);

    // Convert to motor speeds and saturate
    float omega_cmd[6];
    for (int i = 0; i < 6; i++) {
        float omega = sqrtf(std::max(0.0f, omega_sq[i]));
        omega_cmd[i] = std::max(drone_params_.drone.omega_b_min,
                               std::min(drone_params_.drone.omega_b_max, omega));
    }

    // Zero process noise for rollout prediction (deterministic)
    float w[9] = {0};

    // RK4 integration
    State28 k1, k2, k3, k4;

    // k1
    drone_dynamics(state, omega_cmd, w, dt, drone_params_, t, nullptr, k1);

    // k2
    State28 state_k2 = state_add(state, k1, 0.5f * dt);
    drone_dynamics(state_k2, omega_cmd, w, dt, drone_params_, t, nullptr, k2);

    // k3
    State28 state_k3 = state_add(state, k2, 0.5f * dt);
    drone_dynamics(state_k3, omega_cmd, w, dt, drone_params_, t, nullptr, k3);

    // k4
    State28 state_k4 = state_add(state, k3, dt);
    drone_dynamics(state_k4, omega_cmd, w, dt, drone_params_, t, nullptr, k4);

    // Update state: x = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    state.pos = state.pos + (k1.pos + k2.pos * 2.0f + k3.pos * 2.0f + k4.pos) * (dt / 6.0f);
    state.vel = state.vel + (k1.vel + k2.vel * 2.0f + k3.vel * 2.0f + k4.vel) * (dt / 6.0f);

    state.quat.w += (k1.quat.w + k2.quat.w * 2.0f + k3.quat.w * 2.0f + k4.quat.w) * (dt / 6.0f);
    state.quat.x += (k1.quat.x + k2.quat.x * 2.0f + k3.quat.x * 2.0f + k4.quat.x) * (dt / 6.0f);
    state.quat.y += (k1.quat.y + k2.quat.y * 2.0f + k3.quat.y * 2.0f + k4.quat.y) * (dt / 6.0f);
    state.quat.z += (k1.quat.z + k2.quat.z * 2.0f + k3.quat.z * 2.0f + k4.quat.z) * (dt / 6.0f);
    state.quat = state.quat.normalized();  // Normalize quaternion

    state.omega = state.omega + (k1.omega + k2.omega * 2.0f + k3.omega * 2.0f + k4.omega) * (dt / 6.0f);

    for (int i = 0; i < 6; i++) {
        state.omega_m[i] += (k1.omega_m[i] + k2.omega_m[i] * 2.0f + k3.omega_m[i] * 2.0f + k4.omega_m[i]) * (dt / 6.0f);
    }

    state.bias_gyro = state.bias_gyro + (k1.bias_gyro + k2.bias_gyro * 2.0f + k3.bias_gyro * 2.0f + k4.bias_gyro) * (dt / 6.0f);
    state.bias_accel = state.bias_accel + (k1.bias_accel + k2.bias_accel * 2.0f + k3.bias_accel * 2.0f + k4.bias_accel) * (dt / 6.0f);
    state.bias_mag = state.bias_mag + (k1.bias_mag + k2.bias_mag * 2.0f + k3.bias_mag * 2.0f + k4.bias_mag) * (dt / 6.0f);
}

#include "../include/types.hpp"
#include "../include/params.hpp"
#include "../include/math_utils.hpp"
#include "../include/disturbance.hpp"
#include <random>
#include <cmath>

// External function
void drone_dynamics(const State28& state, const float* u_cmd, const float* w,
                    float dt, const Params& params, float t, DisturbanceState* dist_state,
                    State28& state_dot);

// Random number generator
static std::mt19937 rng(42);  // Fixed seed for reproducibility
static std::normal_distribution<float> normal_dist(0.0f, 1.0f);

// Helper: Add two states (for RK4)
State28 state_add(const State28& a, const State28& b, float scale = 1.0f) {
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

// Stochastic Runge-Kutta 4th order integrator
void srk4(State28& state, const float* u_cmd, const Params& params, float dt) {
    // Process noise covariance
    float Q_diag[9] = {
        params.sensor.gyro_random_walk * params.sensor.gyro_random_walk,
        params.sensor.gyro_random_walk * params.sensor.gyro_random_walk,
        params.sensor.gyro_random_walk * params.sensor.gyro_random_walk,
        params.sensor.accel_random_walk * params.sensor.accel_random_walk,
        params.sensor.accel_random_walk * params.sensor.accel_random_walk,
        params.sensor.accel_random_walk * params.sensor.accel_random_walk,
        params.sensor.mag_random_walk * params.sensor.mag_random_walk,
        params.sensor.mag_random_walk * params.sensor.mag_random_walk,
        params.sensor.mag_random_walk * params.sensor.mag_random_walk
    };

    // Scaling for stochastic integration
    float alpha[4] = {1.0f/6.0f, 2.0f/6.0f, 2.0f/6.0f, 1.0f/6.0f};
    float beta = 1.0f / (alpha[0] + alpha[1] + alpha[2] + alpha[3]);

    float scaled_Q[9];
    for (int i = 0; i < 9; i++) {
        scaled_Q[i] = beta * Q_diag[i] / dt;
    }

    // Generate random noise vectors
    float w1[9], w2[9], w3[9], w4[9];
    for (int i = 0; i < 9; i++) {
        w1[i] = sqrtf(scaled_Q[i]) * normal_dist(rng);
        w2[i] = sqrtf(scaled_Q[i]) * normal_dist(rng);
        w3[i] = sqrtf(scaled_Q[i]) * normal_dist(rng);
        w4[i] = sqrtf(scaled_Q[i]) * normal_dist(rng);
    }

    // RK4 stages
    State28 k1, k2, k3, k4;

    // k1 (no disturbance for compatibility)
    drone_dynamics(state, u_cmd, w1, dt, params, 0, nullptr, k1);

    // k2
    State28 state_k2 = state_add(state, k1, 0.5f * dt);
    drone_dynamics(state_k2, u_cmd, w2, dt, params, 0, nullptr, k2);

    // k3
    State28 state_k3 = state_add(state, k2, 0.5f * dt);
    drone_dynamics(state_k3, u_cmd, w3, dt, params, 0, nullptr, k3);

    // k4
    State28 state_k4 = state_add(state, k3, dt);
    drone_dynamics(state_k4, u_cmd, w4, dt, params, 0, nullptr, k4);

    // Update state: x = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    state.pos += (k1.pos + k2.pos * 2.0f + k3.pos * 2.0f + k4.pos) * (dt / 6.0f);
    state.vel += (k1.vel + k2.vel * 2.0f + k3.vel * 2.0f + k4.vel) * (dt / 6.0f);

    state.quat.w += (k1.quat.w + k2.quat.w * 2.0f + k3.quat.w * 2.0f + k4.quat.w) * (dt / 6.0f);
    state.quat.x += (k1.quat.x + k2.quat.x * 2.0f + k3.quat.x * 2.0f + k4.quat.x) * (dt / 6.0f);
    state.quat.y += (k1.quat.y + k2.quat.y * 2.0f + k3.quat.y * 2.0f + k4.quat.y) * (dt / 6.0f);
    state.quat.z += (k1.quat.z + k2.quat.z * 2.0f + k3.quat.z * 2.0f + k4.quat.z) * (dt / 6.0f);

    state.omega += (k1.omega + k2.omega * 2.0f + k3.omega * 2.0f + k4.omega) * (dt / 6.0f);

    for (int i = 0; i < 6; i++) {
        state.omega_m[i] += (k1.omega_m[i] + k2.omega_m[i] * 2.0f + k3.omega_m[i] * 2.0f + k4.omega_m[i]) * (dt / 6.0f);
    }

    state.bias_gyro += (k1.bias_gyro + k2.bias_gyro * 2.0f + k3.bias_gyro * 2.0f + k4.bias_gyro) * (dt / 6.0f);
    state.bias_accel += (k1.bias_accel + k2.bias_accel * 2.0f + k3.bias_accel * 2.0f + k4.bias_accel) * (dt / 6.0f);
    state.bias_mag += (k1.bias_mag + k2.bias_mag * 2.0f + k3.bias_mag * 2.0f + k4.bias_mag) * (dt / 6.0f);

    // Normalize quaternion
    state.quat = state.quat.normalized();
}

// Stochastic Runge-Kutta 4th order integrator (with disturbance)
void srk4_dist(State28& state, const float* u_cmd, const Params& params, float dt, float t,
               DisturbanceState& dist_state) {
    // Process noise covariance
    float Q_diag[9] = {
        params.sensor.gyro_random_walk * params.sensor.gyro_random_walk,
        params.sensor.gyro_random_walk * params.sensor.gyro_random_walk,
        params.sensor.gyro_random_walk * params.sensor.gyro_random_walk,
        params.sensor.accel_random_walk * params.sensor.accel_random_walk,
        params.sensor.accel_random_walk * params.sensor.accel_random_walk,
        params.sensor.accel_random_walk * params.sensor.accel_random_walk,
        params.sensor.mag_random_walk * params.sensor.mag_random_walk,
        params.sensor.mag_random_walk * params.sensor.mag_random_walk,
        params.sensor.mag_random_walk * params.sensor.mag_random_walk
    };

    // Scaling for stochastic integration
    float alpha[4] = {1.0f/6.0f, 2.0f/6.0f, 2.0f/6.0f, 1.0f/6.0f};
    float beta = 1.0f / (alpha[0] + alpha[1] + alpha[2] + alpha[3]);

    float scaled_Q[9];
    for (int i = 0; i < 9; i++) {
        scaled_Q[i] = beta * Q_diag[i] / dt;
    }

    // Generate random noise vectors
    float w1[9], w2[9], w3[9], w4[9];
    for (int i = 0; i < 9; i++) {
        w1[i] = sqrtf(scaled_Q[i]) * normal_dist(rng);
        w2[i] = sqrtf(scaled_Q[i]) * normal_dist(rng);
        w3[i] = sqrtf(scaled_Q[i]) * normal_dist(rng);
        w4[i] = sqrtf(scaled_Q[i]) * normal_dist(rng);
    }

    // RK4 stages
    State28 k1, k2, k3, k4;

    // k1
    drone_dynamics(state, u_cmd, w1, dt, params, t, &dist_state, k1);

    // k2
    State28 state_k2 = state_add(state, k1, 0.5f * dt);
    drone_dynamics(state_k2, u_cmd, w2, dt, params, t + 0.5f*dt, &dist_state, k2);

    // k3
    State28 state_k3 = state_add(state, k2, 0.5f * dt);
    drone_dynamics(state_k3, u_cmd, w3, dt, params, t + 0.5f*dt, &dist_state, k3);

    // k4
    State28 state_k4 = state_add(state, k3, dt);
    drone_dynamics(state_k4, u_cmd, w4, dt, params, t + dt, &dist_state, k4);

    // Update state: x = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    state.pos += (k1.pos + k2.pos * 2.0f + k3.pos * 2.0f + k4.pos) * (dt / 6.0f);
    state.vel += (k1.vel + k2.vel * 2.0f + k3.vel * 2.0f + k4.vel) * (dt / 6.0f);

    state.quat.w += (k1.quat.w + k2.quat.w * 2.0f + k3.quat.w * 2.0f + k4.quat.w) * (dt / 6.0f);
    state.quat.x += (k1.quat.x + k2.quat.x * 2.0f + k3.quat.x * 2.0f + k4.quat.x) * (dt / 6.0f);
    state.quat.y += (k1.quat.y + k2.quat.y * 2.0f + k3.quat.y * 2.0f + k4.quat.y) * (dt / 6.0f);
    state.quat.z += (k1.quat.z + k2.quat.z * 2.0f + k3.quat.z * 2.0f + k4.quat.z) * (dt / 6.0f);

    state.omega += (k1.omega + k2.omega * 2.0f + k3.omega * 2.0f + k4.omega) * (dt / 6.0f);

    for (int i = 0; i < 6; i++) {
        state.omega_m[i] += (k1.omega_m[i] + k2.omega_m[i] * 2.0f + k3.omega_m[i] * 2.0f + k4.omega_m[i]) * (dt / 6.0f);
    }

    state.bias_gyro += (k1.bias_gyro + k2.bias_gyro * 2.0f + k3.bias_gyro * 2.0f + k4.bias_gyro) * (dt / 6.0f);
    state.bias_accel += (k1.bias_accel + k2.bias_accel * 2.0f + k3.bias_accel * 2.0f + k4.bias_accel) * (dt / 6.0f);
    state.bias_mag += (k1.bias_mag + k2.bias_mag * 2.0f + k3.bias_mag * 2.0f + k4.bias_mag) * (dt / 6.0f);

    // Normalize quaternion
    state.quat = state.quat.normalized();
}

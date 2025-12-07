#include "../include/disturbance.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

// Dryden state-space matrices
void dryden_matrices(float h, const char* intensity, float A[6][6], float B[6], float C[3][6]) {
    // Turbulence intensity based on preset
    float sigma_u, sigma_v, sigma_w;
    if (strcmp(intensity, "light") == 0) {
        sigma_u = 1.5f; sigma_v = 1.5f; sigma_w = 1.0f;
    } else if (strcmp(intensity, "moderate") == 0) {
        sigma_u = 3.0f; sigma_v = 3.0f; sigma_w = 2.0f;
    } else if (strcmp(intensity, "severe") == 0) {
        sigma_u = 6.0f; sigma_v = 6.0f; sigma_w = 4.0f;
    } else {
        sigma_u = 1.5f; sigma_v = 1.5f; sigma_w = 1.0f;
    }

    // Scale lengths (low altitude approximation)
    float L_u = h / powf(0.177f + 0.000823f * h, 1.2f);
    float L_v = L_u;
    float L_w = h;

    // Airspeed (nominal)
    float V = 10.0f;  // [m/s]

    // Time constants
    float tau_u = L_u / V;
    float tau_v = L_v / V;
    float tau_w = L_w / V;

    // First-order approximation for each axis
    // A matrix (diagonal)
    memset(A, 0, sizeof(float) * 36);
    A[0][0] = -1.0f / tau_u;
    A[1][1] = -1.0f / tau_u;
    A[2][2] = -1.0f / tau_v;
    A[3][3] = -1.0f / tau_v;
    A[4][4] = -1.0f / tau_w;
    A[5][5] = -1.0f / tau_w;

    // B vector
    B[0] = sqrtf(2.0f / tau_u) * sigma_u;
    B[1] = 0;
    B[2] = sqrtf(2.0f / tau_v) * sigma_v;
    B[3] = 0;
    B[4] = sqrtf(2.0f / tau_w) * sigma_w;
    B[5] = 0;

    // C matrix
    memset(C, 0, sizeof(float) * 18);
    C[0][0] = 1;
    C[1][2] = 1;
    C[2][4] = 1;
}

// Initialize disturbance
void dist_init(Params& params, const char* preset, DisturbanceState& dist_state) {
    // Apply presets
    if (strcmp(preset, "nominal") == 0) {
        params.set_disturbance_preset("nominal");
    } else if (strcmp(preset, "level1") == 0) {
        params.set_disturbance_preset("level1");
    } else if (strcmp(preset, "level2") == 0) {
        params.set_disturbance_preset("level2");
    } else if (strcmp(preset, "level3") == 0) {
        params.set_disturbance_preset("level3");
    }

    // Initialize disturbance state
    dist_state.t_prev = 0;

    // Random sine state (pre-generated phases and frequencies)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            dist_state.random_sine.phase[i][j] = 2.0f * M_PI * dist_state.uniform_dist(dist_state.rng);
            dist_state.random_sine.freq_mult[i][j] = 0.5f + dist_state.uniform_dist(dist_state.rng);
        }
    }

    // Paper-style disturbance state
    dist_state.paper.integral = Vector3(0, 0, 0);
    for (int i = 0; i < 3; i++) {
        float amp_sum = 0;
        for (int j = 0; j < 10; j++) {
            dist_state.paper.freq[i][j] = 0.1f + 2.0f * dist_state.uniform_dist(dist_state.rng);
            dist_state.paper.phase[i][j] = 2.0f * M_PI * dist_state.uniform_dist(dist_state.rng);
            dist_state.paper.amp[i][j] = dist_state.uniform_dist(dist_state.rng);
            amp_sum += dist_state.paper.amp[i][j];
        }
        // Normalize amplitudes
        for (int j = 0; j < 10; j++) {
            dist_state.paper.amp[i][j] /= amp_sum;
        }
    }

    // Dryden filter state
    for (int i = 0; i < 6; i++) {
        dist_state.dryden.x[i] = 0;
    }

    const char* dryden_intensity = "light";
    if (params.dist.enable) {
        if (params.dist.wind_type == DisturbanceParams::DRYDEN) {
            // Determine intensity from wind settings (simplified)
            dryden_intensity = "moderate";
        }
    }

    dryden_matrices(params.dist.aero_A, dryden_intensity,
                   dist_state.dryden.A, dist_state.dryden.B, dist_state.dryden.C);

    // Gust state
    dist_state.gust.active = false;
    dist_state.gust.direction = Vector3(1, 0, 0);
}

// Torque disturbance
Vector3 dist_torque(float t, const Params& params, DisturbanceState& dist_state) {
    Vector3 tau_dist(0, 0, 0);

    if (!params.dist.enable) {
        return tau_dist;
    }

    Vector3 mag = params.dist.torque_magnitude;

    switch (params.dist.torque_type) {
        case DisturbanceParams::NONE:
            break;

        case DisturbanceParams::SINE: {
            float f = params.dist.torque_freq;
            float val = sinf(2.0f * M_PI * f * t);
            tau_dist = mag * val;
            break;
        }

        case DisturbanceParams::RANDOM_SINE: {
            float f_base = params.dist.torque_freq;
            for (int i = 0; i < 3; i++) {
                float sum = 0;
                for (int j = 0; j < 5; j++) {
                    float f_j = f_base * dist_state.random_sine.freq_mult[i][j] * (j + 1);
                    float phi_j = dist_state.random_sine.phase[i][j];
                    sum += sinf(2.0f * M_PI * f_j * t + phi_j) / (j + 1);
                }
                tau_dist.data()[i] = mag.data()[i] * sum / 3.0f;
            }
            break;
        }

        case DisturbanceParams::STEP: {
            if (t >= params.dist.gust_start && t < params.dist.gust_start + params.dist.gust_duration) {
                tau_dist = mag;
            }
            break;
        }

        case DisturbanceParams::IMPULSE: {
            float t_imp = 3.0f;  // Fixed impulse time
            float t_dur = 0.1f;  // Fixed impulse duration
            if (t >= t_imp && t < t_imp + t_dur) {
                tau_dist = mag;
            }
            break;
        }

        case DisturbanceParams::COMBINED: {
            // Random sine component
            Vector3 tau_sine(0, 0, 0);
            float f_base = params.dist.torque_freq;
            for (int i = 0; i < 3; i++) {
                float sum = 0;
                for (int j = 0; j < 5; j++) {
                    float f_j = f_base * dist_state.random_sine.freq_mult[i][j] * (j + 1);
                    float phi_j = dist_state.random_sine.phase[i][j];
                    sum += sinf(2.0f * M_PI * f_j * t + phi_j) / (j + 1);
                }
                tau_sine.data()[i] = mag.data()[i] * sum / 3.0f;
            }

            // Step component (50% magnitude)
            Vector3 tau_step(0, 0, 0);
            if (t >= params.dist.gust_start && t < params.dist.gust_start + params.dist.gust_duration) {
                tau_step = mag * 0.5f;
            }

            tau_dist = tau_sine + tau_step;
            break;
        }

        case DisturbanceParams::PAPER: {
            // Paper-style: integrated random sine
            float dt = t - dist_state.t_prev;
            if (dt <= 0) dt = 0.001f;

            // Compute acceleration: sum of multiple sinusoids per axis
            Vector3 accel(0, 0, 0);
            for (int i = 0; i < 3; i++) {
                float sum = 0;
                for (int j = 0; j < 10; j++) {
                    float f_j = dist_state.paper.freq[i][j];
                    float phi_j = dist_state.paper.phase[i][j];
                    float amp_j = dist_state.paper.amp[i][j];
                    sum += amp_j * sinf(2.0f * M_PI * f_j * t + phi_j);
                }
                accel.data()[i] = mag.data()[i] * sum;
            }

            // Integrate (Euler) - creates drift-like behavior
            dist_state.paper.integral.x += dt * accel.x * 2.0f;
            dist_state.paper.integral.y += dt * accel.y * 2.0f;
            dist_state.paper.integral.z += dt * accel.z * 2.0f;

            // Scale and clamp to bounded range
            float scale = 5.0f;  // Default scale
            float max_tau = 0.02f;  // Default max torque
            tau_dist = dist_state.paper.integral / scale;
            tau_dist.x = std::max(std::min(tau_dist.x, max_tau), -max_tau);
            tau_dist.y = std::max(std::min(tau_dist.y, max_tau), -max_tau);
            tau_dist.z = std::max(std::min(tau_dist.z, max_tau), -max_tau);
            break;
        }
    }

    // Update state
    dist_state.t_prev = t;

    return tau_dist;
}

// Wind disturbance
Vector3 dist_wind(const Vector3& vel_b, const Matrix3& R_b2n, float t, float dt,
                  const Params& params, DisturbanceState& dist_state) {
    Vector3 F_wind(0, 0, 0);

    if (!params.dist.enable) {
        return F_wind;
    }

    Matrix3 R_n2b = R_b2n.transpose();
    Vector3 V_wind_ned(0, 0, 0);

    // Get wind velocity in NED frame
    switch (params.dist.wind_type) {
        case DisturbanceParams::WIND_NONE:
            break;

        case DisturbanceParams::CONSTANT:
            V_wind_ned = params.dist.wind_velocity_ned;
            break;

        case DisturbanceParams::GUST: {
            // Base constant wind
            V_wind_ned = params.dist.wind_velocity_ned;

            // Add gust component
            float t_start = params.dist.gust_start;
            float t_dur = params.dist.gust_duration;

            if (t >= t_start && t < t_start + t_dur) {
                // Gust active
                if (!dist_state.gust.active) {
                    // Initialize gust direction (random in horizontal plane)
                    float theta_gust = 2.0f * M_PI * dist_state.uniform_dist(dist_state.rng);
                    dist_state.gust.direction.x = cosf(theta_gust);
                    dist_state.gust.direction.y = sinf(theta_gust);
                    dist_state.gust.direction.z = 0.1f * dist_state.normal_dist(dist_state.rng);
                    float norm = dist_state.gust.direction.norm();
                    if (norm > 0) {
                        dist_state.gust.direction = dist_state.gust.direction / norm;
                    }
                    dist_state.gust.active = true;
                }

                // 1-cosine gust profile
                float t_rel = t - t_start;
                float gust_profile = 0.5f * (1.0f - cosf(2.0f * M_PI * t_rel / t_dur));
                Vector3 V_gust = dist_state.gust.direction * (params.dist.gust_magnitude * gust_profile);
                V_wind_ned = V_wind_ned + V_gust;
            } else {
                dist_state.gust.active = false;
            }
            break;
        }

        case DisturbanceParams::DRYDEN: {
            // Dryden turbulence model
            // White noise input
            float noise = dist_state.normal_dist(dist_state.rng);

            // Euler integration of state-space model
            float x_dot[6];
            for (int i = 0; i < 6; i++) {
                x_dot[i] = dist_state.dryden.B[i] * noise;
                for (int j = 0; j < 6; j++) {
                    x_dot[i] += dist_state.dryden.A[i][j] * dist_state.dryden.x[j];
                }
            }

            for (int i = 0; i < 6; i++) {
                dist_state.dryden.x[i] += x_dot[i] * dt;
            }

            // Output: turbulence velocity in NED
            Vector3 V_turb(0, 0, 0);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 6; j++) {
                    V_turb.data()[i] += dist_state.dryden.C[i][j] * dist_state.dryden.x[j];
                }
            }

            // Add to base wind
            V_wind_ned = params.dist.wind_velocity_ned + V_turb;
            break;
        }
    }

    // Transform wind to body frame
    Vector3 V_wind_body = R_n2b * V_wind_ned;

    // Relative airspeed in body frame
    Vector3 V_air = vel_b - V_wind_body;

    // Aerodynamic drag force
    float Cd = params.dist.aero_Cd;
    float A = params.dist.aero_A;
    float rho = params.env.rho;

    float V_air_mag = V_air.norm();
    if (V_air_mag > 0.01f) {
        F_wind = V_air * (-0.5f * rho * Cd * A * V_air_mag);
    }

    return F_wind;
}

// Apply parameter uncertainty
void apply_uncertainty(const Params& params_nom, Params& params_true) {
    // Copy all parameters
    params_true = params_nom;

    if (!params_nom.dist.enable) {
        return;
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

    bool applied = false;

    // Mass uncertainty
    float delta_m = 0.1f;  // Default 10%
    if (params_nom.dist.torque_type != DisturbanceParams::NONE) {
        float scale = 1.0f + delta_m * (2.0f * uniform_dist(rng) - 1.0f);
        params_true.drone.m = params_nom.drone.m * scale;
        applied = true;
    }

    // Inertia uncertainty (simplified - same uncertainty for all axes)
    float delta_J = 0.1f;  // Default 10%
    if (params_nom.dist.torque_type != DisturbanceParams::NONE) {
        float scale_xx = 1.0f + delta_J * (2.0f * uniform_dist(rng) - 1.0f);
        float scale_yy = 1.0f + delta_J * (2.0f * uniform_dist(rng) - 1.0f);
        float scale_zz = 1.0f + delta_J * (2.0f * uniform_dist(rng) - 1.0f);

        params_true.drone.J[0] = params_nom.drone.J[0] * scale_xx;
        params_true.drone.J[4] = params_nom.drone.J[4] * scale_yy;
        params_true.drone.J[8] = params_nom.drone.J[8] * scale_zz;
        applied = true;
    }

    // Thrust coefficient uncertainty
    float delta_kT = 0.05f;  // Default 5%
    if (params_nom.dist.torque_type != DisturbanceParams::NONE) {
        float scale = 1.0f + delta_kT * (2.0f * uniform_dist(rng) - 1.0f);
        params_true.drone.k_T = params_nom.drone.k_T * scale;
        applied = true;
    }

    if (applied) {
        std::cout << "=== Applied Parameter Uncertainty ===\n";
        std::cout << "Mass:    " << params_true.drone.m << " kg (nominal: " << params_nom.drone.m << " kg)\n";
        std::cout << "Jxx:     " << params_true.drone.J[0] << " (nominal: " << params_nom.drone.J[0] << ")\n";
        std::cout << "Jyy:     " << params_true.drone.J[4] << " (nominal: " << params_nom.drone.J[4] << ")\n";
        std::cout << "Jzz:     " << params_true.drone.J[8] << " (nominal: " << params_nom.drone.J[8] << ")\n";
        std::cout << "k_T:     " << params_true.drone.k_T << " (nominal: " << params_nom.drone.k_T << ")\n";
        std::cout << "=====================================\n";
    }
}

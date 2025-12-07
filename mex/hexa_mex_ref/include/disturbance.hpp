#ifndef DISTURBANCE_HPP
#define DISTURBANCE_HPP

#include "types.hpp"
#include "params.hpp"
#include <random>

// Disturbance state structure
struct DisturbanceState {
    float t_prev;

    // Random sine state
    struct {
        float phase[3][5];       // 5 harmonics per axis
        float freq_mult[3][5];   // Frequency multipliers
    } random_sine;

    // Paper-style disturbance state
    struct {
        Vector3 integral;        // Integrated value
        float freq[3][10];       // Random frequencies [0.1-2.1] Hz
        float phase[3][10];      // Random phases
        float amp[3][10];        // Random amplitudes (normalized)
    } paper;

    // Dryden filter state
    struct {
        float x[6];              // State vector (6 states: 2 per axis)
        float A[6][6];           // State matrix
        float B[6];              // Input matrix
        float C[3][6];           // Output matrix
    } dryden;

    // Gust state
    struct {
        bool active;
        Vector3 direction;
    } gust;

    // Random number generator
    std::mt19937 rng;
    std::normal_distribution<float> normal_dist;
    std::uniform_real_distribution<float> uniform_dist;

    DisturbanceState() : t_prev(0), normal_dist(0.0f, 1.0f), uniform_dist(0.0f, 1.0f) {
        // Initialize random number generator
        std::random_device rd;
        rng.seed(rd());

        // Initialize states to zero
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                random_sine.phase[i][j] = 0;
                random_sine.freq_mult[i][j] = 0;
            }
            for (int j = 0; j < 10; j++) {
                paper.freq[i][j] = 0;
                paper.phase[i][j] = 0;
                paper.amp[i][j] = 0;
            }
        }

        paper.integral = Vector3(0, 0, 0);

        for (int i = 0; i < 6; i++) {
            dryden.x[i] = 0;
            dryden.B[i] = 0;
            for (int j = 0; j < 6; j++) {
                dryden.A[i][j] = 0;
            }
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 6; j++) {
                dryden.C[i][j] = 0;
            }
        }

        gust.active = false;
        gust.direction = Vector3(1, 0, 0);
    }
};

// Disturbance functions
void dist_init(Params& params, const char* preset, DisturbanceState& dist_state);
Vector3 dist_torque(float t, const Params& params, DisturbanceState& dist_state);
Vector3 dist_wind(const Vector3& vel_b, const Matrix3& R_b2n, float t, float dt,
                  const Params& params, DisturbanceState& dist_state);
void apply_uncertainty(const Params& params_nom, Params& params_true);

// Helper functions
void dryden_matrices(float h, const char* intensity, float A[6][6], float B[6], float C[3][6]);

#endif // DISTURBANCE_HPP

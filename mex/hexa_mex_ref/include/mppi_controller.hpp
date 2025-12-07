#ifndef MPPI_CONTROLLER_HPP
#define MPPI_CONTROLLER_HPP

#include "types.hpp"
#include "params.hpp"
#include <vector>
#include <random>

// Control input: [thrust, tau_x, tau_y, tau_z]
struct ControlInput {
    float thrust;      // Total thrust [N]
    Vector3 tau;       // Torque commands [Nm]

    ControlInput(float t = 0, float tx = 0, float ty = 0, float tz = 0)
        : thrust(t), tau(tx, ty, tz) {}

    void to_array(float* arr) const {
        arr[0] = thrust;
        arr[1] = tau.x;
        arr[2] = tau.y;
        arr[3] = tau.z;
    }

    void from_array(const float* arr) {
        thrust = arr[0];
        tau.x = arr[1];
        tau.y = arr[2];
        tau.z = arr[3];
    }

    ControlInput operator+(const ControlInput& other) const {
        return ControlInput(thrust + other.thrust,
                          tau.x + other.tau.x,
                          tau.y + other.tau.y,
                          tau.z + other.tau.z);
    }

    ControlInput operator*(float s) const {
        return ControlInput(thrust * s, tau.x * s, tau.y * s, tau.z * s);
    }
};

// MPPI cost function parameters
struct MPPICostParams {
    // Position tracking weights
    float w_pos_xy;        // Weight for x-y position error
    float w_pos_z;         // Weight for altitude error

    // Velocity weights
    float w_vel;           // Weight for velocity magnitude

    // Attitude weights
    float w_att_roll;      // Weight for roll error
    float w_att_pitch;     // Weight for pitch error
    float w_att_yaw;       // Weight for yaw error
    float w_omega;         // Weight for angular velocity

    // Control cost (R matrix diagonal)
    float R_thrust;        // Control cost for thrust
    float R_tau;           // Control cost for torque

    // Control rate cost (smoothness penalty)
    float R_du_thrust;     // Control rate cost for thrust change
    float R_du_tau;        // Control rate cost for torque change

    // Terminal cost multiplier
    float terminal_factor; // Multiply running cost by this for terminal cost

    // Obstacle/boundary costs (optional)
    bool enable_boundaries;
    Vector3 pos_min;       // Minimum position bounds (NED)
    Vector3 pos_max;       // Maximum position bounds (NED)
    float boundary_cost;   // Cost for violating boundaries

    MPPICostParams() {
        // Default weights (based on paper quadrotor example)
        w_pos_xy = 2.5f;
        w_pos_z = 150.0f;
        w_vel = 1.0f;
        w_att_roll = 50.0f;
        w_att_pitch = 50.0f;
        w_att_yaw = 50.0f;
        w_omega = 1.0f;
        R_thrust = 0.01f;
        R_tau = 0.1f;
        R_du_thrust = 0.01f;   // Control smoothness penalty
        R_du_tau = 0.01f;      // Control smoothness penalty
        terminal_factor = 10.0f;

        enable_boundaries = false;
        pos_min = Vector3(-100, -100, -100);
        pos_max = Vector3(100, 100, 0);
        boundary_cost = 1000.0f;
    }
};

// MPPI reference trajectory/setpoint
struct MPPIReference {
    Vector3 pos_des;       // Desired position (NED) [m]
    Vector3 vel_des;       // Desired velocity (NED) [m/s]
    Quaternion q_des;      // Desired attitude
    Vector3 omega_des;     // Desired angular velocity [rad/s]

    MPPIReference() {
        pos_des = Vector3(0, 0, -10);
        vel_des = Vector3(0, 0, 0);
        q_des = Quaternion(1, 0, 0, 0);
        omega_des = Vector3(0, 0, 0);
    }
};

// MPPI algorithm parameters
struct MPPIParams {
    int K;                 // Number of rollout samples
    int N;                 // Horizon length (timesteps)
    float dt;              // Timestep [s]
    float lambda;          // Temperature parameter

    // Control noise covariance (diagonal)
    float sigma_thrust;    // Noise std for thrust perturbations
    float sigma_tau;       // Noise std for torque perturbations

    // Control limits
    float thrust_min;      // Minimum thrust [N]
    float thrust_max;      // Maximum thrust [N]
    float tau_max;         // Maximum torque magnitude [Nm]

    MPPIParams() {
        K = 512;           // Number of rollouts (CPU default)
        N = 50;            // 1 second horizon at 50Hz
        dt = 0.02f;        // 50Hz control rate
        lambda = 1.0f;     // Temperature (tune this!)

        sigma_thrust = 5.0f;   // Thrust noise [N]
        sigma_tau = 0.5f;      // Torque noise [Nm]

        thrust_min = 0.0f;
        thrust_max = 200.0f;   // ~6kg * 10m/s^2 * 3.3
        tau_max = 10.0f;
    }
};

// MPPI Controller Class
class MPPIController {
public:
    MPPIController(const MPPIParams& mppi_params,
                   const MPPICostParams& cost_params,
                   const Params& drone_params);

    ~MPPIController();

    // Main MPPI control update
    // Returns optimal control input for current state
    ControlInput compute_control(const State28& state,
                                 const MPPIReference& ref,
                                 float t);

    // Set reference trajectory
    void set_reference(const MPPIReference& ref);

    // Update MPPI parameters online
    void set_mppi_params(const MPPIParams& params);
    void set_cost_params(const MPPICostParams& params);

    // Get internal state for debugging
    const std::vector<ControlInput>& get_control_sequence() const { return u_seq_; }
    float get_last_min_cost() const { return min_cost_; }
    float get_last_avg_cost() const { return avg_cost_; }

private:
    // MPPI algorithm implementation
    void mppi_iteration(const State28& state, float t);

    // Rollout simulation
    float simulate_rollout(const State28& state_init,
                          const std::vector<ControlInput>& u_seq,
                          const std::vector<ControlInput>& du_seq,
                          float t);

    // Cost function
    float compute_running_cost(const State28& state,
                              const ControlInput& u,
                              const ControlInput& u_prev,
                              const MPPIReference& ref,
                              bool is_terminal = false);

    // Control utilities
    void clamp_control(ControlInput& u);
    ControlInput sample_noise();

    // Dynamics integration (one step)
    void integrate_step(State28& state, const ControlInput& u, float dt, float t);

    // Parameters
    MPPIParams mppi_params_;
    MPPICostParams cost_params_;
    Params drone_params_;
    MPPIReference ref_;

    // Control sequence (nominal trajectory)
    std::vector<ControlInput> u_seq_;        // Size N

    // Rollout data
    std::vector<float> costs_;               // Size K (cost per rollout)
    std::vector<float> weights_;             // Size K (importance weights)
    std::vector<std::vector<ControlInput>> du_seqs_;  // Size K x N (control variations)

    // Random number generation
    std::mt19937 rng_;
    std::normal_distribution<float> dist_normal_;

    // Debug info
    float min_cost_;
    float avg_cost_;
    int iteration_count_;
};

// Helper functions for quaternion error
Vector3 quaternion_error(const Quaternion& q_des, const Quaternion& q);
float quaternion_distance(const Quaternion& q1, const Quaternion& q2);

#endif // MPPI_CONTROLLER_HPP

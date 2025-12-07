#ifndef PARAMS_HPP
#define PARAMS_HPP

#include "types.hpp"
#include <cstring>

struct DroneParams {
    // Body
    float m;                    // Mass [kg]
    float J[9];                 // Inertia matrix [kg*m^2] (diagonal)
    float L;                    // Arm length [m]
    int n_motor;                // Number of motors

    // Motor
    float k_T;                  // Thrust coefficient [N/(rad/s)^2]
    float k_M;                  // Moment coefficient ratio
    float tau_up;               // Time constant (up) [s]
    float tau_down;             // Time constant (down) [s]
    float omega_b_max;          // Max angular velocity [rad/s]
    float omega_b_min;          // Min angular velocity [rad/s]

    // Aerodynamics
    float kd;                   // Drag coefficient [N/(m/s)]
};

struct EnvParams {
    float g;                    // Gravity [m/s^2]
    float rho;                  // Air density [kg/m^3]
    Vector3 mag_ned;            // Magnetic field (NED) [Gauss]
};

struct SensorParams {
    // Gyro
    float gyro_noise_density;   // [rad/s/sqrt(Hz)]
    float gyro_random_walk;     // [rad/s^2/sqrt(Hz)]
    float gyro_rate;            // [Hz]

    // Accel
    float accel_noise_density;  // [m/s^2/sqrt(Hz)]
    float accel_random_walk;    // [m/s^3/sqrt(Hz)]
    float accel_rate;           // [Hz]

    // GPS
    Vector3 gps_pos_std;        // [m]
    Vector3 gps_vel_std;        // [m/s]
    float gps_rate;             // [Hz]

    // Barometer
    float baro_noise_std;       // [m]
    float baro_rate;            // [Hz]

    // Magnetometer
    float mag_noise_density;    // [Gauss/sqrt(Hz)]
    float mag_random_walk;      // [Gauss/s/sqrt(Hz)]
    float mag_bias_corr_time;   // [s]
    float mag_rate;             // [Hz]
};

struct DisturbanceParams {
    bool enable;

    // Torque disturbance
    enum TorqueType { NONE, SINE, RANDOM_SINE, STEP, IMPULSE, COMBINED, PAPER };
    TorqueType torque_type;
    Vector3 torque_magnitude;   // [Nm]
    float torque_freq;          // [Hz]

    // Wind disturbance
    enum WindType { WIND_NONE, CONSTANT, GUST, DRYDEN };
    WindType wind_type;
    Vector3 wind_velocity_ned;  // [m/s]
    float gust_magnitude;       // [m/s]
    float gust_start;           // [s]
    float gust_duration;        // [s]

    // Aerodynamic
    float aero_Cd;              // Drag coefficient
    float aero_A;               // Reference area [m^2]

    // Parameter uncertainty
    struct {
        bool enable;
        float mass;             // ±fraction (0.1 = ±10%)
        float inertia;          // ±fraction
        float k_T;              // ±fraction
    } uncertainty;
};

struct Params {
    DroneParams drone;
    EnvParams env;
    SensorParams sensor;
    DisturbanceParams dist;

    // Initialize with hexarotor defaults
    void init_hexa() {
        // Drone
        drone.m = 6.2f;
        drone.J[0] = 0.17f; drone.J[1] = 0.0f; drone.J[2] = 0.0f;
        drone.J[3] = 0.0f; drone.J[4] = 0.175f; drone.J[5] = 0.0f;
        drone.J[6] = 0.0f; drone.J[7] = 0.0f; drone.J[8] = 0.263f;
        drone.L = 0.96f;
        drone.n_motor = 6;
        drone.k_T = 1.29e-04f;
        drone.k_M = 2.33e-05f / drone.k_T;
        drone.tau_up = 0.04f;
        drone.tau_down = 0.06f;
        drone.omega_b_max = 650.0f;
        drone.omega_b_min = 0.0f;
        drone.kd = 0.25f;

        // Environment
        env.g = 9.81f;
        env.rho = 1.225f;
        env.mag_ned = Vector3(0.22f, -0.04f, 0.43f);

        // Sensors
        sensor.gyro_noise_density = 0.0003394f;
        sensor.gyro_random_walk = 3.8785e-05f;
        sensor.gyro_rate = 200.0f;

        sensor.accel_noise_density = 0.004f;
        sensor.accel_random_walk = 6.0e-03f;
        sensor.accel_rate = 200.0f;

        sensor.gps_pos_std = Vector3(0.3f, 0.3f, 0.5f);
        sensor.gps_vel_std = Vector3(0.05f, 0.05f, 0.1f);
        sensor.gps_rate = 5.0f;

        sensor.baro_noise_std = 0.5f;
        sensor.baro_rate = 50.0f;

        sensor.mag_noise_density = 0.0004f;
        sensor.mag_random_walk = 6.4e-06f;
        sensor.mag_bias_corr_time = 600.0f;
        sensor.mag_rate = 100.0f;

        // Disturbance (default: disabled)
        dist.enable = false;
        dist.torque_type = DisturbanceParams::NONE;
        dist.torque_magnitude = Vector3(0.02f, 0.02f, 0.02f);
        dist.torque_freq = 0.5f;
        dist.wind_type = DisturbanceParams::WIND_NONE;
        dist.wind_velocity_ned = Vector3(0, 0, 0);
        dist.gust_magnitude = 0;
        dist.gust_start = 0;
        dist.gust_duration = 0;
        dist.aero_Cd = 1.0f;
        dist.aero_A = 0.1f;
        dist.uncertainty.enable = false;
        dist.uncertainty.mass = 0.0f;
        dist.uncertainty.inertia = 0.0f;
        dist.uncertainty.k_T = 0.0f;
    }

    // Set disturbance preset
    void set_disturbance_preset(const char* preset) {
        if (strcmp(preset, "nominal") == 0) {
            dist.enable = false;
        } else if (strcmp(preset, "level1") == 0) {
            dist.enable = true;
            dist.torque_type = DisturbanceParams::RANDOM_SINE;
            dist.torque_magnitude = Vector3(0.02f, 0.02f, 0.02f);
            dist.wind_type = DisturbanceParams::CONSTANT;
            dist.wind_velocity_ned = Vector3(2.0f, 1.0f, 0.0f);
        } else if (strcmp(preset, "level2") == 0) {
            dist.enable = true;
            dist.torque_type = DisturbanceParams::COMBINED;
            dist.torque_magnitude = Vector3(0.03f, 0.03f, 0.03f);
            dist.wind_type = DisturbanceParams::GUST;
            dist.gust_magnitude = 5.0f;
            dist.gust_start = 10.0f;
            dist.gust_duration = 2.0f;
        } else if (strcmp(preset, "level3") == 0) {
            dist.enable = true;
            dist.torque_type = DisturbanceParams::COMBINED;
            dist.torque_magnitude = Vector3(0.03f, 0.03f, 0.03f);
            dist.wind_type = DisturbanceParams::DRYDEN;
            dist.uncertainty.enable = true;
            dist.uncertainty.mass = 0.1f;
            dist.uncertainty.inertia = 0.1f;
            dist.uncertainty.k_T = 0.05f;
        }
    }
};

// Control gains structures
struct PIDGains {
    Vector3 Kp;
    Vector3 Ki;
    Vector3 Kd;
    Vector3 int_limit;

    PIDGains() : Kp(0,0,0), Ki(0,0,0), Kd(0,0,0), int_limit(0,0,0) {}
};

struct SMCGains {
    float a;
    float b;
    float lambda1;
    float lambda2;
    float r;

    SMCGains() : a(5.0f), b(5.0f), lambda1(0.4f), lambda2(0.4f), r(0.95f) {}
};

struct SMCGainsAlt {
    float a;          // Sliding surface gain
    float lambda1;    // Reaching law gain 1
    float lambda2;    // Reaching law gain 2
    float r;          // Fractional order

    SMCGainsAlt() : a(10.0f), lambda1(10.0f), lambda2(0.8f), r(0.98f) {}
};

struct SMCGainsPos {
    Vector3 a;          // Sliding surface gain
    Vector3 lambda1;    // Reaching law gain 1
    Vector3 lambda2;    // Reaching law gain 2
    float r;            // Fractional order
    Vector3 a_max;      // Acceleration saturation [m/s^2]

    SMCGainsPos() :
        a(2.0f, 2.0f, 4.0f),
        lambda1(1.0f, 1.0f, 2.0f),
        lambda2(0.3f, 0.3f, 0.5f),
        r(0.9f),
        a_max(4.0f, 4.0f, 6.0f) {}
};

// Controller state
struct ControlState {
    Vector3 int_att;
    Vector3 int_pos;
    float int_alt;
    float dt;

    ControlState() : int_att(0,0,0), int_pos(0,0,0), int_alt(0), dt(0.001f) {}
};

#endif // PARAMS_HPP

#include "../include/types.hpp"
#include "../include/params.hpp"
#include "../include/math_utils.hpp"
#include <algorithm>

// Attitude PID controller
void attitude_pid(const Quaternion& q_des, const Quaternion& q, const Vector3& omega,
                  ControlState& ctrl_state, const PIDGains& gains, float dt, Vector3& tau_cmd) {
    // Quaternion error (body frame): q_err = q_inv ⊗ q_des
    Quaternion q_inv = q.conjugate();
    Quaternion q_err = quat_mult(q_inv, q_des);

    // Ensure shortest path
    if (q_err.w < 0) {
        q_err.w = -q_err.w;
        q_err.x = -q_err.x;
        q_err.y = -q_err.y;
        q_err.z = -q_err.z;
    }

    // Attitude error vector: e_att ≈ 2 * q_err.xyz
    Vector3 e_att(2.0f * q_err.x, 2.0f * q_err.y, 2.0f * q_err.z);

    // Angular rate error (desired rate = 0 for hold)
    Vector3 e_rate = Vector3(0, 0, 0) - omega;

    // Integrator with anti-windup
    ctrl_state.int_att.x += e_att.x * dt;
    ctrl_state.int_att.y += e_att.y * dt;
    ctrl_state.int_att.z += e_att.z * dt;

    ctrl_state.int_att.x = std::max(std::min(ctrl_state.int_att.x, gains.int_limit.x), -gains.int_limit.x);
    ctrl_state.int_att.y = std::max(std::min(ctrl_state.int_att.y, gains.int_limit.y), -gains.int_limit.y);
    ctrl_state.int_att.z = std::max(std::min(ctrl_state.int_att.z, gains.int_limit.z), -gains.int_limit.z);

    // PID control law
    tau_cmd.x = gains.Kp.x * e_att.x + gains.Ki.x * ctrl_state.int_att.x + gains.Kd.x * e_rate.x;
    tau_cmd.y = gains.Kp.y * e_att.y + gains.Ki.y * ctrl_state.int_att.y + gains.Kd.y * e_rate.y;
    tau_cmd.z = gains.Kp.z * e_att.z + gains.Ki.z * ctrl_state.int_att.z + gains.Kd.z * e_rate.z;
}

// Altitude PID controller
void altitude_pid(float alt_des, float alt, float vel_z, ControlState& ctrl_state,
                  const PIDGains& gains, float dt, const Params& params, float& thrust_cmd) {
    // Altitude error
    float e_alt = alt_des - alt;

    // Derivative (e_dot = -vel_z)
    float e_dot = -vel_z;

    // Integrator with anti-windup
    ctrl_state.int_alt += e_alt * dt;
    ctrl_state.int_alt = std::max(std::min(ctrl_state.int_alt, gains.int_limit.z), -gains.int_limit.z);

    // PID output (acceleration command)
    float acc_cmd = gains.Kp.z * e_alt + gains.Ki.z * ctrl_state.int_alt + gains.Kd.z * e_dot;

    // Convert to thrust: T = m * (g + acc_cmd)
    // Note: For PID, we don't compensate for tilt to match MATLAB implementation
    // The attitude controller brings drone to level, then altitude control works
    thrust_cmd = params.drone.m * (params.env.g + acc_cmd);

    // Thrust saturation
    float thrust_max = params.drone.n_motor * params.drone.k_T * params.drone.omega_b_max * params.drone.omega_b_max;
    float thrust_min = 0;
    thrust_cmd = std::max(std::min(thrust_cmd, thrust_max), thrust_min);
}

// Position PID controller
void position_pid(const Vector3& pos_des, const Vector3& pos, const Vector3& vel_ned,
                  float yaw_des, ControlState& ctrl_state, const PIDGains& gains,
                  const Params& params, Quaternion& q_des, float& thrust_cmd) {
    // Position error (NED)
    Vector3 e_pos = pos_des - pos;

    // Velocity error (desired velocity = 0 for waypoint hold)
    Vector3 e_vel = Vector3(0, 0, 0) - vel_ned;

    // Integrator with anti-windup
    ctrl_state.int_pos.x += e_pos.x * ctrl_state.dt;
    ctrl_state.int_pos.y += e_pos.y * ctrl_state.dt;
    ctrl_state.int_pos.z += e_pos.z * ctrl_state.dt;

    ctrl_state.int_pos.x = std::max(std::min(ctrl_state.int_pos.x, gains.int_limit.x), -gains.int_limit.x);
    ctrl_state.int_pos.y = std::max(std::min(ctrl_state.int_pos.y, gains.int_limit.y), -gains.int_limit.y);
    ctrl_state.int_pos.z = std::max(std::min(ctrl_state.int_pos.z, gains.int_limit.z), -gains.int_limit.z);

    // PID: desired acceleration (NED)
    Vector3 a_des;
    a_des.x = gains.Kp.x * e_pos.x + gains.Ki.x * ctrl_state.int_pos.x + gains.Kd.x * e_vel.x;
    a_des.y = gains.Kp.y * e_pos.y + gains.Ki.y * ctrl_state.int_pos.y + gains.Kd.y * e_vel.y;
    a_des.z = gains.Kp.z * e_pos.z + gains.Ki.z * ctrl_state.int_pos.z + gains.Kd.z * e_vel.z;

    // Acceleration to thrust and attitude
    Vector3 g_ned(0, 0, params.env.g);
    Vector3 F_des = a_des - g_ned;
    F_des = F_des * params.drone.m;

    // Total thrust magnitude
    thrust_cmd = F_des.norm();

    // Desired body z-axis in NED frame
    Vector3 z_b_des;
    if (thrust_cmd > 1e-6f) {
        z_b_des = F_des * (-1.0f / thrust_cmd);
    } else {
        z_b_des = Vector3(0, 0, -1);  // Default: level
    }

    // Construct desired rotation matrix
    // Desired x-axis direction (from yaw)
    Vector3 x_c(cosf(yaw_des), sinf(yaw_des), 0);

    // Desired y-axis: z × x (then normalize)
    Vector3 y_b_des = z_b_des.cross(x_c);
    float y_norm = y_b_des.norm();
    if (y_norm > 1e-6f) {
        y_b_des = y_b_des / y_norm;
    } else {
        y_b_des = Vector3(0, 1, 0);
    }

    // Recompute x-axis to ensure orthogonality
    Vector3 x_b_des = y_b_des.cross(z_b_des);
    x_b_des = x_b_des.normalized();

    // Rotation matrix (body to NED)
    Matrix3 R_b2n_des;
    R_b2n_des.data[0] = x_b_des.x; R_b2n_des.data[1] = y_b_des.x; R_b2n_des.data[2] = z_b_des.x;
    R_b2n_des.data[3] = x_b_des.y; R_b2n_des.data[4] = y_b_des.y; R_b2n_des.data[5] = z_b_des.y;
    R_b2n_des.data[6] = x_b_des.z; R_b2n_des.data[7] = y_b_des.z; R_b2n_des.data[8] = z_b_des.z;

    // Convert to quaternion
    q_des = dcm_to_quat(R_b2n_des);

    // Ensure positive scalar part
    if (q_des.w < 0) {
        q_des.w = -q_des.w;
        q_des.x = -q_des.x;
        q_des.y = -q_des.y;
        q_des.z = -q_des.z;
    }
}

// Attitude SMC controller
void attitude_smc(const Quaternion& q_des, const Quaternion& q, const Vector3& omega,
                  ControlState& ctrl_state, const SMCGains& gains, const Params& params,
                  Vector3& tau_cmd) {
    (void)ctrl_state;  // Not used for SMC

    // Quaternion error: q_err = conj(q_des) ⊗ q
    Quaternion q_des_inv = q_des.conjugate();
    Quaternion q_err = quat_mult(q_des_inv, q);

    // Ensure shortest path
    float sign_q0 = 1.0f;
    if (q_err.w < 0) {
        q_err.w = -q_err.w;
        q_err.x = -q_err.x;
        q_err.y = -q_err.y;
        q_err.z = -q_err.z;
        sign_q0 = -1.0f;
    }

    // Error vector part
    float q_err_0 = q_err.w;
    Vector3 q_err_v(q_err.x, q_err.y, q_err.z);

    // Fractional power term: |q_err_v|^r .* sign(q_err_v)
    Vector3 b_gain;
    b_gain.x = gains.b * powf(fabsf(q_err_v.x), gains.r) * sign(q_err_v.x);
    b_gain.y = gains.b * powf(fabsf(q_err_v.y), gains.r) * sign(q_err_v.y);
    b_gain.z = gains.b * powf(fabsf(q_err_v.z), gains.r) * sign(q_err_v.z);

    // Sliding surface: s = omega + a*q_err_v + b*|q_err_v|^r .* sign(q_err_v)
    Vector3 s;
    s.x = omega.x + gains.a * q_err_v.x + b_gain.x;
    s.y = omega.y + gains.a * q_err_v.y + b_gain.y;
    s.z = omega.z + gains.a * q_err_v.z + b_gain.z;

    // Fractional reaching law: s_dot = -lambda1*s - lambda2*|s|^r .* sign(s)
    Vector3 s_dot;
    s_dot.x = -gains.lambda1 * s.x - gains.lambda2 * powf(fabsf(s.x), gains.r) * sign(s.x);
    s_dot.y = -gains.lambda1 * s.y - gains.lambda2 * powf(fabsf(s.y), gains.r) * sign(s.y);
    s_dot.z = -gains.lambda1 * s.z - gains.lambda2 * powf(fabsf(s.z), gains.r) * sign(s.z);

    // Inertia matrix
    const float* J = params.drone.J;

    // omega × (J*omega)
    Vector3 J_omega(J[0] * omega.x, J[4] * omega.y, J[8] * omega.z);
    Vector3 omega_cross_Jomega = omega.cross(J_omega);

    // Quaternion derivative contribution (MATLAB version)
    // coeff = a + b*r*|q_err_v|^(r-1)
    float eps_val = 1e-6f;
    Vector3 q_err_v_safe;
    q_err_v_safe.x = std::max(fabsf(q_err_v.x), eps_val);
    q_err_v_safe.y = std::max(fabsf(q_err_v.y), eps_val);
    q_err_v_safe.z = std::max(fabsf(q_err_v.z), eps_val);

    Vector3 coeff;
    coeff.x = gains.a + gains.b * gains.r * powf(q_err_v_safe.x, gains.r - 1.0f);
    coeff.y = gains.a + gains.b * gains.r * powf(q_err_v_safe.y, gains.r - 1.0f);
    coeff.z = gains.a + gains.b * gains.r * powf(q_err_v_safe.z, gains.r - 1.0f);

    // M = skew(q_err_v) + q_err_0 * I
    Matrix3 skew_qv = skew3(q_err_v);
    Matrix3 M;
    M.data[0] = skew_qv.data[0] + q_err_0;  M.data[1] = skew_qv.data[1];           M.data[2] = skew_qv.data[2];
    M.data[3] = skew_qv.data[3];            M.data[4] = skew_qv.data[4] + q_err_0; M.data[5] = skew_qv.data[5];
    M.data[6] = skew_qv.data[6];            M.data[7] = skew_qv.data[7];           M.data[8] = skew_qv.data[8] + q_err_0;

    // q_dot_term = coeff .* (M * omega / 2)
    Vector3 M_omega = M * omega;
    Vector3 q_dot_term;
    q_dot_term.x = coeff.x * M_omega.x * 0.5f;
    q_dot_term.y = coeff.y * M_omega.y * 0.5f;
    q_dot_term.z = coeff.z * M_omega.z * 0.5f;

    // Control torque: tau = omega×(J*omega) - J*sign(q_err_0)*q_dot_term + J*s_dot
    tau_cmd.x = omega_cross_Jomega.x - J[0] * sign_q0 * q_dot_term.x + J[0] * s_dot.x;
    tau_cmd.y = omega_cross_Jomega.y - J[4] * sign_q0 * q_dot_term.y + J[4] * s_dot.y;
    tau_cmd.z = omega_cross_Jomega.z - J[8] * sign_q0 * q_dot_term.z + J[8] * s_dot.z;
}

// Altitude SMC controller
void altitude_smc(float alt_des, float alt, float vel_z, const Quaternion& q,
                  ControlState& ctrl_state, const SMCGainsAlt& gains, float dt,
                  const Params& params, float& thrust_cmd) {
    (void)ctrl_state;  // Not used for SMC
    (void)dt;  // Not used

    float m = params.drone.m;
    float g = params.env.g;

    // Attitude compensation - get roll and pitch for thrust direction
    Vector3 euler = quat_to_euler(q);
    float phi = euler.x;    // roll
    float theta = euler.y;  // pitch

    // Avoid division by zero
    float cos_phi_theta = cosf(phi) * cosf(theta);
    if (fabsf(cos_phi_theta) < 0.1f) {
        cos_phi_theta = sign(cos_phi_theta) * 0.1f;
    }

    // Altitude error
    float e = alt_des - alt;

    // Sliding surface (fractional-order): s = -ż + a*|e|^r * sign(e)
    float eps_val = 1e-6f;
    float e_safe = std::max(fabsf(e), eps_val);
    float s = -vel_z + gains.a * powf(e_safe, gains.r) * sign(e);

    // Reaching law (fractional-order): ṡ = -λ₁*s - λ₂*|s|^r * sign(s)
    float s_safe = std::max(fabsf(s), eps_val);
    float s_dot = -gains.lambda1 * s - gains.lambda2 * powf(s_safe, gains.r) * sign(s);

    // Control input: T = (a*(-ż) + g - ṡ) * m / (cosφ*cosθ)
    thrust_cmd = (gains.a * (-vel_z) + g - s_dot) * m / cos_phi_theta;

    // Thrust saturation
    float thrust_max = params.drone.n_motor * params.drone.k_T * params.drone.omega_b_max * params.drone.omega_b_max;
    float thrust_min = 0;
    thrust_cmd = std::max(std::min(thrust_cmd, thrust_max), thrust_min);
}

// Position SMC controller
void position_smc(const Vector3& pos_des, const Vector3& pos, const Vector3& vel_ned,
                  float yaw_des, ControlState& ctrl_state, const SMCGainsPos& gains,
                  const Params& params, Quaternion& q_des, float& thrust_cmd) {
    (void)ctrl_state;  // Not used for SMC

    // Position error (NED)
    Vector3 e_pos = pos_des - pos;

    // Sliding surface: s = -vel + a.*e_pos
    Vector3 s;
    s.x = -vel_ned.x + gains.a.x * e_pos.x;
    s.y = -vel_ned.y + gains.a.y * e_pos.y;
    s.z = -vel_ned.z + gains.a.z * e_pos.z;

    // Safe fractional power (avoid singularity at s=0) - MATLAB: max(abs(s), eps_val)
    float eps_val = 1e-6f;
    Vector3 s_safe;
    s_safe.x = std::max(fabsf(s.x), eps_val);
    s_safe.y = std::max(fabsf(s.y), eps_val);
    s_safe.z = std::max(fabsf(s.z), eps_val);

    // Control law: a_des = -a.*vel + lambda1.*s + lambda2.*|s|^r.*sign(s)
    Vector3 a_des;
    a_des.x = -gains.a.x * vel_ned.x + gains.lambda1.x * s.x + gains.lambda2.x * powf(s_safe.x, gains.r) * sign(s.x);
    a_des.y = -gains.a.y * vel_ned.y + gains.lambda1.y * s.y + gains.lambda2.y * powf(s_safe.y, gains.r) * sign(s.y);
    a_des.z = -gains.a.z * vel_ned.z + gains.lambda1.z * s.z + gains.lambda2.z * powf(s_safe.z, gains.r) * sign(s.z);

    // Acceleration saturation
    a_des.x = std::max(std::min(a_des.x, gains.a_max.x), -gains.a_max.x);
    a_des.y = std::max(std::min(a_des.y, gains.a_max.y), -gains.a_max.y);
    a_des.z = std::max(std::min(a_des.z, gains.a_max.z), -gains.a_max.z);

    // Convert acceleration to thrust and attitude
    Vector3 g_ned(0, 0, params.env.g);
    Vector3 F_des = a_des - g_ned;
    F_des = F_des * params.drone.m;

    // MATLAB: Prevent thrust reversal (drone can't flip over)
    // F_des(3) should be negative (upward force in NED)
    if (F_des.z > -0.1f * params.drone.m * params.env.g) {
        F_des.z = -0.1f * params.drone.m * params.env.g;
    }

    // Total thrust magnitude
    thrust_cmd = F_des.norm();

    // Desired body z-axis in NED frame
    Vector3 z_b_des;
    if (thrust_cmd > 1e-6f) {
        z_b_des = F_des * (-1.0f / thrust_cmd);
    } else {
        z_b_des = Vector3(0, 0, -1);  // Default: level
    }

    // Construct desired rotation matrix
    // Desired x-axis direction (from yaw)
    Vector3 x_c(cosf(yaw_des), sinf(yaw_des), 0);

    // Desired y-axis: z × x (then normalize)
    Vector3 y_b_des = z_b_des.cross(x_c);
    float y_norm = y_b_des.norm();
    if (y_norm > 1e-6f) {
        y_b_des = y_b_des / y_norm;
    } else {
        y_b_des = Vector3(0, 1, 0);
    }

    // Recompute x-axis to ensure orthogonality
    Vector3 x_b_des = y_b_des.cross(z_b_des);
    x_b_des = x_b_des.normalized();

    // Rotation matrix (body to NED)
    Matrix3 R_b2n_des;
    R_b2n_des.data[0] = x_b_des.x; R_b2n_des.data[1] = y_b_des.x; R_b2n_des.data[2] = z_b_des.x;
    R_b2n_des.data[3] = x_b_des.y; R_b2n_des.data[4] = y_b_des.y; R_b2n_des.data[5] = z_b_des.y;
    R_b2n_des.data[6] = x_b_des.z; R_b2n_des.data[7] = y_b_des.z; R_b2n_des.data[8] = z_b_des.z;

    // Convert to quaternion
    q_des = dcm_to_quat(R_b2n_des);

    // Ensure positive scalar part
    if (q_des.w < 0) {
        q_des.w = -q_des.w;
        q_des.x = -q_des.x;
        q_des.y = -q_des.y;
        q_des.z = -q_des.z;
    }
}

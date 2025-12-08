/*
 * hexa_mppi_mex_v2.cu - Hexarotor GMPPI Controller (CUDA MEX)
 * 
 * RK4 integration for rigid body dynamics
 * Euler integration for motor dynamics
 * PD reference rollouts (K_pid samples)
 * 
 * State: 19x1 [pos(3), vel(3), quat(4), omega(3), omega_motor(6)]
 * Control: 6x1 [omega_1, ..., omega_6] motor speed commands [rad/s]
 * 
 * Compile: mexcuda hexa_mppi_mex_v2.cu -lcurand
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>

// Persistent GPU memory
static bool initialized = false;
static int K_alloc = 0, N_alloc = 0;
static float *d_x0 = nullptr, *d_u_seq = nullptr, *d_u_prev = nullptr;
static float *d_pos_des = nullptr, *d_q_des = nullptr;
static float *d_noise = nullptr, *d_costs = nullptr, *d_weights = nullptr;
static float *d_du_all = nullptr, *d_du_weighted = nullptr;
static float *d_cost_breakdown = nullptr;
static float *d_pid_noise = nullptr;
static int *d_sat_count = nullptr;
static curandGenerator_t gen;

void cleanup(void) {
    if (initialized) {
        cudaFree(d_x0); cudaFree(d_u_seq); cudaFree(d_u_prev);
        cudaFree(d_pos_des); cudaFree(d_q_des);
        cudaFree(d_noise); cudaFree(d_costs); cudaFree(d_weights);
        cudaFree(d_du_all); cudaFree(d_du_weighted);
        cudaFree(d_cost_breakdown); cudaFree(d_sat_count);
        cudaFree(d_pid_noise);
        curandDestroyGenerator(gen);
        initialized = false;
    }
}

__global__ void scale_noise_kernel(float* noise, float sigma, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        noise[idx] *= sigma;
    }
}

// Device function: compute rigid body derivatives
__device__ void compute_derivatives(
    float px, float py, float pz,
    float vx, float vy, float vz,
    float qw, float qx, float qy, float qz,
    float wx, float wy, float wz,
    float wm1, float wm2, float wm3, float wm4, float wm5, float wm6,
    float m, float Jxx, float Jyy, float Jzz, float g,
    float k_T, float kL, float b,
    float& px_dot, float& py_dot, float& pz_dot,
    float& vx_dot, float& vy_dot, float& vz_dot,
    float& qw_dot, float& qx_dot, float& qy_dot, float& qz_dot,
    float& wx_dot, float& wy_dot, float& wz_dot)
{
    const float s3 = 0.866025403784f;
    
    float R11 = 1.0f - 2.0f*(qy*qy + qz*qz);
    float R12 = 2.0f*(qx*qy - qz*qw);
    float R13 = 2.0f*(qx*qz + qy*qw);
    float R21 = 2.0f*(qx*qy + qz*qw);
    float R22 = 1.0f - 2.0f*(qx*qx + qz*qz);
    float R23 = 2.0f*(qy*qz - qx*qw);
    float R31 = 2.0f*(qx*qz - qy*qw);
    float R32 = 2.0f*(qy*qz + qx*qw);
    float R33 = 1.0f - 2.0f*(qx*qx + qy*qy);
    
    float w1sq = wm1*wm1, w2sq = wm2*wm2, w3sq = wm3*wm3;
    float w4sq = wm4*wm4, w5sq = wm5*wm5, w6sq = wm6*wm6;
    
    float T = k_T * (w1sq + w2sq + w3sq + w4sq + w5sq + w6sq);
    float tau_x = kL * (-0.5f*w1sq + 0.5f*w2sq + w3sq + 0.5f*w4sq - 0.5f*w5sq - w6sq);
    float tau_y = kL * s3 * (w1sq + w2sq - w4sq - w5sq);
    float tau_z = b * (w1sq - w2sq + w3sq - w4sq + w5sq - w6sq);
    
    px_dot = R11*vx + R12*vy + R13*vz;
    py_dot = R21*vx + R22*vy + R23*vz;
    pz_dot = R31*vx + R32*vy + R33*vz;
    
    float gx_b = R31 * g;
    float gy_b = R32 * g;
    float gz_b = R33 * g;
    
    float cross_x = wy*vz - wz*vy;
    float cross_y = wz*vx - wx*vz;
    float cross_z = wx*vy - wy*vx;
    
    vx_dot = gx_b - cross_x;
    vy_dot = gy_b - cross_y;
    vz_dot = -T/m + gz_b - cross_z;
    
    qw_dot = 0.5f * (-wx*qx - wy*qy - wz*qz);
    qx_dot = 0.5f * ( wx*qw + wz*qy - wy*qz);
    qy_dot = 0.5f * ( wy*qw - wz*qx + wx*qz);
    qz_dot = 0.5f * ( wz*qw + wy*qx - wx*qy);
    
    float Jw_x = Jxx * wx;
    float Jw_y = Jyy * wy;
    float Jw_z = Jzz * wz;
    
    wx_dot = (tau_x - (wy*Jw_z - wz*Jw_y)) / Jxx;
    wy_dot = (tau_y - (wz*Jw_x - wx*Jw_z)) / Jyy;
    wz_dot = (tau_z - (wx*Jw_y - wy*Jw_x)) / Jzz;
}

// Device function: Position PD controller
// Returns desired quaternion and thrust
__device__ void position_pd(
    float px, float py, float pz,           // current position (NED)
    float vn, float ve, float vd,           // current velocity (NED)
    float pd_x, float pd_y, float pd_z,     // desired position (NED)
    float yaw_des,                          // desired yaw
    float Kp_x, float Kp_y, float Kp_z,     // P gains
    float Kd_x, float Kd_y, float Kd_z,     // D gains
    float m, float g,
    float& qw_des, float& qx_des, float& qy_des, float& qz_des,
    float& thrust)
{
    // Position error
    float ex = pd_x - px;
    float ey = pd_y - py;
    float ez = pd_z - pz;
    
    // Velocity error (desired = 0)
    float evx = -vn;
    float evy = -ve;
    float evz = -vd;
    
    // Desired acceleration (NED)
    float ax_des = Kp_x * ex + Kd_x * evx;
    float ay_des = Kp_y * ey + Kd_y * evy;
    float az_des = Kp_z * ez + Kd_z * evz;
    
    // Desired force: F = m * (a_des - g_ned), g_ned = [0,0,g]
    float Fx = m * ax_des;
    float Fy = m * ay_des;
    float Fz = m * (az_des - g);
    
    // Thrust magnitude
    thrust = sqrtf(Fx*Fx + Fy*Fy + Fz*Fz);
    if (thrust < 0.1f) thrust = 0.1f;
    
    // Desired body z-axis (NED): z_b = -F/|F|
    float zb_x = -Fx / thrust;
    float zb_y = -Fy / thrust;
    float zb_z = -Fz / thrust;
    
    // Desired x direction from yaw
    float xc_x = cosf(yaw_des);
    float xc_y = sinf(yaw_des);
    float xc_z = 0.0f;
    
    // y_b = z_b x x_c (cross product)
    float yb_x = zb_y * xc_z - zb_z * xc_y;
    float yb_y = zb_z * xc_x - zb_x * xc_z;
    float yb_z = zb_x * xc_y - zb_y * xc_x;
    float yb_norm = sqrtf(yb_x*yb_x + yb_y*yb_y + yb_z*yb_z);
    if (yb_norm < 1e-6f) {
        yb_x = 0.0f; yb_y = 1.0f; yb_z = 0.0f;
    } else {
        yb_x /= yb_norm; yb_y /= yb_norm; yb_z /= yb_norm;
    }
    
    // x_b = y_b x z_b
    float xb_x = yb_y * zb_z - yb_z * zb_y;
    float xb_y = yb_z * zb_x - yb_x * zb_z;
    float xb_z = yb_x * zb_y - yb_y * zb_x;
    float xb_norm = sqrtf(xb_x*xb_x + xb_y*xb_y + xb_z*xb_z);
    xb_x /= xb_norm; xb_y /= xb_norm; xb_z /= xb_norm;
    
    // DCM to quaternion (Shepperd method)
    // R = [xb, yb, zb] (columns)
    float R11 = xb_x, R12 = yb_x, R13 = zb_x;
    float R21 = xb_y, R22 = yb_y, R23 = zb_y;
    float R31 = xb_z, R32 = yb_z, R33 = zb_z;
    
    float trace = R11 + R22 + R33;
    if (trace > 0) {
        float s = 0.5f / sqrtf(trace + 1.0f);
        qw_des = 0.25f / s;
        qx_des = (R32 - R23) * s;
        qy_des = (R13 - R31) * s;
        qz_des = (R21 - R12) * s;
    } else if (R11 > R22 && R11 > R33) {
        float s = 2.0f * sqrtf(1.0f + R11 - R22 - R33);
        qw_des = (R32 - R23) / s;
        qx_des = 0.25f * s;
        qy_des = (R12 + R21) / s;
        qz_des = (R13 + R31) / s;
    } else if (R22 > R33) {
        float s = 2.0f * sqrtf(1.0f + R22 - R11 - R33);
        qw_des = (R13 - R31) / s;
        qx_des = (R12 + R21) / s;
        qy_des = 0.25f * s;
        qz_des = (R23 + R32) / s;
    } else {
        float s = 2.0f * sqrtf(1.0f + R33 - R11 - R22);
        qw_des = (R21 - R12) / s;
        qx_des = (R13 + R31) / s;
        qy_des = (R23 + R32) / s;
        qz_des = 0.25f * s;
    }
    
    // Normalize
    float qn = rsqrtf(qw_des*qw_des + qx_des*qx_des + qy_des*qy_des + qz_des*qz_des);
    qw_des *= qn; qx_des *= qn; qy_des *= qn; qz_des *= qn;
    
    // Ensure positive scalar
    if (qw_des < 0) {
        qw_des = -qw_des; qx_des = -qx_des;
        qy_des = -qy_des; qz_des = -qz_des;
    }
}

// Device function: Attitude PD controller
// Returns torque command
__device__ void attitude_pd(
    float qw, float qx, float qy, float qz,         // current quaternion
    float wx, float wy, float wz,                   // current angular velocity
    float qw_des, float qx_des, float qy_des, float qz_des,  // desired quaternion
    float Kp_r, float Kp_p, float Kp_y,             // P gains (roll, pitch, yaw)
    float Kd_r, float Kd_p, float Kd_y,             // D gains
    float& tau_x, float& tau_y, float& tau_z)
{
    // Quaternion error: q_err = q_des^{-1} * q = conj(q_des) * q
    float qd_conj_w = qw_des;
    float qd_conj_x = -qx_des;
    float qd_conj_y = -qy_des;
    float qd_conj_z = -qz_des;
    
    // Quaternion multiplication: q_err = qd_conj * q
    float qe_w = qd_conj_w*qw - qd_conj_x*qx - qd_conj_y*qy - qd_conj_z*qz;
    float qe_x = qd_conj_w*qx + qd_conj_x*qw + qd_conj_y*qz - qd_conj_z*qy;
    float qe_y = qd_conj_w*qy - qd_conj_x*qz + qd_conj_y*qw + qd_conj_z*qx;
    float qe_z = qd_conj_w*qz + qd_conj_x*qy - qd_conj_y*qx + qd_conj_z*qw;
    
    // Ensure shortest path
    if (qe_w < 0) {
        qe_w = -qe_w; qe_x = -qe_x; qe_y = -qe_y; qe_z = -qe_z;
    }
    
    // Attitude error: e_att = 2 * q_err_vec
    float e_r = 2.0f * qe_x;
    float e_p = 2.0f * qe_y;
    float e_y = 2.0f * qe_z;
    
    // Angular rate error (desired = 0)
    float ew_r = -wx;
    float ew_p = -wy;
    float ew_y = -wz;
    
    // PD control
    tau_x = Kp_r * e_r + Kd_r * ew_r;
    tau_y = Kp_p * e_p + Kd_p * ew_p;
    tau_z = Kp_y * e_y + Kd_y * ew_y;
}

// Device function: Control allocator inverse (hexarotor)
// [T; tau_x; tau_y; tau_z] -> [omega_1, ..., omega_6]
__device__ void control_allocator_inverse(
    float T, float tau_x, float tau_y, float tau_z,
    float k_T, float k_M, float L,
    float omega_min, float omega_max,
    float& u1, float& u2, float& u3, float& u4, float& u5, float& u6)
{
    // Hexarotor allocation matrix pseudo-inverse (precomputed)
    // B = [k, k, k, k, k, k;
    //      -k*L/2, k*L/2, k*L, k*L/2, -k*L/2, -k*L;
    //      k*L*s3, k*L*s3, 0, -k*L*s3, -k*L*s3, 0;
    //      b, -b, b, -b, b, -b]
    // omega_sq = pinv(B) * [T; tau]
    
    float k = k_T;
    float b = k_T * k_M;
    float s3 = 0.866025403784f;
    
    // Simplified pseudo-inverse for symmetric hexarotor
    // Each motor contributes 1/6 of thrust
    float T_per_motor = T / 6.0f;
    
    // Roll (tau_x) allocation
    float roll_1 = -tau_x / (3.0f * k * L);
    float roll_2 =  tau_x / (3.0f * k * L);
    float roll_3 =  tau_x / (1.5f * k * L);
    float roll_4 =  tau_x / (3.0f * k * L);
    float roll_5 = -tau_x / (3.0f * k * L);
    float roll_6 = -tau_x / (1.5f * k * L);
    
    // Pitch (tau_y) allocation
    float pitch_1 =  tau_y / (4.0f * k * L * s3);
    float pitch_2 =  tau_y / (4.0f * k * L * s3);
    float pitch_3 =  0.0f;
    float pitch_4 = -tau_y / (4.0f * k * L * s3);
    float pitch_5 = -tau_y / (4.0f * k * L * s3);
    float pitch_6 =  0.0f;
    
    // Yaw (tau_z) allocation
    float yaw_1 =  tau_z / (6.0f * b);
    float yaw_2 = -tau_z / (6.0f * b);
    float yaw_3 =  tau_z / (6.0f * b);
    float yaw_4 = -tau_z / (6.0f * b);
    float yaw_5 =  tau_z / (6.0f * b);
    float yaw_6 = -tau_z / (6.0f * b);
    
    // Total omega^2 for each motor
    float w1_sq = T_per_motor / k + roll_1 + pitch_1 + yaw_1;
    float w2_sq = T_per_motor / k + roll_2 + pitch_2 + yaw_2;
    float w3_sq = T_per_motor / k + roll_3 + pitch_3 + yaw_3;
    float w4_sq = T_per_motor / k + roll_4 + pitch_4 + yaw_4;
    float w5_sq = T_per_motor / k + roll_5 + pitch_5 + yaw_5;
    float w6_sq = T_per_motor / k + roll_6 + pitch_6 + yaw_6;
    
    // Clamp and sqrt
    w1_sq = fmaxf(w1_sq, 0.0f);
    w2_sq = fmaxf(w2_sq, 0.0f);
    w3_sq = fmaxf(w3_sq, 0.0f);
    w4_sq = fmaxf(w4_sq, 0.0f);
    w5_sq = fmaxf(w5_sq, 0.0f);
    w6_sq = fmaxf(w6_sq, 0.0f);
    
    u1 = sqrtf(w1_sq);
    u2 = sqrtf(w2_sq);
    u3 = sqrtf(w3_sq);
    u4 = sqrtf(w4_sq);
    u5 = sqrtf(w5_sq);
    u6 = sqrtf(w6_sq);
    
    // Saturate
    u1 = fminf(fmaxf(u1, omega_min), omega_max);
    u2 = fminf(fmaxf(u2, omega_min), omega_max);
    u3 = fminf(fmaxf(u3, omega_min), omega_max);
    u4 = fminf(fmaxf(u4, omega_min), omega_max);
    u5 = fminf(fmaxf(u5, omega_min), omega_max);
    u6 = fminf(fmaxf(u6, omega_min), omega_max);
}

// Main MPPI rollout kernel
__global__ void hexa_mppi_rollout_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ u_seq,
    const float* __restrict__ u_prev,
    const float* __restrict__ pos_des,
    const float* __restrict__ q_des,
    const float* __restrict__ noise,
    const float* __restrict__ pid_noise,
    float* __restrict__ costs,
    float* __restrict__ du_all,
    float* __restrict__ cost_breakdown,
    int* __restrict__ sat_count,
    float m, float Jxx, float Jyy, float Jzz, float g,
    float k_T, float k_M, float L,
    float omega_max, float omega_min,
    float tau_up, float tau_down,
    float dt, int N, int K, int K_pid,
    float w_pos, float w_vel, float w_att, float w_omega,
    float w_terminal, float w_smooth, float R_val,
    float crash_cost, float crash_angle,
    float lambda, float nu_c,
    float Kp_pos_x, float Kp_pos_y, float Kp_pos_z,
    float Kd_pos_x, float Kd_pos_y, float Kd_pos_z,
    float Kp_att_r, float Kp_att_p, float Kp_att_y,
    float Kd_att_r, float Kd_att_p, float Kd_att_y,
    float sigma_pid)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    // Load initial state
    float px = x0[0], py = x0[1], pz = x0[2];
    float vx = x0[3], vy = x0[4], vz = x0[5];
    float qw = x0[6], qx = x0[7], qy = x0[8], qz = x0[9];
    float wx = x0[10], wy = x0[11], wz = x0[12];
    float wm1 = x0[13], wm2 = x0[14], wm3 = x0[15];
    float wm4 = x0[16], wm5 = x0[17], wm6 = x0[18];
    
    float pd_x = pos_des[0], pd_y = pos_des[1], pd_z = pos_des[2];
    float qd_w = q_des[0], qd_x = q_des[1], qd_y = q_des[2], qd_z = q_des[3];
    
    float u1_last = u_prev[0], u2_last = u_prev[1], u3_last = u_prev[2];
    float u4_last = u_prev[3], u5_last = u_prev[4], u6_last = u_prev[5];
    
    float b = k_T * k_M;
    float kL = k_T * L;
    float cos_crash = cosf(crash_angle);
    float dt_half = dt * 0.5f;
    float dt_sixth = dt / 6.0f;
    
    // PID gains with noise for this rollout (if PID rollout)
    float kp_px = Kp_pos_x, kp_py = Kp_pos_y, kp_pz = Kp_pos_z;
    float kd_px = Kd_pos_x, kd_py = Kd_pos_y, kd_pz = Kd_pos_z;
    float kp_ar = Kp_att_r, kp_ap = Kp_att_p, kp_ay = Kp_att_y;
    float kd_ar = Kd_att_r, kd_ap = Kd_att_p, kd_ay = Kd_att_y;
    
    if (k < K_pid) {
        // Add noise to PID gains (12 noise values per PID rollout)
        int noise_idx = k * 12;
        kp_px *= (1.0f + sigma_pid * pid_noise[noise_idx + 0]);
        kp_py *= (1.0f + sigma_pid * pid_noise[noise_idx + 1]);
        kp_pz *= (1.0f + sigma_pid * pid_noise[noise_idx + 2]);
        kd_px *= (1.0f + sigma_pid * pid_noise[noise_idx + 3]);
        kd_py *= (1.0f + sigma_pid * pid_noise[noise_idx + 4]);
        kd_pz *= (1.0f + sigma_pid * pid_noise[noise_idx + 5]);
        kp_ar *= (1.0f + sigma_pid * pid_noise[noise_idx + 6]);
        kp_ap *= (1.0f + sigma_pid * pid_noise[noise_idx + 7]);
        kp_ay *= (1.0f + sigma_pid * pid_noise[noise_idx + 8]);
        kd_ar *= (1.0f + sigma_pid * pid_noise[noise_idx + 9]);
        kd_ap *= (1.0f + sigma_pid * pid_noise[noise_idx + 10]);
        kd_ay *= (1.0f + sigma_pid * pid_noise[noise_idx + 11]);
        
        // Ensure gains stay positive
        kp_px = fmaxf(kp_px, 0.1f); kp_py = fmaxf(kp_py, 0.1f); kp_pz = fmaxf(kp_pz, 0.1f);
        kd_px = fmaxf(kd_px, 0.1f); kd_py = fmaxf(kd_py, 0.1f); kd_pz = fmaxf(kd_pz, 0.1f);
        kp_ar = fmaxf(kp_ar, 0.1f); kp_ap = fmaxf(kp_ap, 0.1f); kp_ay = fmaxf(kp_ay, 0.1f);
        kd_ar = fmaxf(kd_ar, 0.1f); kd_ap = fmaxf(kd_ap, 0.1f); kd_ay = fmaxf(kd_ay, 0.1f);
    }
    
    float S_pos = 0.0f, S_vel = 0.0f, S_att = 0.0f;
    float S_omega = 0.0f, S_ctrl = 0.0f, S_smooth = 0.0f;
    int sat_cnt = 0;
    bool crashed = false;
    
    for (int t = 0; t < N; t++) {
        float u1, u2, u3, u4, u5, u6;
        float d1, d2, d3, d4, d5, d6;
        
        if (k < K_pid) {
            // === PD Rollout ===
            // Compute velocity in NED frame
            float R11 = 1.0f - 2.0f*(qy*qy + qz*qz);
            float R12 = 2.0f*(qx*qy - qz*qw);
            float R13 = 2.0f*(qx*qz + qy*qw);
            float R21 = 2.0f*(qx*qy + qz*qw);
            float R22 = 1.0f - 2.0f*(qx*qx + qz*qz);
            float R23 = 2.0f*(qy*qz - qx*qw);
            float R31 = 2.0f*(qx*qz - qy*qw);
            float R32 = 2.0f*(qy*qz + qx*qw);
            float R33 = 1.0f - 2.0f*(qx*qx + qy*qy);
            
            float vn = R11*vx + R12*vy + R13*vz;
            float ve = R21*vx + R22*vy + R23*vz;
            float vd = R31*vx + R32*vy + R33*vz;
            
            // Position PD
            float qw_cmd, qx_cmd, qy_cmd, qz_cmd, thrust;
            position_pd(px, py, pz, vn, ve, vd, pd_x, pd_y, pd_z, 0.0f,
                       kp_px, kp_py, kp_pz, kd_px, kd_py, kd_pz,
                       m, g, qw_cmd, qx_cmd, qy_cmd, qz_cmd, thrust);
            
            // Attitude PD
            float tau_x, tau_y, tau_z;
            attitude_pd(qw, qx, qy, qz, wx, wy, wz,
                       qw_cmd, qx_cmd, qy_cmd, qz_cmd,
                       kp_ar, kp_ap, kp_ay, kd_ar, kd_ap, kd_ay,
                       tau_x, tau_y, tau_z);
            
            // Control allocation
            control_allocator_inverse(thrust, tau_x, tau_y, tau_z,
                                     k_T, k_M, L, omega_min, omega_max,
                                     u1, u2, u3, u4, u5, u6);
            
            // Compute delta_u (difference from nominal)
            float un1 = u_seq[0 + t*6], un2 = u_seq[1 + t*6], un3 = u_seq[2 + t*6];
            float un4 = u_seq[3 + t*6], un5 = u_seq[4 + t*6], un6 = u_seq[5 + t*6];
            d1 = u1 - un1; d2 = u2 - un2; d3 = u3 - un3;
            d4 = u4 - un4; d5 = u5 - un5; d6 = u6 - un6;
            
        } else {
            // === Random Rollout ===
            d1 = noise[0*N*K + t*K + k];
            d2 = noise[1*N*K + t*K + k];
            d3 = noise[2*N*K + t*K + k];
            d4 = noise[3*N*K + t*K + k];
            d5 = noise[4*N*K + t*K + k];
            d6 = noise[5*N*K + t*K + k];
            
            float un1 = u_seq[0 + t*6], un2 = u_seq[1 + t*6], un3 = u_seq[2 + t*6];
            float un4 = u_seq[3 + t*6], un5 = u_seq[4 + t*6], un6 = u_seq[5 + t*6];
            
            float u1_raw = un1 + d1, u2_raw = un2 + d2, u3_raw = un3 + d3;
            float u4_raw = un4 + d4, u5_raw = un5 + d5, u6_raw = un6 + d6;
            
            if (u1_raw < omega_min || u1_raw > omega_max) sat_cnt++;
            if (u2_raw < omega_min || u2_raw > omega_max) sat_cnt++;
            if (u3_raw < omega_min || u3_raw > omega_max) sat_cnt++;
            if (u4_raw < omega_min || u4_raw > omega_max) sat_cnt++;
            if (u5_raw < omega_min || u5_raw > omega_max) sat_cnt++;
            if (u6_raw < omega_min || u6_raw > omega_max) sat_cnt++;
            
            u1 = fminf(fmaxf(u1_raw, omega_min), omega_max);
            u2 = fminf(fmaxf(u2_raw, omega_min), omega_max);
            u3 = fminf(fmaxf(u3_raw, omega_min), omega_max);
            u4 = fminf(fmaxf(u4_raw, omega_min), omega_max);
            u5 = fminf(fmaxf(u5_raw, omega_min), omega_max);
            u6 = fminf(fmaxf(u6_raw, omega_min), omega_max);
        }
        
        // Store delta_u
        du_all[0*N*K + t*K + k] = d1;
        du_all[1*N*K + t*K + k] = d2;
        du_all[2*N*K + t*K + k] = d3;
        du_all[3*N*K + t*K + k] = d4;
        du_all[4*N*K + t*K + k] = d5;
        du_all[5*N*K + t*K + k] = d6;
        
        // ===== COST FUNCTION =====
        float ex = px - pd_x, ey = py - pd_y, ez = pz - pd_z;
        float cost_pos = w_pos * (ex*ex + ey*ey + ez*ez);
        
        float R11 = 1.0f - 2.0f*(qy*qy + qz*qz);
        float R12 = 2.0f*(qx*qy - qz*qw);
        float R13 = 2.0f*(qx*qz + qy*qw);
        float R21 = 2.0f*(qx*qy + qz*qw);
        float R22 = 1.0f - 2.0f*(qx*qx + qz*qz);
        float R23 = 2.0f*(qy*qz - qx*qw);
        float R31 = 2.0f*(qx*qz - qy*qw);
        float R32 = 2.0f*(qy*qz + qx*qw);
        float R33 = 1.0f - 2.0f*(qx*qx + qy*qy);
        
        float vn = R11*vx + R12*vy + R13*vz;
        float ve = R21*vx + R22*vy + R23*vz;
        float vd = R31*vx + R32*vy + R33*vz;
        float cost_vel = w_vel * (vn*vn + ve*ve + vd*vd);
        
        float q_dot = qd_w*qw + qd_x*qx + qd_y*qy + qd_z*qz;
        float cost_att = w_att * (1.0f - q_dot*q_dot);
        
        if (R33 < cos_crash && !crashed) crashed = true;
        
        float cost_omega = w_omega * (wx*wx + wy*wy + wz*wz);
        
        float u_sq = u1*u1 + u2*u2 + u3*u3 + u4*u4 + u5*u5 + u6*u6;
        float cost_ctrl = R_val * u_sq;
        
        float du1 = u1 - u1_last, du2 = u2 - u2_last, du3 = u3 - u3_last;
        float du4 = u4 - u4_last, du5 = u5 - u5_last, du6 = u6 - u6_last;
        float cost_smooth = w_smooth * (du1*du1 + du2*du2 + du3*du3 + du4*du4 + du5*du5 + du6*du6);
        
        u1_last = u1; u2_last = u2; u3_last = u3;
        u4_last = u4; u5_last = u5; u6_last = u6;
        
        S_pos += cost_pos * dt;
        S_vel += cost_vel * dt;
        S_att += cost_att * dt;
        S_omega += cost_omega * dt;
        S_ctrl += cost_ctrl * dt;
        S_smooth += cost_smooth * dt;
        
        // ===== Motor Dynamics (Euler) =====
        float tau1 = (u1 >= wm1) ? tau_up : tau_down;
        float tau2 = (u2 >= wm2) ? tau_up : tau_down;
        float tau3 = (u3 >= wm3) ? tau_up : tau_down;
        float tau4 = (u4 >= wm4) ? tau_up : tau_down;
        float tau5 = (u5 >= wm5) ? tau_up : tau_down;
        float tau6 = (u6 >= wm6) ? tau_up : tau_down;
        
        wm1 += (u1 - wm1) * dt / tau1;
        wm2 += (u2 - wm2) * dt / tau2;
        wm3 += (u3 - wm3) * dt / tau3;
        wm4 += (u4 - wm4) * dt / tau4;
        wm5 += (u5 - wm5) * dt / tau5;
        wm6 += (u6 - wm6) * dt / tau6;
        
        wm1 = fminf(fmaxf(wm1, omega_min), omega_max);
        wm2 = fminf(fmaxf(wm2, omega_min), omega_max);
        wm3 = fminf(fmaxf(wm3, omega_min), omega_max);
        wm4 = fminf(fmaxf(wm4, omega_min), omega_max);
        wm5 = fminf(fmaxf(wm5, omega_min), omega_max);
        wm6 = fminf(fmaxf(wm6, omega_min), omega_max);
        
        // ===== Rigid Body Dynamics (RK4) =====
        float k1_px, k1_py, k1_pz, k1_vx, k1_vy, k1_vz;
        float k1_qw, k1_qx, k1_qy, k1_qz, k1_wx, k1_wy, k1_wz;
        float k2_px, k2_py, k2_pz, k2_vx, k2_vy, k2_vz;
        float k2_qw, k2_qx, k2_qy, k2_qz, k2_wx, k2_wy, k2_wz;
        float k3_px, k3_py, k3_pz, k3_vx, k3_vy, k3_vz;
        float k3_qw, k3_qx, k3_qy, k3_qz, k3_wx, k3_wy, k3_wz;
        float k4_px, k4_py, k4_pz, k4_vx, k4_vy, k4_vz;
        float k4_qw, k4_qx, k4_qy, k4_qz, k4_wx, k4_wy, k4_wz;
        
        compute_derivatives(px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz,
                           wm1, wm2, wm3, wm4, wm5, wm6,
                           m, Jxx, Jyy, Jzz, g, k_T, kL, b,
                           k1_px, k1_py, k1_pz, k1_vx, k1_vy, k1_vz,
                           k1_qw, k1_qx, k1_qy, k1_qz, k1_wx, k1_wy, k1_wz);
        
        float px2 = px + dt_half*k1_px, py2 = py + dt_half*k1_py, pz2 = pz + dt_half*k1_pz;
        float vx2 = vx + dt_half*k1_vx, vy2 = vy + dt_half*k1_vy, vz2 = vz + dt_half*k1_vz;
        float qw2 = qw + dt_half*k1_qw, qx2 = qx + dt_half*k1_qx;
        float qy2 = qy + dt_half*k1_qy, qz2 = qz + dt_half*k1_qz;
        float wx2 = wx + dt_half*k1_wx, wy2 = wy + dt_half*k1_wy, wz2 = wz + dt_half*k1_wz;
        float qn2 = rsqrtf(qw2*qw2 + qx2*qx2 + qy2*qy2 + qz2*qz2);
        qw2 *= qn2; qx2 *= qn2; qy2 *= qn2; qz2 *= qn2;
        
        compute_derivatives(px2, py2, pz2, vx2, vy2, vz2, qw2, qx2, qy2, qz2, wx2, wy2, wz2,
                           wm1, wm2, wm3, wm4, wm5, wm6,
                           m, Jxx, Jyy, Jzz, g, k_T, kL, b,
                           k2_px, k2_py, k2_pz, k2_vx, k2_vy, k2_vz,
                           k2_qw, k2_qx, k2_qy, k2_qz, k2_wx, k2_wy, k2_wz);
        
        float px3 = px + dt_half*k2_px, py3 = py + dt_half*k2_py, pz3 = pz + dt_half*k2_pz;
        float vx3 = vx + dt_half*k2_vx, vy3 = vy + dt_half*k2_vy, vz3 = vz + dt_half*k2_vz;
        float qw3 = qw + dt_half*k2_qw, qx3 = qx + dt_half*k2_qx;
        float qy3 = qy + dt_half*k2_qy, qz3 = qz + dt_half*k2_qz;
        float wx3 = wx + dt_half*k2_wx, wy3 = wy + dt_half*k2_wy, wz3 = wz + dt_half*k2_wz;
        float qn3 = rsqrtf(qw3*qw3 + qx3*qx3 + qy3*qy3 + qz3*qz3);
        qw3 *= qn3; qx3 *= qn3; qy3 *= qn3; qz3 *= qn3;
        
        compute_derivatives(px3, py3, pz3, vx3, vy3, vz3, qw3, qx3, qy3, qz3, wx3, wy3, wz3,
                           wm1, wm2, wm3, wm4, wm5, wm6,
                           m, Jxx, Jyy, Jzz, g, k_T, kL, b,
                           k3_px, k3_py, k3_pz, k3_vx, k3_vy, k3_vz,
                           k3_qw, k3_qx, k3_qy, k3_qz, k3_wx, k3_wy, k3_wz);
        
        float px4 = px + dt*k3_px, py4 = py + dt*k3_py, pz4 = pz + dt*k3_pz;
        float vx4 = vx + dt*k3_vx, vy4 = vy + dt*k3_vy, vz4 = vz + dt*k3_vz;
        float qw4 = qw + dt*k3_qw, qx4 = qx + dt*k3_qx;
        float qy4 = qy + dt*k3_qy, qz4 = qz + dt*k3_qz;
        float wx4 = wx + dt*k3_wx, wy4 = wy + dt*k3_wy, wz4 = wz + dt*k3_wz;
        float qn4 = rsqrtf(qw4*qw4 + qx4*qx4 + qy4*qy4 + qz4*qz4);
        qw4 *= qn4; qx4 *= qn4; qy4 *= qn4; qz4 *= qn4;
        
        compute_derivatives(px4, py4, pz4, vx4, vy4, vz4, qw4, qx4, qy4, qz4, wx4, wy4, wz4,
                           wm1, wm2, wm3, wm4, wm5, wm6,
                           m, Jxx, Jyy, Jzz, g, k_T, kL, b,
                           k4_px, k4_py, k4_pz, k4_vx, k4_vy, k4_vz,
                           k4_qw, k4_qx, k4_qy, k4_qz, k4_wx, k4_wy, k4_wz);
        
        px += dt_sixth * (k1_px + 2.0f*k2_px + 2.0f*k3_px + k4_px);
        py += dt_sixth * (k1_py + 2.0f*k2_py + 2.0f*k3_py + k4_py);
        pz += dt_sixth * (k1_pz + 2.0f*k2_pz + 2.0f*k3_pz + k4_pz);
        vx += dt_sixth * (k1_vx + 2.0f*k2_vx + 2.0f*k3_vx + k4_vx);
        vy += dt_sixth * (k1_vy + 2.0f*k2_vy + 2.0f*k3_vy + k4_vy);
        vz += dt_sixth * (k1_vz + 2.0f*k2_vz + 2.0f*k3_vz + k4_vz);
        qw += dt_sixth * (k1_qw + 2.0f*k2_qw + 2.0f*k3_qw + k4_qw);
        qx += dt_sixth * (k1_qx + 2.0f*k2_qx + 2.0f*k3_qx + k4_qx);
        qy += dt_sixth * (k1_qy + 2.0f*k2_qy + 2.0f*k3_qy + k4_qy);
        qz += dt_sixth * (k1_qz + 2.0f*k2_qz + 2.0f*k3_qz + k4_qz);
        wx += dt_sixth * (k1_wx + 2.0f*k2_wx + 2.0f*k3_wx + k4_wx);
        wy += dt_sixth * (k1_wy + 2.0f*k2_wy + 2.0f*k3_wy + k4_wy);
        wz += dt_sixth * (k1_wz + 2.0f*k2_wz + 2.0f*k3_wz + k4_wz);
        
        float qnorm = rsqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        qw *= qnorm; qx *= qnorm; qy *= qnorm; qz *= qnorm;
    }
    
    // Terminal cost
    float ex_T = px - pd_x, ey_T = py - pd_y, ez_T = pz - pd_z;
    float cost_pos_T = w_terminal * (ex_T*ex_T + ey_T*ey_T + ez_T*ez_T);
    
    float q_dot_T = qd_w*qw + qd_x*qx + qd_y*qy + qd_z*qz;
    float cost_att_T = w_att * (1.0f - q_dot_T*q_dot_T);
    
    float R11_T = 1.0f - 2.0f*(qy*qy + qz*qz);
    float R12_T = 2.0f*(qx*qy - qz*qw);
    float R13_T = 2.0f*(qx*qz + qy*qw);
    float R21_T = 2.0f*(qx*qy + qz*qw);
    float R22_T = 1.0f - 2.0f*(qx*qx + qz*qz);
    float R23_T = 2.0f*(qy*qz - qx*qw);
    float R31_T = 2.0f*(qx*qz - qy*qw);
    float R32_T = 2.0f*(qy*qz + qx*qw);
    float R33_T = 1.0f - 2.0f*(qx*qx + qy*qy);
    
    float vn_T = R11_T*vx + R12_T*vy + R13_T*vz;
    float ve_T = R21_T*vx + R22_T*vy + R23_T*vz;
    float vd_T = R31_T*vx + R32_T*vy + R33_T*vz;
    float cost_vel_T = 0.5f * w_terminal * (vn_T*vn_T + ve_T*ve_T + vd_T*vd_T);
    
    float S_total = S_pos + S_vel + S_att + S_omega + S_ctrl + S_smooth;
    S_total += cost_pos_T + cost_att_T + cost_vel_T;
    
    if (crashed) S_total += crash_cost;
    
    costs[k] = S_total;
    
    cost_breakdown[0*K + k] = S_pos;
    cost_breakdown[1*K + k] = S_vel;
    cost_breakdown[2*K + k] = S_att;
    cost_breakdown[3*K + k] = S_omega;
    cost_breakdown[4*K + k] = S_ctrl;
    cost_breakdown[5*K + k] = S_smooth + (crashed ? crash_cost : 0.0f);
    
    sat_count[k] = sat_cnt;
}

__global__ void compute_weights_kernel(
    const float* __restrict__ costs,
    float* __restrict__ weights,
    float min_cost, float lambda, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    weights[k] = expf(-(costs[k] - min_cost) / lambda);
}

__global__ void weighted_sum_kernel(
    const float* __restrict__ du_all,
    const float* __restrict__ weights,
    float* __restrict__ du_weighted,
    float weight_sum, int N, int K)
{
    int t = blockIdx.x;
    int m = threadIdx.x;
    if (t >= N || m >= 6) return;
    
    float sum = 0.0f;
    for (int kk = 0; kk < K; kk++) {
        sum += du_all[m*N*K + t*K + kk] * weights[kk];
    }
    du_weighted[m + t*6] = sum / weight_sum;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("MPPI:nrhs", "Seven inputs required.");
    }
    
    float* h_x0 = (float*)mxGetData(prhs[0]);
    float* h_u_seq = (float*)mxGetData(prhs[1]);
    float* h_u_prev = (float*)mxGetData(prhs[2]);
    float* h_pos_des = (float*)mxGetData(prhs[3]);
    float* h_q_des = (float*)mxGetData(prhs[4]);
    
    int N = (int)mxGetN(prhs[1]);
    
    float m = (float)mxGetScalar(mxGetField(prhs[5], 0, "m"));
    float Jxx = (float)mxGetScalar(mxGetField(prhs[5], 0, "Jxx"));
    float Jyy = (float)mxGetScalar(mxGetField(prhs[5], 0, "Jyy"));
    float Jzz = (float)mxGetScalar(mxGetField(prhs[5], 0, "Jzz"));
    float g = (float)mxGetScalar(mxGetField(prhs[5], 0, "g"));
    float k_T = (float)mxGetScalar(mxGetField(prhs[5], 0, "k_T"));
    float k_M = (float)mxGetScalar(mxGetField(prhs[5], 0, "k_M"));
    float L = (float)mxGetScalar(mxGetField(prhs[5], 0, "L"));
    float omega_max = (float)mxGetScalar(mxGetField(prhs[5], 0, "omega_max"));
    float omega_min = (float)mxGetScalar(mxGetField(prhs[5], 0, "omega_min"));
    float tau_up = (float)mxGetScalar(mxGetField(prhs[5], 0, "tau_up"));
    float tau_down = (float)mxGetScalar(mxGetField(prhs[5], 0, "tau_down"));
    
    int K = (int)mxGetScalar(mxGetField(prhs[6], 0, "K"));
    int K_pid = (int)mxGetScalar(mxGetField(prhs[6], 0, "K_pid"));
    float dt = (float)mxGetScalar(mxGetField(prhs[6], 0, "dt"));
    float lambda = (float)mxGetScalar(mxGetField(prhs[6], 0, "lambda"));
    float nu = (float)mxGetScalar(mxGetField(prhs[6], 0, "nu"));
    float sigma = (float)mxGetScalar(mxGetField(prhs[6], 0, "sigma"));
    
    float w_pos = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_pos"));
    float w_vel = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_vel"));
    float w_att = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_att"));
    float w_omega = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_omega"));
    float w_terminal = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_terminal"));
    float w_smooth = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_smooth"));
    float R_val = (float)mxGetScalar(mxGetField(prhs[6], 0, "R"));
    
    float crash_cost = (float)mxGetScalar(mxGetField(prhs[6], 0, "crash_cost"));
    float crash_angle = (float)mxGetScalar(mxGetField(prhs[6], 0, "crash_angle"));
    
    // PID gains
    double* Kp_pos = mxGetPr(mxGetField(prhs[6], 0, "Kp_pos"));
    double* Kd_pos = mxGetPr(mxGetField(prhs[6], 0, "Kd_pos"));
    double* Kp_att = mxGetPr(mxGetField(prhs[6], 0, "Kp_att"));
    double* Kd_att = mxGetPr(mxGetField(prhs[6], 0, "Kd_att"));
    float sigma_pid = (float)mxGetScalar(mxGetField(prhs[6], 0, "sigma_pid"));
    
    float Kp_pos_x = (float)Kp_pos[0], Kp_pos_y = (float)Kp_pos[1], Kp_pos_z = (float)Kp_pos[2];
    float Kd_pos_x = (float)Kd_pos[0], Kd_pos_y = (float)Kd_pos[1], Kd_pos_z = (float)Kd_pos[2];
    float Kp_att_r = (float)Kp_att[0], Kp_att_p = (float)Kp_att[1], Kp_att_y = (float)Kp_att[2];
    float Kd_att_r = (float)Kd_att[0], Kd_att_p = (float)Kd_att[1], Kd_att_y = (float)Kd_att[2];
    
    float nu_c = 1.0f - 1.0f / nu;
    
    if (!initialized || K != K_alloc || N != N_alloc) {
        if (initialized) cleanup();
        
        cudaMalloc(&d_x0, 19 * sizeof(float));
        cudaMalloc(&d_u_seq, 6 * N * sizeof(float));
        cudaMalloc(&d_u_prev, 6 * sizeof(float));
        cudaMalloc(&d_pos_des, 3 * sizeof(float));
        cudaMalloc(&d_q_des, 4 * sizeof(float));
        cudaMalloc(&d_noise, 6 * N * K * sizeof(float));
        cudaMalloc(&d_pid_noise, 12 * K_pid * sizeof(float));
        cudaMalloc(&d_costs, K * sizeof(float));
        cudaMalloc(&d_weights, K * sizeof(float));
        cudaMalloc(&d_du_all, 6 * N * K * sizeof(float));
        cudaMalloc(&d_du_weighted, 6 * N * sizeof(float));
        cudaMalloc(&d_cost_breakdown, 6 * K * sizeof(float));
        cudaMalloc(&d_sat_count, K * sizeof(int));
        
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 42);
        
        mexAtExit(cleanup);
        initialized = true;
        K_alloc = K;
        N_alloc = N;
    }
    
    cudaMemcpy(d_x0, h_x0, 19 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_seq, h_u_seq, 6 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_prev, h_u_prev, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_des, h_pos_des, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_des, h_q_des, 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Generate noise for random rollouts
    curandGenerateNormal(gen, d_noise, 6 * N * K, 0.0f, 1.0f);
    int bs = 256;
    int gs = (6 * N * K + bs - 1) / bs;
    scale_noise_kernel<<<gs, bs>>>(d_noise, sigma, 6 * N * K);
    
    // Generate noise for PID gains
    if (K_pid > 0) {
        curandGenerateNormal(gen, d_pid_noise, 12 * K_pid, 0.0f, 1.0f);
    }
    
    gs = (K + bs - 1) / bs;
    hexa_mppi_rollout_kernel<<<gs, bs>>>(
        d_x0, d_u_seq, d_u_prev, d_pos_des, d_q_des, d_noise, d_pid_noise,
        d_costs, d_du_all, d_cost_breakdown, d_sat_count,
        m, Jxx, Jyy, Jzz, g, k_T, k_M, L, omega_max, omega_min,
        tau_up, tau_down,
        dt, N, K, K_pid,
        w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth, R_val,
        crash_cost, crash_angle, lambda, nu_c,
        Kp_pos_x, Kp_pos_y, Kp_pos_z, Kd_pos_x, Kd_pos_y, Kd_pos_z,
        Kp_att_r, Kp_att_p, Kp_att_y, Kd_att_r, Kd_att_p, Kd_att_y,
        sigma_pid
    );
    
    float* h_costs = (float*)mxMalloc(K * sizeof(float));
    float* h_cost_bd = (float*)mxMalloc(6 * K * sizeof(float));
    int* h_sat_count = (int*)mxMalloc(K * sizeof(int));
    
    cudaMemcpy(h_costs, d_costs, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost_bd, d_cost_breakdown, 6 * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sat_count, d_sat_count, K * sizeof(int), cudaMemcpyDeviceToHost);
    
    float min_cost = h_costs[0];
    float sum_cost = h_costs[0];
    int min_idx = 0;
    for (int kk = 1; kk < K; kk++) {
        if (h_costs[kk] < min_cost) {
            min_cost = h_costs[kk];
            min_idx = kk;
        }
        sum_cost += h_costs[kk];
    }
    float avg_cost = sum_cost / K;
    
    int total_sat = 0;
    for (int kk = 0; kk < K; kk++) total_sat += h_sat_count[kk];
    float sat_ratio = (float)total_sat / (float)(6 * N * K);
    
    compute_weights_kernel<<<gs, bs>>>(d_costs, d_weights, min_cost, lambda, K);
    
    float* h_weights = (float*)mxMalloc(K * sizeof(float));
    cudaMemcpy(h_weights, d_weights, K * sizeof(float), cudaMemcpyDeviceToHost);
    
    float wsum = 0.0f;
    float wsum_sq = 0.0f;
    for (int kk = 0; kk < K; kk++) {
        wsum += h_weights[kk];
        wsum_sq += h_weights[kk] * h_weights[kk];
    }
    if (wsum < 1e-10f) wsum = 1.0f;
    
    float ess = (wsum * wsum) / (wsum_sq + 1e-10f);
    
    weighted_sum_kernel<<<N, 6>>>(d_du_all, d_weights, d_du_weighted, wsum, N, K);
    
    float* h_du = (float*)mxMalloc(6 * N * sizeof(float));
    cudaMemcpy(h_du, d_du_weighted, 6 * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float* h_u_new = (float*)mxMalloc(6 * N * sizeof(float));
    for (int t = 0; t < N; t++) {
        for (int mi = 0; mi < 6; mi++) {
            float u_updated = h_u_seq[mi + t*6] + h_du[mi + t*6];
            h_u_new[mi + t*6] = fminf(fmaxf(u_updated, omega_min), omega_max);
        }
    }
    
    plhs[0] = mxCreateNumericMatrix(6, 1, mxSINGLE_CLASS, mxREAL);
    float* out_u = (float*)mxGetData(plhs[0]);
    for (int mi = 0; mi < 6; mi++) out_u[mi] = h_u_new[mi];
    
    plhs[1] = mxCreateNumericMatrix(6, N, mxSINGLE_CLASS, mxREAL);
    memcpy(mxGetData(plhs[1]), h_u_new, 6 * N * sizeof(float));
    
    if (nlhs > 2) {
        const char* field_names[] = {"min_cost", "avg_cost", "cost_breakdown", 
                                     "saturation_ratio", "effective_sample_size",
                                     "best_is_pid"};
        plhs[2] = mxCreateStructMatrix(1, 1, 6, field_names);
        
        mxSetField(plhs[2], 0, "min_cost", mxCreateDoubleScalar((double)min_cost));
        mxSetField(plhs[2], 0, "avg_cost", mxCreateDoubleScalar((double)avg_cost));
        
        mxArray* cb = mxCreateDoubleMatrix(6, 1, mxREAL);
        double* cb_data = mxGetPr(cb);
        for (int i = 0; i < 6; i++) {
            cb_data[i] = (double)h_cost_bd[i*K + min_idx];
        }
        mxSetField(plhs[2], 0, "cost_breakdown", cb);
        
        mxSetField(plhs[2], 0, "saturation_ratio", mxCreateDoubleScalar((double)sat_ratio));
        mxSetField(plhs[2], 0, "effective_sample_size", mxCreateDoubleScalar((double)ess));
        mxSetField(plhs[2], 0, "best_is_pid", mxCreateDoubleScalar((double)(min_idx < K_pid ? 1 : 0)));
    }
    
    mxFree(h_costs);
    mxFree(h_cost_bd);
    mxFree(h_sat_count);
    mxFree(h_weights);
    mxFree(h_du);
    mxFree(h_u_new);
}
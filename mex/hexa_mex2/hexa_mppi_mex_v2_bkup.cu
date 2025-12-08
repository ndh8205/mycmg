/*
 * hexa_mppi_mex_v2.cu - Hexarotor MPPI Controller (CUDA MEX)
 * 
 * Improved cost function:
 *   - Quadratic attitude cost (replaces log barrier)
 *   - Quaternion error based attitude tracking
 *   - Control smoothness penalty
 *   - Crash penalty for large tilt angles
 *   - Complete terminal cost
 * 
 * State: 19x1 [pos(3), vel(3), quat(4), omega(3), omega_motor(6)]
 * Control: 6x1 [omega_1, ..., omega_6] motor speed commands [rad/s]
 * 
 * Compile: mexcuda hexa_mppi_mex_v2.cu -lcurand
 * Usage:   [u_opt, u_seq_new, stats] = hexa_mppi_mex_v2(x0, u_seq, u_prev, pos_des, q_des, params, mppi_params)
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
static int *d_sat_count = nullptr;
static curandGenerator_t gen;

void cleanup(void) {
    if (initialized) {
        cudaFree(d_x0); cudaFree(d_u_seq); cudaFree(d_u_prev);
        cudaFree(d_pos_des); cudaFree(d_q_des);
        cudaFree(d_noise); cudaFree(d_costs); cudaFree(d_weights);
        cudaFree(d_du_all); cudaFree(d_du_weighted);
        cudaFree(d_cost_breakdown); cudaFree(d_sat_count);
        curandDestroyGenerator(gen);
        initialized = false;
    }
}

// Scale noise kernel
__global__ void scale_noise_kernel(float* noise, float sigma, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 6 * N * K;
    if (idx < total) {
        noise[idx] *= sigma;
    }
}

// Main MPPI rollout kernel - one thread per sample
__global__ void hexa_mppi_rollout_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ u_seq,
    const float* __restrict__ u_prev,      // Previous control (6x1) for smoothness
    const float* __restrict__ pos_des,
    const float* __restrict__ q_des,       // Desired quaternion (4x1)
    const float* __restrict__ noise,
    float* __restrict__ costs,
    float* __restrict__ du_all,
    float* __restrict__ cost_breakdown,
    int* __restrict__ sat_count,
    // Drone parameters
    float m, float Jxx, float Jyy, float Jzz, float g,
    float k_T, float k_M, float L,
    float omega_max, float omega_min,
    float tau_up, float tau_down,
    // MPPI parameters
    float dt, int N, int K,
    // Cost weights (simplified)
    float w_pos, float w_vel, float w_att, float w_omega,
    float w_terminal, float w_smooth, float R_val,
    float crash_cost, float crash_angle,
    float lambda, float nu_c)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    // Load initial state (19 states)
    float px = x0[0], py = x0[1], pz = x0[2];
    float vx = x0[3], vy = x0[4], vz = x0[5];
    float qw = x0[6], qx = x0[7], qy = x0[8], qz = x0[9];
    float wx = x0[10], wy = x0[11], wz = x0[12];
    
    // Motor states
    float wm1 = x0[13], wm2 = x0[14], wm3 = x0[15];
    float wm4 = x0[16], wm5 = x0[17], wm6 = x0[18];
    
    // Desired states
    float pd_x = pos_des[0], pd_y = pos_des[1], pd_z = pos_des[2];
    float qd_w = q_des[0], qd_x = q_des[1], qd_y = q_des[2], qd_z = q_des[3];
    
    // Previous control for smoothness (t=0)
    float u1_last = u_prev[0], u2_last = u_prev[1], u3_last = u_prev[2];
    float u4_last = u_prev[3], u5_last = u_prev[4], u6_last = u_prev[5];
    
    // Constants
    float b = k_T * k_M;
    float s3 = 0.866025403784f;
    float kL = k_T * L;
    float cos_crash = cosf(crash_angle);  // Pre-compute crash threshold
    
    // Accumulated costs by component
    float S_pos = 0.0f, S_vel = 0.0f, S_att = 0.0f;
    float S_omega = 0.0f, S_ctrl = 0.0f, S_smooth = 0.0f;
    int sat_cnt = 0;
    bool crashed = false;
    
    for (int t = 0; t < N; t++) {
        // Get noise for this timestep
        float d1 = noise[0*N*K + t*K + k];
        float d2 = noise[1*N*K + t*K + k];
        float d3 = noise[2*N*K + t*K + k];
        float d4 = noise[3*N*K + t*K + k];
        float d5 = noise[4*N*K + t*K + k];
        float d6 = noise[5*N*K + t*K + k];
        
        // Store delta_u for weighted average
        du_all[0*N*K + t*K + k] = d1;
        du_all[1*N*K + t*K + k] = d2;
        du_all[2*N*K + t*K + k] = d3;
        du_all[3*N*K + t*K + k] = d4;
        du_all[4*N*K + t*K + k] = d5;
        du_all[5*N*K + t*K + k] = d6;
        
        // Nominal control
        float un1 = u_seq[0 + t*6], un2 = u_seq[1 + t*6], un3 = u_seq[2 + t*6];
        float un4 = u_seq[3 + t*6], un5 = u_seq[4 + t*6], un6 = u_seq[5 + t*6];
        
        // Motor speed commands (with noise)
        float u1_raw = un1 + d1, u2_raw = un2 + d2, u3_raw = un3 + d3;
        float u4_raw = un4 + d4, u5_raw = un5 + d5, u6_raw = un6 + d6;
        
        // Check saturation
        if (u1_raw < omega_min || u1_raw > omega_max) sat_cnt++;
        if (u2_raw < omega_min || u2_raw > omega_max) sat_cnt++;
        if (u3_raw < omega_min || u3_raw > omega_max) sat_cnt++;
        if (u4_raw < omega_min || u4_raw > omega_max) sat_cnt++;
        if (u5_raw < omega_min || u5_raw > omega_max) sat_cnt++;
        if (u6_raw < omega_min || u6_raw > omega_max) sat_cnt++;
        
        // Saturate commands
        float u1 = fminf(fmaxf(u1_raw, omega_min), omega_max);
        float u2 = fminf(fmaxf(u2_raw, omega_min), omega_max);
        float u3 = fminf(fmaxf(u3_raw, omega_min), omega_max);
        float u4 = fminf(fmaxf(u4_raw, omega_min), omega_max);
        float u5 = fminf(fmaxf(u5_raw, omega_min), omega_max);
        float u6 = fminf(fmaxf(u6_raw, omega_min), omega_max);
        
        // ===== IMPROVED COST FUNCTION =====
        
        // --- Position Cost (Quadratic) ---
        float ex = px - pd_x, ey = py - pd_y, ez = pz - pd_z;
        float cost_pos = w_pos * (ex*ex + ey*ey + ez*ez);
        
        // --- Velocity Cost (NED frame for intuitive tuning) ---
        // Transform body velocity to NED
        float R11 = 1.0f - 2.0f*(qy*qy + qz*qz);
        float R12 = 2.0f*(qx*qy - qz*qw);
        float R13 = 2.0f*(qx*qz + qy*qw);
        float R21 = 2.0f*(qx*qy + qz*qw);
        float R22 = 1.0f - 2.0f*(qx*qx + qz*qz);
        float R23 = 2.0f*(qy*qz - qx*qw);
        float R31 = 2.0f*(qx*qz - qy*qw);
        float R32 = 2.0f*(qy*qz + qx*qw);
        float R33 = 1.0f - 2.0f*(qx*qx + qy*qy);
        
        float vn = R11*vx + R12*vy + R13*vz;  // North velocity
        float ve = R21*vx + R22*vy + R23*vz;  // East velocity
        float vd = R31*vx + R32*vy + R33*vz;  // Down velocity
        float cost_vel = w_vel * (vn*vn + ve*ve + vd*vd);
        
        // --- Attitude Cost (Quaternion Error - QUADRATIC) ---
        // q_err = q_des^{-1} * q = conj(q_des) * q
        // For attitude tracking: cost = 1 - (q_des . q)^2
        // This is geodesic distance squared on SO(3)
        float q_dot = qd_w*qw + qd_x*qx + qd_y*qy + qd_z*qz;  // Inner product
        float cost_att = w_att * (1.0f - q_dot*q_dot);
        
        // --- Crash Detection ---
        // R33 = cos(tilt angle from vertical)
        // If R33 < cos(crash_angle), drone is tilted too much
        if (R33 < cos_crash && !crashed) {
            crashed = true;
        }
        
        // --- Angular Velocity Cost ---
        float cost_omega = w_omega * (wx*wx + wy*wy + wz*wz);
        
        // --- Control Effort Cost (Simplified) ---
        float u_sq = u1*u1 + u2*u2 + u3*u3 + u4*u4 + u5*u5 + u6*u6;
        float cost_ctrl = R_val * u_sq;
        
        // --- Smoothness Cost (Control Rate) ---
        float du1 = u1 - u1_last, du2 = u2 - u2_last, du3 = u3 - u3_last;
        float du4 = u4 - u4_last, du5 = u5 - u5_last, du6 = u6 - u6_last;
        float cost_smooth = w_smooth * (du1*du1 + du2*du2 + du3*du3 + du4*du4 + du5*du5 + du6*du6);
        
        // Update last control for next timestep
        u1_last = u1; u2_last = u2; u3_last = u3;
        u4_last = u4; u5_last = u5; u6_last = u6;
        
        // Accumulate running costs
        S_pos += cost_pos * dt;
        S_vel += cost_vel * dt;
        S_att += cost_att * dt;
        S_omega += cost_omega * dt;
        S_ctrl += cost_ctrl * dt;
        S_smooth += cost_smooth * dt;
        
        // ===== Motor Dynamics =====
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
        
        // Saturate motor speeds
        wm1 = fminf(fmaxf(wm1, omega_min), omega_max);
        wm2 = fminf(fmaxf(wm2, omega_min), omega_max);
        wm3 = fminf(fmaxf(wm3, omega_min), omega_max);
        wm4 = fminf(fmaxf(wm4, omega_min), omega_max);
        wm5 = fminf(fmaxf(wm5, omega_min), omega_max);
        wm6 = fminf(fmaxf(wm6, omega_min), omega_max);
        
        // ===== Rigid Body Dynamics =====
        float w1sq = wm1*wm1, w2sq = wm2*wm2, w3sq = wm3*wm3;
        float w4sq = wm4*wm4, w5sq = wm5*wm5, w6sq = wm6*wm6;
        
        float T = k_T * (w1sq + w2sq + w3sq + w4sq + w5sq + w6sq);
        float tau_x = kL * (-0.5f*w1sq + 0.5f*w2sq + w3sq + 0.5f*w4sq - 0.5f*w5sq - w6sq);
        float tau_y = kL * s3 * (w1sq + w2sq - w4sq - w5sq);
        float tau_z = b * (w1sq - w2sq + w3sq - w4sq + w5sq - w6sq);
        
        // Position derivative
        float px_dot = R11*vx + R12*vy + R13*vz;
        float py_dot = R21*vx + R22*vy + R23*vz;
        float pz_dot = R31*vx + R32*vy + R33*vz;
        
        // Velocity derivative
        float gx_b = R31 * g;
        float gy_b = R32 * g;
        float gz_b = R33 * g;
        
        float cross_x = wy*vz - wz*vy;
        float cross_y = wz*vx - wx*vz;
        float cross_z = wx*vy - wy*vx;
        
        float vx_dot = gx_b - cross_x;
        float vy_dot = gy_b - cross_y;
        float vz_dot = -T/m + gz_b - cross_z;
        
        // Quaternion derivative
        float qw_dot = 0.5f * (-wx*qx - wy*qy - wz*qz);
        float qx_dot = 0.5f * ( wx*qw + wz*qy - wy*qz);
        float qy_dot = 0.5f * ( wy*qw - wz*qx + wx*qz);
        float qz_dot = 0.5f * ( wz*qw + wy*qx - wx*qy);
        
        // Angular velocity derivative
        float Jw_x = Jxx * wx;
        float Jw_y = Jyy * wy;
        float Jw_z = Jzz * wz;
        
        float wx_dot = (tau_x - (wy*Jw_z - wz*Jw_y)) / Jxx;
        float wy_dot = (tau_y - (wz*Jw_x - wx*Jw_z)) / Jyy;
        float wz_dot = (tau_z - (wx*Jw_y - wy*Jw_x)) / Jzz;
        
        // Euler integration
        px += px_dot * dt;
        py += py_dot * dt;
        pz += pz_dot * dt;
        vx += vx_dot * dt;
        vy += vy_dot * dt;
        vz += vz_dot * dt;
        qw += qw_dot * dt;
        qx += qx_dot * dt;
        qy += qy_dot * dt;
        qz += qz_dot * dt;
        wx += wx_dot * dt;
        wy += wy_dot * dt;
        wz += wz_dot * dt;
        
        // Normalize quaternion
        float qnorm = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        qw /= qnorm; qx /= qnorm; qy /= qnorm; qz /= qnorm;
        
        // Ground collision check (NED: z >= 0 means at/below ground)
        if (pz >= 0.0f) {
            crashed = true;
        }
    }
    
    // ===== Terminal Cost (Complete) =====
    float ex_T = px - pd_x, ey_T = py - pd_y, ez_T = pz - pd_z;
    float cost_pos_T = w_terminal * (ex_T*ex_T + ey_T*ey_T + ez_T*ez_T);
    
    // Terminal attitude (quaternion error)
    float q_dot_T = qd_w*qw + qd_x*qx + qd_y*qy + qd_z*qz;
    float cost_att_T = w_terminal * w_att * (1.0f - q_dot_T*q_dot_T);
    
    // Terminal velocity
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
    float cost_vel_T = w_terminal * w_vel * (vn_T*vn_T + ve_T*ve_T + vd_T*vd_T);
    
    // Terminal angular velocity
    float cost_omega_T = w_terminal * w_omega * (wx*wx + wy*wy + wz*wz);
    
    // Total cost
    float S_total = S_pos + S_vel + S_att + S_omega + S_ctrl + S_smooth
                  + cost_pos_T + cost_att_T + cost_vel_T + cost_omega_T;
    
    // Add crash penalty if crashed
    if (crashed) {
        S_total += crash_cost;
    }
    
    costs[k] = S_total;
    sat_count[k] = sat_cnt;
    
    // Cost breakdown (6 components)
    cost_breakdown[0*K + k] = S_pos + cost_pos_T;
    cost_breakdown[1*K + k] = S_vel + cost_vel_T;
    cost_breakdown[2*K + k] = S_att + cost_att_T;
    cost_breakdown[3*K + k] = S_omega + cost_omega_T;
    cost_breakdown[4*K + k] = S_ctrl;
    cost_breakdown[5*K + k] = S_smooth + (crashed ? crash_cost : 0.0f);
}

// Compute weights kernel
__global__ void compute_weights_kernel(
    const float* costs, float* weights, float min_cost, float lambda, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    weights[k] = expf(-(costs[k] - min_cost) / lambda);
}

// Weighted sum kernel
__global__ void weighted_sum_kernel(
    const float* du_all, const float* weights, float* du_weighted, 
    float weight_sum, int N, int K)
{
    int t = blockIdx.x;
    int mi = threadIdx.x;
    if (t >= N || mi >= 6) return;
    
    float sum = 0.0f;
    int base = mi * N * K + t * K;
    for (int k = 0; k < K; k++) {
        sum += du_all[base + k] * weights[k];
    }
    du_weighted[mi + t * 6] = sum / weight_sum;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Updated interface: 7 inputs
    // [u_opt, u_seq_new, stats] = hexa_mppi_mex_v2(x0, u_seq, u_prev, pos_des, q_des, params, mppi_params)
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("hexa_mppi_mex_v2:nrhs", "7 inputs required: x0, u_seq, u_prev, pos_des, q_des, params, mppi_params");
    }
    
    mxInitGPU();
    
    // Parse inputs
    float* h_x0 = (float*)mxGetData(prhs[0]);       // 19x1 state
    float* h_u_seq = (float*)mxGetData(prhs[1]);    // 6xN control sequence
    float* h_u_prev = (float*)mxGetData(prhs[2]);   // 6x1 previous control
    float* h_pos_des = (float*)mxGetData(prhs[3]);  // 3x1 desired position
    float* h_q_des = (float*)mxGetData(prhs[4]);    // 4x1 desired quaternion
    
    int N = (int)mxGetN(prhs[1]);
    
    // Drone parameters (prhs[5])
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
    
    // MPPI parameters (prhs[6]) - Simplified weights
    int K = (int)mxGetScalar(mxGetField(prhs[6], 0, "K"));
    float dt = (float)mxGetScalar(mxGetField(prhs[6], 0, "dt"));
    float lambda = (float)mxGetScalar(mxGetField(prhs[6], 0, "lambda"));
    float nu = (float)mxGetScalar(mxGetField(prhs[6], 0, "nu"));
    float sigma = (float)mxGetScalar(mxGetField(prhs[6], 0, "sigma"));
    
    // Simplified cost weights (7 instead of 12)
    float w_pos = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_pos"));
    float w_vel = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_vel"));
    float w_att = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_att"));
    float w_omega = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_omega"));
    float w_terminal = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_terminal"));
    float w_smooth = (float)mxGetScalar(mxGetField(prhs[6], 0, "w_smooth"));
    float R_val = (float)mxGetScalar(mxGetField(prhs[6], 0, "R"));
    
    // Crash detection parameters
    float crash_cost = (float)mxGetScalar(mxGetField(prhs[6], 0, "crash_cost"));
    float crash_angle = (float)mxGetScalar(mxGetField(prhs[6], 0, "crash_angle"));
    
    float nu_c = 1.0f - 1.0f / nu;
    
    // Allocate GPU memory
    if (!initialized || K != K_alloc || N != N_alloc) {
        if (initialized) cleanup();
        
        cudaMalloc(&d_x0, 19 * sizeof(float));
        cudaMalloc(&d_u_seq, 6 * N * sizeof(float));
        cudaMalloc(&d_u_prev, 6 * sizeof(float));
        cudaMalloc(&d_pos_des, 3 * sizeof(float));
        cudaMalloc(&d_q_des, 4 * sizeof(float));
        cudaMalloc(&d_noise, 6 * N * K * sizeof(float));
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
    
    // Copy inputs to GPU
    cudaMemcpy(d_x0, h_x0, 19 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_seq, h_u_seq, 6 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_prev, h_u_prev, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_des, h_pos_des, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_des, h_q_des, 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Generate noise
    curandGenerateNormal(gen, d_noise, 6 * N * K, 0.0f, 1.0f);
    
    int bs = 256;
    int gs = (6 * N * K + bs - 1) / bs;
    scale_noise_kernel<<<gs, bs>>>(d_noise, sigma, N, K);
    
    // Launch rollout kernel
    gs = (K + bs - 1) / bs;
    hexa_mppi_rollout_kernel<<<gs, bs>>>(
        d_x0, d_u_seq, d_u_prev, d_pos_des, d_q_des, d_noise,
        d_costs, d_du_all, d_cost_breakdown, d_sat_count,
        m, Jxx, Jyy, Jzz, g, k_T, k_M, L, omega_max, omega_min,
        tau_up, tau_down,
        dt, N, K,
        w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth, R_val,
        crash_cost, crash_angle, lambda, nu_c
    );
    
    // Get costs and diagnostics
    float* h_costs = (float*)mxMalloc(K * sizeof(float));
    float* h_cost_bd = (float*)mxMalloc(6 * K * sizeof(float));
    int* h_sat_count = (int*)mxMalloc(K * sizeof(int));
    
    cudaMemcpy(h_costs, d_costs, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost_bd, d_cost_breakdown, 6 * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sat_count, d_sat_count, K * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Find min cost
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
    
    // Saturation ratio
    int total_sat = 0;
    for (int kk = 0; kk < K; kk++) total_sat += h_sat_count[kk];
    float sat_ratio = (float)total_sat / (float)(6 * N * K);
    
    // Compute weights
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
    
    // Weighted sum of perturbations
    weighted_sum_kernel<<<N, 6>>>(d_du_all, d_weights, d_du_weighted, wsum, N, K);
    
    float* h_du = (float*)mxMalloc(6 * N * sizeof(float));
    cudaMemcpy(h_du, d_du_weighted, 6 * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Update control sequence
    float* h_u_new = (float*)mxMalloc(6 * N * sizeof(float));
    for (int t = 0; t < N; t++) {
        for (int mi = 0; mi < 6; mi++) {
            float u_updated = h_u_seq[mi + t*6] + h_du[mi + t*6];
            h_u_new[mi + t*6] = fminf(fmaxf(u_updated, omega_min), omega_max);
        }
    }
    
    // Output 1: optimal control (first timestep)
    plhs[0] = mxCreateNumericMatrix(6, 1, mxSINGLE_CLASS, mxREAL);
    float* out_u = (float*)mxGetData(plhs[0]);
    for (int mi = 0; mi < 6; mi++) out_u[mi] = h_u_new[mi];
    
    // Output 2: updated control sequence
    plhs[1] = mxCreateNumericMatrix(6, N, mxSINGLE_CLASS, mxREAL);
    memcpy(mxGetData(plhs[1]), h_u_new, 6 * N * sizeof(float));
    
    // Output 3: stats struct
    if (nlhs > 2) {
        const char* field_names[] = {"min_cost", "avg_cost", "cost_breakdown", 
                                     "saturation_ratio", "effective_sample_size"};
        plhs[2] = mxCreateStructMatrix(1, 1, 5, field_names);
        
        mxSetField(plhs[2], 0, "min_cost", mxCreateDoubleScalar((double)min_cost));
        mxSetField(plhs[2], 0, "avg_cost", mxCreateDoubleScalar((double)avg_cost));
        
        // Cost breakdown: [pos, vel, att, omega, ctrl, smooth+crash]
        mxArray* cb = mxCreateDoubleMatrix(6, 1, mxREAL);
        double* cb_data = mxGetPr(cb);
        for (int i = 0; i < 6; i++) {
            cb_data[i] = (double)h_cost_bd[i*K + min_idx];
        }
        mxSetField(plhs[2], 0, "cost_breakdown", cb);
        
        mxSetField(plhs[2], 0, "saturation_ratio", mxCreateDoubleScalar((double)sat_ratio));
        mxSetField(plhs[2], 0, "effective_sample_size", mxCreateDoubleScalar((double)ess));
    }
    
    // Cleanup
    mxFree(h_costs);
    mxFree(h_cost_bd);
    mxFree(h_sat_count);
    mxFree(h_weights);
    mxFree(h_du);
    mxFree(h_u_new);
}
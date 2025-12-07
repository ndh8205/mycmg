/*
 * hexa_mppi_mex.cu - Hexarotor MPPI Controller (CUDA MEX)
 * 
 * 6D motor speed direct control with noise
 * State: 13x1 [pos(3), vel(3), quat(4), omega(3)]
 * Control: 6x1 [omega_1, ..., omega_6] motor speeds [rad/s]
 * 
 * Compile: mexcuda hexa_mppi_mex.cu -lcurand
 * Usage:   [u_opt, u_seq_new, stats] = hexa_mppi_mex(x0, u_seq, pos_des, yaw_des, params, mppi_params)
 *
 * Cost weights (separated):
 *   w_pos_xy, w_pos_z      - position XY vs Z
 *   w_vel_xy, w_vel_z      - velocity XY vs Z
 *   w_att                  - attitude (roll/pitch via qw)
 *   w_yaw                  - yaw angle
 *   w_omega_rp, w_omega_yaw - angular velocity roll/pitch vs yaw
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>

// Persistent GPU memory
static bool initialized = false;
static int K_alloc = 0, N_alloc = 0;
static float *d_x0 = nullptr, *d_u_seq = nullptr, *d_pos_des = nullptr;
static float *d_noise = nullptr, *d_costs = nullptr, *d_weights = nullptr;
static float *d_du_all = nullptr, *d_du_weighted = nullptr;
static float *d_cost_breakdown = nullptr;
static int *d_sat_count = nullptr;
static curandGenerator_t gen;

void cleanup(void) {
    if (initialized) {
        cudaFree(d_x0); cudaFree(d_u_seq); cudaFree(d_pos_des);
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
    const float* __restrict__ pos_des,
    const float* __restrict__ noise,
    float* __restrict__ costs,
    float* __restrict__ du_all,
    float* __restrict__ cost_breakdown,
    int* __restrict__ sat_count,
    float yaw_des,
    float m, float Jxx, float Jyy, float Jzz, float g,
    float k_T, float k_M, float L,
    float omega_max, float omega_min,
    float dt, int N, int K,
    // Separated cost weights
    float w_pos_xy, float w_pos_z,
    float w_vel_xy, float w_vel_z,
    float w_att, float w_yaw,
    float w_omega_rp, float w_omega_yaw,
    float w_terminal, float R_val, float nu_c)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    // Load initial state
    float px = x0[0], py = x0[1], pz = x0[2];
    float vx = x0[3], vy = x0[4], vz = x0[5];
    float qw = x0[6], qx = x0[7], qy = x0[8], qz = x0[9];
    float wx = x0[10], wy = x0[11], wz = x0[12];
    
    float pd_x = pos_des[0], pd_y = pos_des[1], pd_z = pos_des[2];
    
    float b = k_T * k_M;
    float s3 = 0.866025403784f;
    float kL = k_T * L;
    
    // Accumulated costs by component
    float S_pos = 0.0f, S_vel = 0.0f, S_att = 0.0f;
    float S_yaw = 0.0f, S_omega = 0.0f, S_ctrl = 0.0f;
    int sat_cnt = 0;
    
    for (int t = 0; t < N; t++) {
        float d1 = noise[0*N*K + t*K + k];
        float d2 = noise[1*N*K + t*K + k];
        float d3 = noise[2*N*K + t*K + k];
        float d4 = noise[3*N*K + t*K + k];
        float d5 = noise[4*N*K + t*K + k];
        float d6 = noise[5*N*K + t*K + k];
        
        du_all[0*N*K + t*K + k] = d1;
        du_all[1*N*K + t*K + k] = d2;
        du_all[2*N*K + t*K + k] = d3;
        du_all[3*N*K + t*K + k] = d4;
        du_all[4*N*K + t*K + k] = d5;
        du_all[5*N*K + t*K + k] = d6;
        
        float un1 = u_seq[0 + t*6], un2 = u_seq[1 + t*6], un3 = u_seq[2 + t*6];
        float un4 = u_seq[3 + t*6], un5 = u_seq[4 + t*6], un6 = u_seq[5 + t*6];
        
        // Check saturation before clipping
        float u1_raw = un1 + d1, u2_raw = un2 + d2, u3_raw = un3 + d3;
        float u4_raw = un4 + d4, u5_raw = un5 + d5, u6_raw = un6 + d6;
        
        if (u1_raw < omega_min || u1_raw > omega_max) sat_cnt++;
        if (u2_raw < omega_min || u2_raw > omega_max) sat_cnt++;
        if (u3_raw < omega_min || u3_raw > omega_max) sat_cnt++;
        if (u4_raw < omega_min || u4_raw > omega_max) sat_cnt++;
        if (u5_raw < omega_min || u5_raw > omega_max) sat_cnt++;
        if (u6_raw < omega_min || u6_raw > omega_max) sat_cnt++;
        
        float u1 = fminf(fmaxf(u1_raw, omega_min), omega_max);
        float u2 = fminf(fmaxf(u2_raw, omega_min), omega_max);
        float u3 = fminf(fmaxf(u3_raw, omega_min), omega_max);
        float u4 = fminf(fmaxf(u4_raw, omega_min), omega_max);
        float u5 = fminf(fmaxf(u5_raw, omega_min), omega_max);
        float u6 = fminf(fmaxf(u6_raw, omega_min), omega_max);
        
        // ===== Running Cost (SEPARATED) =====
        float ex = px - pd_x, ey = py - pd_y, ez = pz - pd_z;
        
        // Position cost: XY vs Z
        float cost_pos = w_pos_xy * (ex*ex + ey*ey) + w_pos_z * ez*ez;
        
        // Velocity cost: XY vs Z
        float cost_vel = w_vel_xy * (vx*vx + vy*vy) + w_vel_z * vz*vz;
        
        // Attitude cost (roll/pitch via qw)
        float cost_att = w_att * (1.0f - qw*qw);
        
        // Yaw cost
        float yaw_curr = atan2f(2.0f*(qw*qz + qx*qy), 1.0f - 2.0f*(qy*qy + qz*qz));
        float yaw_err = atan2f(sinf(yaw_curr - yaw_des), cosf(yaw_curr - yaw_des));
        float cost_yaw = w_yaw * yaw_err * yaw_err;
        
        // Angular velocity cost: roll/pitch rate vs yaw rate
        float cost_omega = w_omega_rp * (wx*wx + wy*wy) + w_omega_yaw * wz*wz;
        
        // Control cost
        float delta_sq = d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6;
        float un_delta = un1*d1 + un2*d2 + un3*d3 + un4*d4 + un5*d5 + un6*d6;
        float un_sq = un1*un1 + un2*un2 + un3*un3 + un4*un4 + un5*un5 + un6*un6;
        float cost_ctrl = nu_c * 0.5f * R_val * delta_sq + R_val * un_delta + 0.5f * R_val * un_sq;
        
        S_pos += cost_pos * dt;
        S_vel += cost_vel * dt;
        S_att += cost_att * dt;
        S_yaw += cost_yaw * dt;
        S_omega += cost_omega * dt;
        S_ctrl += cost_ctrl * dt;
        
        // ===== Dynamics =====
        float w1sq = u1*u1, w2sq = u2*u2, w3sq = u3*u3;
        float w4sq = u4*u4, w5sq = u5*u5, w6sq = u6*u6;
        
        float T = k_T * (w1sq + w2sq + w3sq + w4sq + w5sq + w6sq);
        float tau_x = kL * (-0.5f*w1sq + 0.5f*w2sq + w3sq + 0.5f*w4sq - 0.5f*w5sq - w6sq);
        float tau_y = kL * s3 * (w1sq + w2sq - w4sq - w5sq);
        float tau_z = b * (w1sq - w2sq + w3sq - w4sq + w5sq - w6sq);
        
        float R11 = 1.0f - 2.0f*(qy*qy + qz*qz);
        float R12 = 2.0f*(qx*qy - qz*qw);
        float R13 = 2.0f*(qx*qz + qy*qw);
        float R21 = 2.0f*(qx*qy + qz*qw);
        float R22 = 1.0f - 2.0f*(qx*qx + qz*qz);
        float R23 = 2.0f*(qy*qz - qx*qw);
        float R31 = 2.0f*(qx*qz - qy*qw);
        float R32 = 2.0f*(qy*qz + qx*qw);
        float R33 = 1.0f - 2.0f*(qx*qx + qy*qy);
        
        float px_dot = R11*vx + R12*vy + R13*vz;
        float py_dot = R21*vx + R22*vy + R23*vz;
        float pz_dot = R31*vx + R32*vy + R33*vz;
        
        float gx_b = R31 * g;
        float gy_b = R32 * g;
        float gz_b = R33 * g;
        
        float cross_x = wy*vz - wz*vy;
        float cross_y = wz*vx - wx*vz;
        float cross_z = wx*vy - wy*vx;
        
        float vx_dot = gx_b - cross_x;
        float vy_dot = gy_b - cross_y;
        float vz_dot = -T/m + gz_b - cross_z;
        
        float qw_dot = 0.5f * (-wx*qx - wy*qy - wz*qz);
        float qx_dot = 0.5f * ( wx*qw + wz*qy - wy*qz);
        float qy_dot = 0.5f * ( wy*qw - wz*qx + wx*qz);
        float qz_dot = 0.5f * ( wz*qw + wy*qx - wx*qy);
        
        float Jwx = Jxx * wx, Jwy = Jyy * wy, Jwz = Jzz * wz;
        float wx_dot = (tau_x - (wy*Jwz - wz*Jwy)) / Jxx;
        float wy_dot = (tau_y - (wz*Jwx - wx*Jwz)) / Jyy;
        float wz_dot = (tau_z - (wx*Jwy - wy*Jwx)) / Jzz;
        
        px += px_dot * dt; py += py_dot * dt; pz += pz_dot * dt;
        vx += vx_dot * dt; vy += vy_dot * dt; vz += vz_dot * dt;
        qw += qw_dot * dt; qx += qx_dot * dt; qy += qy_dot * dt; qz += qz_dot * dt;
        wx += wx_dot * dt; wy += wy_dot * dt; wz += wz_dot * dt;
        
        float qnorm = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
        if (qnorm > 1e-8f) {
            qw /= qnorm; qx /= qnorm; qy /= qnorm; qz /= qnorm;
        }
    }
    
    // Terminal cost
    float ex_T = px - pd_x, ey_T = py - pd_y, ez_T = pz - pd_z;
    S_pos += w_terminal * (w_pos_xy * (ex_T*ex_T + ey_T*ey_T) + w_pos_z * ez_T*ez_T);
    S_att += w_att * (1.0f - qw*qw);
    S_vel += 0.5f * w_terminal * (w_vel_xy * (vx*vx + vy*vy) + w_vel_z * vz*vz);
    
    // Store total cost
    costs[k] = S_pos + S_vel + S_att + S_yaw + S_omega + S_ctrl;
    
    // Store cost breakdown
    cost_breakdown[0*K + k] = S_pos;
    cost_breakdown[1*K + k] = S_vel;
    cost_breakdown[2*K + k] = S_att;
    cost_breakdown[3*K + k] = S_yaw;
    cost_breakdown[4*K + k] = S_omega;
    cost_breakdown[5*K + k] = S_ctrl;
    
    sat_count[k] = sat_cnt;
}

// Compute importance weights
__global__ void compute_weights_kernel(float* costs, float* weights, float min_cost, float lambda, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        weights[k] = expf(-(costs[k] - min_cost) / lambda);
    }
}

// Weighted sum of perturbations
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
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("hexa_mppi_mex:nrhs", "6 inputs required");
    }
    
    mxInitGPU();
    
    float* h_x0 = (float*)mxGetData(prhs[0]);
    float* h_u_seq = (float*)mxGetData(prhs[1]);
    float* h_pos_des = (float*)mxGetData(prhs[2]);
    float yaw_des = (float)mxGetScalar(prhs[3]);
    
    int N = (int)mxGetN(prhs[1]);
    
    // Drone parameters
    float m = (float)mxGetScalar(mxGetField(prhs[4], 0, "m"));
    float Jxx = (float)mxGetScalar(mxGetField(prhs[4], 0, "Jxx"));
    float Jyy = (float)mxGetScalar(mxGetField(prhs[4], 0, "Jyy"));
    float Jzz = (float)mxGetScalar(mxGetField(prhs[4], 0, "Jzz"));
    float g = (float)mxGetScalar(mxGetField(prhs[4], 0, "g"));
    float k_T = (float)mxGetScalar(mxGetField(prhs[4], 0, "k_T"));
    float k_M = (float)mxGetScalar(mxGetField(prhs[4], 0, "k_M"));
    float L = (float)mxGetScalar(mxGetField(prhs[4], 0, "L"));
    float omega_max = (float)mxGetScalar(mxGetField(prhs[4], 0, "omega_max"));
    float omega_min = (float)mxGetScalar(mxGetField(prhs[4], 0, "omega_min"));
    
    // MPPI parameters
    int K = (int)mxGetScalar(mxGetField(prhs[5], 0, "K"));
    float dt = (float)mxGetScalar(mxGetField(prhs[5], 0, "dt"));
    float lambda = (float)mxGetScalar(mxGetField(prhs[5], 0, "lambda"));
    float nu = (float)mxGetScalar(mxGetField(prhs[5], 0, "nu"));
    float sigma = (float)mxGetScalar(mxGetField(prhs[5], 0, "sigma"));
    
    // Separated cost weights
    float w_pos_xy = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_pos_xy"));
    float w_pos_z = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_pos_z"));
    float w_vel_xy = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_vel_xy"));
    float w_vel_z = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_vel_z"));
    float w_att = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_att"));
    float w_yaw = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_yaw"));
    float w_omega_rp = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_omega_rp"));
    float w_omega_yaw = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_omega_yaw"));
    float w_terminal = (float)mxGetScalar(mxGetField(prhs[5], 0, "w_terminal"));
    float R_val = (float)mxGetScalar(mxGetField(prhs[5], 0, "R"));
    
    float nu_c = 1.0f - 1.0f / nu;
    
    // Allocate GPU memory
    if (!initialized || K != K_alloc || N != N_alloc) {
        if (initialized) cleanup();
        
        cudaMalloc(&d_x0, 13 * sizeof(float));
        cudaMalloc(&d_u_seq, 6 * N * sizeof(float));
        cudaMalloc(&d_pos_des, 3 * sizeof(float));
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
    
    cudaMemcpy(d_x0, h_x0, 13 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_seq, h_u_seq, 6 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_des, h_pos_des, 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    curandGenerateNormal(gen, d_noise, 6 * N * K, 0.0f, 1.0f);
    
    int bs = 256;
    int gs = (6 * N * K + bs - 1) / bs;
    scale_noise_kernel<<<gs, bs>>>(d_noise, sigma, N, K);
    
    gs = (K + bs - 1) / bs;
    hexa_mppi_rollout_kernel<<<gs, bs>>>(
        d_x0, d_u_seq, d_pos_des, d_noise, d_costs, d_du_all,
        d_cost_breakdown, d_sat_count,
        yaw_des, m, Jxx, Jyy, Jzz, g, k_T, k_M, L, omega_max, omega_min,
        dt, N, K,
        w_pos_xy, w_pos_z, w_vel_xy, w_vel_z,
        w_att, w_yaw, w_omega_rp, w_omega_yaw,
        w_terminal, R_val, nu_c
    );
    
    // Get costs and diagnostics
    float* h_costs = (float*)mxMalloc(K * sizeof(float));
    float* h_cost_bd = (float*)mxMalloc(6 * K * sizeof(float));
    int* h_sat_count = (int*)mxMalloc(K * sizeof(int));
    
    cudaMemcpy(h_costs, d_costs, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost_bd, d_cost_breakdown, 6 * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sat_count, d_sat_count, K * sizeof(int), cudaMemcpyDeviceToHost);
    
    float min_cost = h_costs[0];
    float sum_cost = h_costs[0];
    int min_idx = 0;
    for (int k = 1; k < K; k++) {
        if (h_costs[k] < min_cost) {
            min_cost = h_costs[k];
            min_idx = k;
        }
        sum_cost += h_costs[k];
    }
    float avg_cost = sum_cost / K;
    
    int total_sat = 0;
    for (int k = 0; k < K; k++) total_sat += h_sat_count[k];
    float sat_ratio = (float)total_sat / (float)(6 * N * K);
    
    compute_weights_kernel<<<gs, bs>>>(d_costs, d_weights, min_cost, lambda, K);
    
    float* h_weights = (float*)mxMalloc(K * sizeof(float));
    cudaMemcpy(h_weights, d_weights, K * sizeof(float), cudaMemcpyDeviceToHost);
    
    float wsum = 0.0f;
    float wsum_sq = 0.0f;
    for (int k = 0; k < K; k++) {
        wsum += h_weights[k];
        wsum_sq += h_weights[k] * h_weights[k];
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
    
    // Output 1: optimal control
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
        
        mxArray* cb = mxCreateDoubleMatrix(6, 1, mxREAL);
        double* cb_data = mxGetPr(cb);
        for (int i = 0; i < 6; i++) {
            cb_data[i] = (double)h_cost_bd[i*K + min_idx];
        }
        mxSetField(plhs[2], 0, "cost_breakdown", cb);
        
        mxSetField(plhs[2], 0, "saturation_ratio", mxCreateDoubleScalar((double)sat_ratio));
        mxSetField(plhs[2], 0, "effective_sample_size", mxCreateDoubleScalar((double)ess));
    }
    
    mxFree(h_costs);
    mxFree(h_cost_bd);
    mxFree(h_sat_count);
    mxFree(h_weights);
    mxFree(h_du);
    mxFree(h_u_new);
}
/*
 * cmg_mppi_mex.cu - CMG MPPI Controller (CUDA MEX)
 * 
 * Compile: mexcuda cmg_mppi_mex.cu -lcurand
 * Usage:   [u_opt, u_seq_new] = cmg_mppi_mex(x0, u_seq, att_des, params, mppi_params)
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <curand.h>

// Persistent GPU memory
static bool initialized = false;
static int K_alloc = 0, N_alloc = 0;
static float *d_x0 = nullptr, *d_u_seq = nullptr, *d_att_des = nullptr;
static float *d_noise = nullptr, *d_costs = nullptr, *d_weights = nullptr;
static float *d_du_all = nullptr, *d_du_weighted = nullptr;
static curandGenerator_t gen;

void cleanup(void) {
    if (initialized) {
        cudaFree(d_x0); cudaFree(d_u_seq); cudaFree(d_att_des);
        cudaFree(d_noise); cudaFree(d_costs); cudaFree(d_weights);
        cudaFree(d_du_all); cudaFree(d_du_weighted);
        curandDestroyGenerator(gen);
        initialized = false;
    }
}

__global__ void scale_noise_kernel(float* noise, float sigma1, float sigma2, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx < total) {
        noise[idx] *= sigma1;
        noise[total + idx] *= sigma2;
    }
}

__global__ void mppi_rollout_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ u_seq,
    const float* __restrict__ att_des,
    const float* __restrict__ noise,
    float* __restrict__ costs,
    float* __restrict__ du_all,
    float Jxx, float Jyy, float Jzz,
    float Jxi, float Jyi, float Jzi,
    float h1, float h2, float gr_max,
    float dt, int N, int K,
    float Q2, float Q3, float Qw, float Sw, float Wt,
    float R1, float R2, float nu_c)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    float a1 = x0[0], a2 = x0[1], a3 = x0[2];
    float w1 = x0[3], w2 = x0[4], w3 = x0[5];
    float g1 = x0[6], g2 = x0[7];
    
    float ad2 = att_des[1], ad3 = att_des[2];
    float S = 0.0f;
    
    for (int t = 0; t < N; t++) {
        int idx = t * K + k;
        float d1 = noise[idx];
        float d2 = noise[N * K + idx];
        
        du_all[idx] = d1;
        du_all[N * K + idx] = d2;
        
        float un1 = u_seq[t * 2], un2 = u_seq[t * 2 + 1];
        float u1 = fminf(fmaxf(un1 + d1, -gr_max), gr_max);
        float u2 = fminf(fmaxf(un2 + d2, -gr_max), gr_max);
        
        float sg1 = sinf(g1), cg1 = cosf(g1);
        float sg2 = sinf(g2), cg2 = cosf(g2);
        
        float hy = h1*cg1 + h2*cg2;
        float hz = h1*sg1 + h2*sg2;
        
        float ty = -h1*sg1*u1 - h2*sg2*u2;
        float tz =  h1*cg1*u1 + h2*cg2*u2;
        
        float Jw1 = Jxx*w1, Jw2 = Jyy*w2, Jw3 = Jzz*w3;
        float gx = w3*Jw2 - w2*Jw3;
        float gy = w1*Jw3 - w3*Jw1;
        float gz = w2*Jw1 - w1*Jw2;
        
        float cx = w3*hy - w2*hz;
        float cy = -w1*hz;
        float cz = w1*hy;
        
        a1 += w1*dt; a2 += w2*dt; a3 += w3*dt;
        w1 += (gx+cx)*Jxi*dt;
        w2 += (ty+gy+cy)*Jyi*dt;
        w3 += (tz+gz+cz)*Jzi*dt;
        g1 += u1*dt; g2 += u2*dt;
        
        float e2 = a2-ad2, e3 = a3-ad3;
        S += (Q2*e2*e2 + Q3*e3*e3 + Qw*(w1*w1+w2*w2+w3*w3)
            + Sw/(fabsf(sinf(g1-g2))+1e-6f)
            + nu_c*0.5f*(R1*d1*d1+R2*d2*d2) + R1*un1*d1 + R2*un2*d2
            + 0.5f*(R1*un1*un1+R2*un2*un2)) * dt;
    }
    
    costs[k] = S + Wt*((a2-ad2)*(a2-ad2) + (a3-ad3)*(a3-ad3));
}

__global__ void compute_weights_kernel(float* costs, float* weights, float min_cost, float lambda, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) weights[k] = expf(-(costs[k] - min_cost) / lambda);
}

__global__ void weighted_sum_kernel(
    const float* du_all, const float* weights, float* du_weighted, float weight_sum, int N, int K)
{
    int t = blockIdx.x;
    int dim = threadIdx.x;
    if (t >= N || dim >= 2) return;
    
    float sum = 0.0f;
    int base = dim * N * K + t * K;
    for (int k = 0; k < K; k++) sum += du_all[base + k] * weights[k];
    du_weighted[t * 2 + dim] = sum / weight_sum;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 5) mexErrMsgIdAndTxt("cmg_mppi_mex:nrhs", "5 inputs required.");
    
    mxInitGPU();
    
    // Use mxGetData instead of mxGetSingles for compatibility
    float* h_x0 = (float*)mxGetData(prhs[0]);
    float* h_u_seq = (float*)mxGetData(prhs[1]);
    float* h_att_des = (float*)mxGetData(prhs[2]);
    
    int N = (int)mxGetN(prhs[1]);
    
    float Jxx = (float)mxGetScalar(mxGetField(prhs[3], 0, "Jxx"));
    float Jyy = (float)mxGetScalar(mxGetField(prhs[3], 0, "Jyy"));
    float Jzz = (float)mxGetScalar(mxGetField(prhs[3], 0, "Jzz"));
    float h1 = (float)mxGetScalar(mxGetField(prhs[3], 0, "h1"));
    float h2 = (float)mxGetScalar(mxGetField(prhs[3], 0, "h2"));
    float gr_max = (float)mxGetScalar(mxGetField(prhs[3], 0, "gr_max"));
    
    int K = (int)mxGetScalar(mxGetField(prhs[4], 0, "K"));
    float dt = (float)mxGetScalar(mxGetField(prhs[4], 0, "dt"));
    float lambda = (float)mxGetScalar(mxGetField(prhs[4], 0, "lambda"));
    float nu = (float)mxGetScalar(mxGetField(prhs[4], 0, "nu"));
    float sigma1 = (float)mxGetScalar(mxGetField(prhs[4], 0, "sigma1"));
    float sigma2 = (float)mxGetScalar(mxGetField(prhs[4], 0, "sigma2"));
    float R1 = (float)mxGetScalar(mxGetField(prhs[4], 0, "R1"));
    float R2 = (float)mxGetScalar(mxGetField(prhs[4], 0, "R2"));
    float Q2 = (float)mxGetScalar(mxGetField(prhs[4], 0, "Q2"));
    float Q3 = (float)mxGetScalar(mxGetField(prhs[4], 0, "Q3"));
    float Qw = (float)mxGetScalar(mxGetField(prhs[4], 0, "Qw"));
    float Sw = (float)mxGetScalar(mxGetField(prhs[4], 0, "Sw"));
    float Wt = (float)mxGetScalar(mxGetField(prhs[4], 0, "Wt"));
    
    if (!initialized || K != K_alloc || N != N_alloc) {
        if (initialized) cleanup();
        
        cudaMalloc(&d_x0, 8 * sizeof(float));
        cudaMalloc(&d_u_seq, 2 * N * sizeof(float));
        cudaMalloc(&d_att_des, 3 * sizeof(float));
        cudaMalloc(&d_noise, 2 * N * K * sizeof(float));
        cudaMalloc(&d_costs, K * sizeof(float));
        cudaMalloc(&d_weights, K * sizeof(float));
        cudaMalloc(&d_du_all, 2 * N * K * sizeof(float));
        cudaMalloc(&d_du_weighted, 2 * N * sizeof(float));
        
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 42);
        
        mexAtExit(cleanup);
        initialized = true;
        K_alloc = K; N_alloc = N;
    }
    
    cudaMemcpy(d_x0, h_x0, 8 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_seq, h_u_seq, 2 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_att_des, h_att_des, 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    curandGenerateNormal(gen, d_noise, 2 * N * K, 0.0f, 1.0f);
    int bs = 256, gs = (N * K + bs - 1) / bs;
    scale_noise_kernel<<<gs, bs>>>(d_noise, sigma1, sigma2, N, K);
    
    gs = (K + bs - 1) / bs;
    mppi_rollout_kernel<<<gs, bs>>>(
        d_x0, d_u_seq, d_att_des, d_noise, d_costs, d_du_all,
        Jxx, Jyy, Jzz, 1.0f/Jxx, 1.0f/Jyy, 1.0f/Jzz,
        h1, h2, gr_max, dt, N, K,
        Q2, Q3, Qw, Sw, Wt, R1, R2, 1.0f - 1.0f/nu
    );
    
    float* h_costs = (float*)mxMalloc(K * sizeof(float));
    cudaMemcpy(h_costs, d_costs, K * sizeof(float), cudaMemcpyDeviceToHost);
    float min_cost = h_costs[0];
    for (int k = 1; k < K; k++) if (h_costs[k] < min_cost) min_cost = h_costs[k];
    
    compute_weights_kernel<<<gs, bs>>>(d_costs, d_weights, min_cost, lambda, K);
    
    float* h_weights = (float*)mxMalloc(K * sizeof(float));
    cudaMemcpy(h_weights, d_weights, K * sizeof(float), cudaMemcpyDeviceToHost);
    float wsum = 0.0f;
    for (int k = 0; k < K; k++) wsum += h_weights[k];
    if (wsum < 1e-10f) wsum = 1.0f;
    
    weighted_sum_kernel<<<N, 2>>>(d_du_all, d_weights, d_du_weighted, wsum, N, K);
    
    float* h_du = (float*)mxMalloc(2 * N * sizeof(float));
    cudaMemcpy(h_du, d_du_weighted, 2 * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float* h_u_new = (float*)mxMalloc(2 * N * sizeof(float));
    for (int t = 0; t < N; t++) {
        h_u_new[t*2] = fminf(fmaxf(h_u_seq[t*2] + h_du[t*2], -gr_max), gr_max);
        h_u_new[t*2+1] = fminf(fmaxf(h_u_seq[t*2+1] + h_du[t*2+1], -gr_max), gr_max);
    }
    
    // Output using mxGetData
    plhs[0] = mxCreateNumericMatrix(2, 1, mxSINGLE_CLASS, mxREAL);
    float* out0 = (float*)mxGetData(plhs[0]);
    out0[0] = h_u_new[0];
    out0[1] = h_u_new[1];
    
    plhs[1] = mxCreateNumericMatrix(2, N, mxSINGLE_CLASS, mxREAL);
    memcpy(mxGetData(plhs[1]), h_u_new, 2 * N * sizeof(float));
    
    mxFree(h_costs); mxFree(h_weights); mxFree(h_du); mxFree(h_u_new);
}
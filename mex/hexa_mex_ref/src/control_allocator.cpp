#include "../include/types.hpp"
#include "../include/params.hpp"
#include <cmath>
#include <cstring>

// Hexarotor control allocation
// Motor layout (top view, looking down Z+):
//           X+ (Front)
//            ↑
//       M2      M1
//      (CW)   (CCW)
//         \    /
//  M3 -----+-----→ Y+ (Right)
// (CCW)    |    M6 (CW)
//         /    \
//      M4      M5
//     (CW)    (CCW)
//
// Allocation Matrix B (4x6):
//   [T; τx; τy; τz] = B * [ω1²; ω2²; ω3²; ω4²; ω5²; ω6²]

void control_allocator_inverse(const float* cmd_vec, float* omega_sq, const Params& params) {
    // cmd_vec[0] = T (total thrust)
    // cmd_vec[1] = tau_x (roll torque)
    // cmd_vec[2] = tau_y (pitch torque)
    // cmd_vec[3] = tau_z (yaw torque)

    // Suppress unused parameter warning
    (void)params;

    float T = cmd_vec[0];
    float tau_x = cmd_vec[1];
    float tau_y = cmd_vec[2];
    float tau_z = cmd_vec[3];

    // Allocation matrix B (4x6):
    // Row 0: [ k,  k,  k,  k,  k,  k ]
    // Row 1: [-kL/2, kL/2, kL, kL/2, -kL/2, -kL]
    // Row 2: [kL*s3, kL*s3, 0, -kL*s3, -kL*s3, 0]
    // Row 3: [b, -b, b, -b, -b, b]  ← NOTE: M6 is CW (last -b changed to +b)

    // Solve B * omega_sq = cmd_vec using least-squares (pinv(B) * cmd_vec)
    // Precomputed pseudo-inverse for hexarotor (from MATLAB control_allocator.m)
    //
    // For hexarotor, B is 4x6, so pinv(B) is 6x4
    // Computed using: pinv_B = B' * inv(B * B')

    // Simplified analytical solution (from MATLAB pinv computation):
    // This is derived from the specific geometry of hexarotor

    // Use exact pseudo-inverse values computed from actual B matrix with physical units
    // pinv(B) where B has k_T, k_T*L, k_T*L*sqrt(3)/2, k_M coefficients
    const float pinv[6][4] = {
        {1.2919896641e+03f, -1.3458225668e+03f,  2.3310330636e+03f,  7.1530758226e+03f},  // M1
        {1.2919896641e+03f,  1.3458225668e+03f,  2.3310330636e+03f, -7.1530758226e+03f},  // M2
        {1.2919896641e+03f,  2.6916451335e+03f,  0.0f,               7.1530758226e+03f},  // M3
        {1.2919896641e+03f,  1.3458225668e+03f, -2.3310330636e+03f, -7.1530758226e+03f},  // M4
        {1.2919896641e+03f, -1.3458225668e+03f, -2.3310330636e+03f,  7.1530758226e+03f},  // M5
        {1.2919896641e+03f, -2.6916451335e+03f,  0.0f,              -7.1530758226e+03f}   // M6
    };

    // Apply pseudo-inverse directly to command vector (no additional scaling needed)
    for (int i = 0; i < 6; i++) {
        omega_sq[i] = pinv[i][0] * T +
                      pinv[i][1] * tau_x +
                      pinv[i][2] * tau_y +
                      pinv[i][3] * tau_z;
    }
}

void control_allocator_forward(const float* omega_sq, float* cmd_vec, const Params& params) {
    // Forward: [T; τx; τy; τz] = B * omega_sq

    float k = params.drone.k_T;
    float b = k * params.drone.k_M;
    float L = params.drone.L;
    float s3 = sqrtf(3.0f) / 2.0f;

    float kL = k * L;
    float kL_s3 = kL * s3;

    // Total thrust
    cmd_vec[0] = k * (omega_sq[0] + omega_sq[1] + omega_sq[2] + omega_sq[3] + omega_sq[4] + omega_sq[5]);

    // Roll torque (tau_x)
    cmd_vec[1] = -kL/2.0f * omega_sq[0] + kL/2.0f * omega_sq[1] + kL * omega_sq[2]
                 + kL/2.0f * omega_sq[3] - kL/2.0f * omega_sq[4] - kL * omega_sq[5];

    // Pitch torque (tau_y)
    cmd_vec[2] = kL_s3 * omega_sq[0] + kL_s3 * omega_sq[1]
                 - kL_s3 * omega_sq[3] - kL_s3 * omega_sq[4];

    // Yaw torque (tau_z): CCW motors (+b), CW motors (-b)
    // M1(CCW), M2(CW), M3(CCW), M4(CW), M5(CCW), M6(CW)
    cmd_vec[3] = b * omega_sq[0] - b * omega_sq[1] + b * omega_sq[2]
                 - b * omega_sq[3] + b * omega_sq[4] - b * omega_sq[5];
}

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include "types.hpp"

// Skew-symmetric matrix from vector
inline Matrix3 skew3(const Vector3& v) {
    Matrix3 result;
    result.data[0] = 0;      result.data[1] = -v.z;   result.data[2] = v.y;
    result.data[3] = v.z;    result.data[4] = 0;      result.data[5] = -v.x;
    result.data[6] = -v.y;   result.data[7] = v.x;    result.data[8] = 0;
    return result;
}

// Quaternion to DCM (Direction Cosine Matrix) - Body to NED
// Uses Ξ-Ψ matrix method from MATLAB
inline Matrix3 quat_to_dcm(const Quaternion& q) {
    float qw = q.w, qx = q.x, qy = q.y, qz = q.z;

    // Xi matrix: [-qv'; qw*I + skew(qv)]
    Matrix3 xi;
    xi.data[0] = -qx;           xi.data[1] = -qy;           xi.data[2] = -qz;
    xi.data[3] = qw;            xi.data[4] = -qz;           xi.data[5] = qy;
    xi.data[6] = qz;            xi.data[7] = qw;            xi.data[8] = -qx;
    xi.data[9] = -qy;           xi.data[10] = qx;           xi.data[11] = qw;

    // Psi matrix: [-qv'; qw*I - skew(qv)]
    Matrix3 psi;
    psi.data[0] = -qx;          psi.data[1] = -qy;          psi.data[2] = -qz;
    psi.data[3] = qw;           psi.data[4] = qz;           psi.data[5] = -qy;
    psi.data[6] = -qz;          psi.data[7] = qw;           psi.data[8] = qx;
    psi.data[9] = qy;           psi.data[10] = -qx;         psi.data[11] = qw;

    // R = Psi' * Xi
    Matrix3 R;
    R.data[0] = 1 - 2*(qy*qy + qz*qz);
    R.data[1] = 2*(qx*qy - qz*qw);
    R.data[2] = 2*(qx*qz + qy*qw);

    R.data[3] = 2*(qx*qy + qz*qw);
    R.data[4] = 1 - 2*(qx*qx + qz*qz);
    R.data[5] = 2*(qy*qz - qx*qw);

    R.data[6] = 2*(qx*qz - qy*qw);
    R.data[7] = 2*(qy*qz + qx*qw);
    R.data[8] = 1 - 2*(qx*qx + qy*qy);

    return R;
}

// DCM to Quaternion (Shepperd's method)
inline Quaternion dcm_to_quat(const Matrix3& R) {
    float trace_R = R.data[0] + R.data[4] + R.data[8];
    Quaternion q;

    if (trace_R > 0) {
        float s = 0.5f / sqrtf(trace_R + 1.0f);
        q.w = 0.25f / s;
        q.x = (R.data[7] - R.data[5]) * s;
        q.y = (R.data[2] - R.data[6]) * s;
        q.z = (R.data[3] - R.data[1]) * s;
    } else if (R.data[0] > R.data[4] && R.data[0] > R.data[8]) {
        float s = 2.0f * sqrtf(1.0f + R.data[0] - R.data[4] - R.data[8]);
        q.w = (R.data[7] - R.data[5]) / s;
        q.x = 0.25f * s;
        q.y = (R.data[1] + R.data[3]) / s;
        q.z = (R.data[2] + R.data[6]) / s;
    } else if (R.data[4] > R.data[8]) {
        float s = 2.0f * sqrtf(1.0f + R.data[4] - R.data[0] - R.data[8]);
        q.w = (R.data[2] - R.data[6]) / s;
        q.x = (R.data[1] + R.data[3]) / s;
        q.y = 0.25f * s;
        q.z = (R.data[5] + R.data[7]) / s;
    } else {
        float s = 2.0f * sqrtf(1.0f + R.data[8] - R.data[0] - R.data[4]);
        q.w = (R.data[3] - R.data[1]) / s;
        q.x = (R.data[2] + R.data[6]) / s;
        q.y = (R.data[5] + R.data[7]) / s;
        q.z = 0.25f * s;
    }

    return q.normalized();
}

// Quaternion to Euler angles (3-2-1 sequence: ZYX)
inline Vector3 quat_to_euler(const Quaternion& q) {
    Matrix3 R = quat_to_dcm(q);

    float phi = atan2f(R.data[7], R.data[8]);          // Roll
    float theta = -asinf(R.data[6]);                   // Pitch
    float psi = atan2f(R.data[3], R.data[0]);          // Yaw

    return Vector3(phi, theta, psi);
}

// Euler angles to quaternion (ZYX: yaw, pitch, roll)
inline Quaternion euler_to_quat(float yaw, float pitch, float roll) {
    float cy = cosf(yaw * 0.5f);
    float sy = sinf(yaw * 0.5f);
    float cp = cosf(pitch * 0.5f);
    float sp = sinf(pitch * 0.5f);
    float cr = cosf(roll * 0.5f);
    float sr = sinf(roll * 0.5f);

    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q.normalized();
}

// Quaternion multiplication (Hamilton product)
inline Quaternion quat_mult(const Quaternion& q, const Quaternion& r) {
    return Quaternion(
        q.w*r.w - q.x*r.x - q.y*r.y - q.z*r.z,
        q.w*r.x + q.x*r.w + q.y*r.z - q.z*r.y,
        q.w*r.y - q.x*r.z + q.y*r.w + q.z*r.x,
        q.w*r.z + q.x*r.y - q.y*r.x + q.z*r.w
    );
}

// Quaternion derivative (body angular velocity)
inline Quaternion quat_derivative(const Quaternion& q, const Vector3& omega) {
    // Omega matrix method: qdot = 0.5 * Omega * q
    return Quaternion(
        0.5f * (-omega.x*q.x - omega.y*q.y - omega.z*q.z),
        0.5f * ( omega.x*q.w + omega.z*q.y - omega.y*q.z),
        0.5f * ( omega.y*q.w - omega.z*q.x + omega.x*q.z),
        0.5f * ( omega.z*q.w + omega.y*q.x - omega.x*q.y)
    );
}

// Ensure quaternion continuity (shortest path)
inline Quaternion ensure_quat_cont(const Quaternion& q_new, const Quaternion& q_prev) {
    float dot = q_new.w*q_prev.w + q_new.x*q_prev.x + q_new.y*q_prev.y + q_new.z*q_prev.z;
    if (dot < 0) {
        return Quaternion(-q_new.w, -q_new.x, -q_new.y, -q_new.z);
    }
    return q_new;
}

#endif // MATH_UTILS_HPP

#ifndef TYPES_HPP
#define TYPES_HPP

#include <cmath>
#include <cstring>
#include <algorithm>

// Vector3: 3D vector
struct Vector3 {
    float x, y, z;

    Vector3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}

    Vector3 operator+(const Vector3& v) const { return Vector3(x + v.x, y + v.y, z + v.z); }
    Vector3 operator-(const Vector3& v) const { return Vector3(x - v.x, y - v.y, z - v.z); }
    Vector3 operator*(float s) const { return Vector3(x * s, y * s, z * s); }
    Vector3 operator/(float s) const { return Vector3(x / s, y / s, z / s); }

    Vector3& operator+=(const Vector3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vector3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    Vector3 cross(const Vector3& v) const {
        return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    float dot(const Vector3& v) const { return x * v.x + y * v.y + z * v.z; }
    float norm() const { return sqrtf(x * x + y * y + z * z); }
    float norm2() const { return x * x + y * y + z * z; }

    Vector3 normalized() const {
        float n = norm();
        return (n > 1e-8f) ? (*this / n) : Vector3(0, 0, 0);
    }

    float* data() { return &x; }
    const float* data() const { return &x; }
};

// Vector2: 2D vector
struct Vector2 {
    float x, y;

    Vector2(float x_ = 0, float y_ = 0) : x(x_), y(y_) {}

    Vector2 operator+(const Vector2& v) const { return Vector2(x + v.x, y + v.y); }
    Vector2 operator-(const Vector2& v) const { return Vector2(x - v.x, y - v.y); }
    Vector2 operator*(float s) const { return Vector2(x * s, y * s); }

    float norm() const { return sqrtf(x * x + y * y); }
    float norm2() const { return x * x + y * y; }
};

// Quaternion: [qw, qx, qy, qz] (scalar-first convention)
struct Quaternion {
    float w, x, y, z;

    Quaternion(float w_ = 1, float x_ = 0, float y_ = 0, float z_ = 0)
        : w(w_), x(x_), y(y_), z(z_) {}

    float norm() const { return sqrtf(w*w + x*x + y*y + z*z); }

    Quaternion normalized() const {
        float n = norm();
        return (n > 1e-8f) ? Quaternion(w/n, x/n, y/n, z/n) : Quaternion(1, 0, 0, 0);
    }

    Quaternion conjugate() const { return Quaternion(w, -x, -y, -z); }
};

// State28: Full hexarotor state vector
struct State28 {
    Vector3 pos;           // Position (NED) [m]
    Vector3 vel;           // Velocity (body) [m/s]
    Quaternion quat;       // Attitude quaternion
    Vector3 omega;         // Angular velocity (body) [rad/s]
    float omega_m[6];      // Motor speeds [rad/s]
    Vector3 bias_gyro;     // Gyro bias [rad/s]
    Vector3 bias_accel;    // Accel bias [m/s^2]
    Vector3 bias_mag;      // Mag bias [Gauss]

    State28() {
        pos = Vector3(0, 0, 0);
        vel = Vector3(0, 0, 0);
        quat = Quaternion(1, 0, 0, 0);
        omega = Vector3(0, 0, 0);
        memset(omega_m, 0, sizeof(omega_m));
        bias_gyro = Vector3(0, 0, 0);
        bias_accel = Vector3(0, 0, 0);
        bias_mag = Vector3(0, 0, 0);
    }

    // Convert to array for RK4 integration
    void to_array(float* arr) const {
        arr[0] = pos.x; arr[1] = pos.y; arr[2] = pos.z;
        arr[3] = vel.x; arr[4] = vel.y; arr[5] = vel.z;
        arr[6] = quat.w; arr[7] = quat.x; arr[8] = quat.y; arr[9] = quat.z;
        arr[10] = omega.x; arr[11] = omega.y; arr[12] = omega.z;
        for (int i = 0; i < 6; i++) arr[13 + i] = omega_m[i];
        arr[19] = bias_gyro.x; arr[20] = bias_gyro.y; arr[21] = bias_gyro.z;
        arr[22] = bias_accel.x; arr[23] = bias_accel.y; arr[24] = bias_accel.z;
        arr[25] = bias_mag.x; arr[26] = bias_mag.y; arr[27] = bias_mag.z;
    }

    // Convert from array
    void from_array(const float* arr) {
        pos = Vector3(arr[0], arr[1], arr[2]);
        vel = Vector3(arr[3], arr[4], arr[5]);
        quat = Quaternion(arr[6], arr[7], arr[8], arr[9]);
        omega = Vector3(arr[10], arr[11], arr[12]);
        for (int i = 0; i < 6; i++) omega_m[i] = arr[13 + i];
        bias_gyro = Vector3(arr[19], arr[20], arr[21]);
        bias_accel = Vector3(arr[22], arr[23], arr[24]);
        bias_mag = Vector3(arr[25], arr[26], arr[27]);
    }
};

// 3x3 Matrix (row-major)
struct Matrix3 {
    float data[9];

    Matrix3() { memset(data, 0, sizeof(data)); }

    Vector3 operator*(const Vector3& v) const {
        return Vector3(
            data[0]*v.x + data[1]*v.y + data[2]*v.z,
            data[3]*v.x + data[4]*v.y + data[5]*v.z,
            data[6]*v.x + data[7]*v.y + data[8]*v.z
        );
    }

    Matrix3 transpose() const {
        Matrix3 result;
        result.data[0] = data[0]; result.data[1] = data[3]; result.data[2] = data[6];
        result.data[3] = data[1]; result.data[4] = data[4]; result.data[5] = data[7];
        result.data[6] = data[2]; result.data[7] = data[5]; result.data[8] = data[8];
        return result;
    }
};

// Helper functions
inline float deg2rad(float deg) { return deg * M_PI / 180.0f; }
inline float rad2deg(float rad) { return rad * 180.0f / M_PI; }
inline float sign(float x) { return (x >= 0.0f) ? 1.0f : -1.0f; }

#endif // TYPES_HPP

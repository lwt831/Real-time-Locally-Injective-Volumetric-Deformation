#pragma once
#include <cmath>
#include <algorithm>

template<typename R=float>
struct Quat     // quaternion
{
    Quat() {}
    Quat(R x, R y, R z, R w) { q[0] = x; q[1] = y; q[2] = z; q[3] = w; }
    Quat(const R* v) { memcpy(q, v, sizeof(R) * 4); }

    R operator [](int i) const { return q[i]; }
    R& operator [](int i) { return q[i]; }

    Quat operator +(const Quat& x) const { return Quat({ q[0] + x[0], q[1] + x[1], q[2] + x[2], q[3] + x[3] }); }
    Quat operator -(const Quat& x) const { return Quat({ q[0] - x[0], q[1] - x[1], q[2] - x[2], q[3] - x[3] }); }

    // Quaternions always obey:  a^2 + b^2 + c^2 + d^2 = 1
    // If not, dividing by their magnitude will re-normalize them
    Quat normalized() const
    {
        double m = 1. / ::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        return Quat{ R(q[0] * m), R(q[1] * m), R(q[2] * m), R(q[3] * m) };
    }

    // Given two rotations, e1 and e2, expressed as quaternion rotations,
    // compute the equivalent single rotation
    Quat operator *(const Quat& x) const
    {
        R t1[3] = { q[0] * x[3], q[1] * x[3], q[2] * x[3] };
        R t2[3] = { x[0] * q[3], x[1] * q[3], x[2] * q[3] };
        R t3[3];
        vcross(x.q, q, t3);

        Quat r;
        for (int i = 0; i < 3; i++)
            r[i] = t1[i] + t2[i] + t3[i];

        r[3] = q[3] * x[3] - q[0] * x[0] - q[1] * x[1] - q[2] * x[2];

        return r.normalized();
    }

    // Build a rotation matrix, given a quaternion rotation.
    void rotmatrix(float *m) const
    {
        std::fill_n(m, 16, 0.f);
        m[0] = 1.f - 2.f * (q[1] * q[1] + q[2] * q[2]);
        m[1] = 2.f * (q[0] * q[1] - q[2] * q[3]);
        m[2] = 2.f * (q[2] * q[0] + q[1] * q[3]);

        m[4 + 0] = 2.f * (q[0] * q[1] + q[2] * q[3]);
        m[4 + 1] = 1.f - 2.f * (q[2] * q[2] + q[0] * q[0]);
        m[4 + 2] = 2.f * (q[1] * q[2] - q[0] * q[3]);

        m[8 + 0] = 2.f * (q[2] * q[0] - q[1] * q[3]);
        m[8 + 1] = 2.f * (q[1] * q[2] + q[0] * q[3]);
        m[8 + 2] = 1.f - 2.f * (q[1] * q[1] + q[0] * q[0]);

        m[15] = 1.f;
    }

    // simulate a track-ball.  Project the points onto the virtual
    // trackball, then figure out the axis of rotation, which is the cross
    // product of P1 P2 and O P1 (O is the center of the ball, 0,0,0)
    // Note:  This is a deformed trackball-- a trackball in the center,
    // but is deformed into a hyperbolic sheet of rotation away from the
    // center.  This particular function was chosen after trying out
    // several variations.
    // It is assumed that the arguments to this routine are in the range (-1.0 ... 1.0)
    static Quat trackball(R x1, R y1, R x2, R y2)
    {
        // This size should really be based on the distance from the center of
        // rotation to the point on the object underneath the mouse.  That
        // point would then track the mouse as closely as possible.  This is a
        // simple example, though, so that is left as an Exercise for the Programmer.
        const float TRACKBALLSIZE = 0.8f;

        if (x1 == x2 && y1 == y2)  // Zero rotation
            return Quat( 0, 0, 0, 1 );

        // First, figure out z-coordinates for projection of P1 and P2 to deformed sphere
        R p1[3] = { x1, y1, project_to_sphere(TRACKBALLSIZE, x1, y1) };
        R p2[3] = { x2, y2, project_to_sphere(TRACKBALLSIZE, x2, y2) };

        //  Now, we want the cross product of P1 and P2
        R a[3]; // Axis of rotation
        vcross(p2, p1, a);

        //  Figure out how much to rotate around that axis.
        R d[3] = { p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2] };
        double t = ::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (2.0*TRACKBALLSIZE);

        // Avoid problems with out-of-control values...
        t = std::max(std::min(t, 1.), -1.);
		//t = std::clamp(t, -1., 1.);

        // how much to rotate about axis
        double phi = 2.0 * asin(t);

        return axis_angle_to_quat(a, R(phi));
    }

    static void vcross(const R *x, const R *y, R *cross)
    {
        R temp[3] = { x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0] };

        std::copy_n(temp, 3, cross);
    }

    // Project point (x,y) onto a sphere of radius r OR a hyperbolic sheet
    // if we are away from the center of the sphere.
    static R project_to_sphere(R r, R x, R y)
    {
        double d2 = x*x + y*y;
        double r2 = r*r;
        if (d2 < r2*0.5)     /* Inside sphere */
            return R( ::sqrt(r2 - d2) );

        // On hyperbola
        double t2 = r2*0.5;
        return R(t2 / ::sqrt(d2));
    }

    // Given an axis and angle, compute quaternion.
    static Quat axis_angle_to_quat(const R a[3], R phi)
    {
        double r = 1. / ::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double b[3] = { a[0] * r, a[1] * r, a[2] * r };

        r = sin(phi / 2.);
        return Quat({ R(b[0] * r), R(b[1] * r), R(b[2] * r), R(cos(phi / 2.)) });
    }

    R q[4];
};

#pragma once

#include <vector>

#include <Base/Definitions.h>

#include "SmoothCurve.h"

struct CameraWaypoint
{
    RealVector2D center;
    float zoom = 1.0f;

    // Seconds of the video in which the camera moves from the previous waypoint to this one
    float duration = 5.0f;

    // Factor by which the simulation runs faster in the video than at normal speed
    float timeLapse = 1.0f;

    bool operator==(CameraWaypoint const&) const = default;
};

struct CameraState
{
    RealVector2D center;
    float zoom = 1.0f;
    float timeLapse = 1.0f;
};

// Smooth curve through waypoints. The centers are interpolated linearly, the zoom and the time-lapse geometrically, so that zooming and
// speeding up proceed evenly. A Gaussian kernel smooths the interpolation, see SmoothCurve.
class CameraPath
{
public:
    static auto constexpr MinDuration = 0.1f;

    // Needs at least one waypoint. The smoothing is the standard deviation of the kernel in seconds.
    CameraPath(std::vector<CameraWaypoint> const& waypoints, float smoothing);

    float getDuration() const;
    CameraState getState(double time) const;

private:
    static SmoothCurve createCurve(std::vector<CameraWaypoint> const& waypoints, float smoothing);

    SmoothCurve _curve;
};

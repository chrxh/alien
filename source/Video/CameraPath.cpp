#include "CameraPath.h"

#include <algorithm>
#include <cmath>

namespace
{
    // The zoom and the time-lapse are interpolated by their logarithms
    auto constexpr MinPositiveValue = 1.0e-6f;

    enum Channel_
    {
        Channel_CenterX,
        Channel_CenterY,
        Channel_LogZoom,
        Channel_LogTimeLapse,
    };
}

CameraPath::CameraPath(std::vector<CameraWaypoint> const& waypoints, float smoothing)
    : _curve(createCurve(waypoints, smoothing))
{}

float CameraPath::getDuration() const
{
    return toFloat(_curve.getEndTime());
}

CameraState CameraPath::getState(double time) const
{
    auto values = _curve.getValues(time);
    return CameraState{
        .center = {toFloat(values.at(Channel_CenterX)), toFloat(values.at(Channel_CenterY))},
        .zoom = toFloat(std::exp(values.at(Channel_LogZoom))),
        .timeLapse = toFloat(std::exp(values.at(Channel_LogTimeLapse))),
    };
}

SmoothCurve CameraPath::createCurve(std::vector<CameraWaypoint> const& waypoints, float smoothing)
{
    std::vector<SmoothCurve::Keyframe> keyframes;
    auto time = 0.0;
    for (auto const& waypoint : waypoints) {
        if (!keyframes.empty()) {
            time += std::max(MinDuration, waypoint.duration);
        }
        keyframes.emplace_back(SmoothCurve::Keyframe{
            .time = time,
            .values =
                {
                    waypoint.center.x,
                    waypoint.center.y,
                    std::log(std::max(MinPositiveValue, waypoint.zoom)),
                    std::log(std::max(MinPositiveValue, waypoint.timeLapse)),
                },
        });
    }
    return SmoothCurve(keyframes, smoothing);
}

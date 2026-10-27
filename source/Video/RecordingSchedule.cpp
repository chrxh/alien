#include "RecordingSchedule.h"

#include <algorithm>
#include <cmath>

namespace
{
    // Durations that are multiples of the frame duration and whole numbers of time steps should not lose a frame or a time step by rounding
    // errors
    auto constexpr RoundingTolerance = 1.0e-6;
}

RecordingSchedule::RecordingSchedule(CameraPath const& path, int fps, double timestepsPerSecond)
    : _path(path)
    , _fps(std::max(1, fps))
{
    auto numFrames = toInt(std::floor(toDouble(path.getDuration()) * _fps + RoundingTolerance)) + 1;
    _timestepsUntilFrame.reserve(numFrames);
    _timestepsUntilFrame.emplace_back(0);

    auto timesteps = 0.0;
    for (int frame = 0; frame + 1 < numFrames; ++frame) {
        // The time-lapse in the middle between two frames approximates its mean over them
        auto timeLapse = path.getState((toDouble(frame) + 0.5) / _fps).timeLapse;
        timesteps += timestepsPerSecond * timeLapse / _fps;
        _timestepsUntilFrame.emplace_back(static_cast<uint64_t>(std::floor(timesteps + RoundingTolerance)));
    }
}

int RecordingSchedule::getNumFrames() const
{
    return toInt(_timestepsUntilFrame.size());
}

CameraState RecordingSchedule::getCameraState(int frame) const
{
    return _path.getState(toDouble(frame) / _fps);
}

uint64_t RecordingSchedule::getTimestepsAfterFrame(int frame) const
{
    if (frame + 1 >= getNumFrames()) {
        return 0;
    }
    return _timestepsUntilFrame.at(frame + 1) - _timestepsUntilFrame.at(frame);
}

uint64_t RecordingSchedule::getTotalTimesteps() const
{
    return _timestepsUntilFrame.back();
}

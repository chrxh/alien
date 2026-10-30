#pragma once

#include <cstdint>
#include <vector>

#include "CameraPath.h"

// Frames of a video that records a camera path. Between two frames, the simulation advances by the time steps that the time-lapse prescribes.
class RecordingSchedule
{
public:
    // At a time-lapse of 1, the video shows the given number of time steps per second
    RecordingSchedule(CameraPath const& path, int fps, double timestepsPerSecond);

    int getNumFrames() const;
    CameraState getCameraState(int frame) const;

    // Time steps from the frame to the next one. Fractions of time steps are carried over to the following frames.
    uint64_t getTimestepsAfterFrame(int frame) const;
    uint64_t getTotalTimesteps() const;

private:
    CameraPath _path;
    int _fps = 1;

    // Time steps from the first frame to each frame
    std::vector<uint64_t> _timestepsUntilFrame;
};

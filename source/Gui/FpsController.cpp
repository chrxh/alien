#include "FpsController.h"

void _FpsController::processForceFps(int fps)
{
    auto callTimepoint = std::chrono::steady_clock::now();
    if (_lastCallTimepoint) {
        auto desiredDuration = std::chrono::milliseconds(1000 / fps);
        auto actualDuration = std::chrono::duration_cast<std::chrono::milliseconds>(callTimepoint - *_lastCallTimepoint);
        auto remainingTime = desiredDuration - actualDuration;
        if (remainingTime.count() > 0) {
            while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - callTimepoint) < remainingTime) {
            }
        }
    }
    _lastCallTimepoint = callTimepoint;
}

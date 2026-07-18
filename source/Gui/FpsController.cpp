#include "FpsController.h"

#include <thread>

void FpsController::processForceFps(int fps)
{
    auto callTimepoint = std::chrono::steady_clock::now();
    if (_lastCallTimepoint) {
        auto desiredDuration = std::chrono::milliseconds(1000 / fps);
        auto actualDuration = std::chrono::duration_cast<std::chrono::milliseconds>(callTimepoint - *_lastCallTimepoint);
        auto remainingTime = desiredDuration - actualDuration;
        if (remainingTime.count() > 0) {
            auto sleepEnd = callTimepoint + remainingTime - std::chrono::milliseconds(1);
            if (sleepEnd > callTimepoint) {
                std::this_thread::sleep_until(sleepEnd);
            }
            while (std::chrono::steady_clock::now() < callTimepoint + remainingTime) {
            }
        }
    }
    _lastCallTimepoint = std::chrono::steady_clock::now();
}

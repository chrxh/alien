#pragma once

#include <chrono>

#include "Definitions.h"

class _FpsController
{
public:
    void processForceFps(int fps);

private:
    std::optional<std::chrono::steady_clock::time_point> _lastCallTimepoint;
};
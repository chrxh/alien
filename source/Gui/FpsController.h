#pragma once

#include <chrono>

#include "Base/Singleton.h"

#include "Definitions.h"

class FpsController
{
    MAKE_SINGLETON(FpsController);

public:
    void processForceFps(int fps);

private:
    std::optional<std::chrono::steady_clock::time_point> _lastCallTimepoint;
};
#pragma once

#include <thread>

class ActiveSimulatorReporter
{
public:
    ActiveSimulatorReporter();

private:
    std::jthread _reportThread;
};

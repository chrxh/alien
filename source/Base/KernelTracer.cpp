#include "KernelTracer.h"

#include <iomanip>
#include <iostream>

void KernelTracer::traceBegin(char const* name)
{
    if (!_enabled) {
        return;
    }
    std::lock_guard lock(_mutex);
    std::cout << "[" << std::setw(10) << ++_callIndex << " | timestep " << _timestep << "] " << std::left << std::setw(56) << name << std::right << std::flush;
}

void KernelTracer::traceEnd(std::chrono::steady_clock::duration duration)
{
    if (!_enabled) {
        return;
    }
    auto milliseconds = std::chrono::duration<double, std::milli>(duration).count();

    std::lock_guard lock(_mutex);
    std::cout << "done " << std::fixed << std::setprecision(3) << std::setw(10) << milliseconds << " ms" << std::endl;
}

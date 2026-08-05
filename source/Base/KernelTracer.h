#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>

#include "Singleton.h"

// Prints a trace line for every kernel call. Only active when enabled; fed by the kernel-call macros in debug mode,
// where each launch is followed by a stream sync. The kernel name is written and flushed before the launch and the
// completion is appended afterwards, so a kernel that never returns leaves its name as the last unfinished line.
class KernelTracer
{
    MAKE_SINGLETON(KernelTracer);

public:
    void setEnabled(bool value) { _enabled = value; }
    bool isEnabled() const { return _enabled; }

    void setTimestep(uint64_t value) { _timestep = value; }

    void traceBegin(char const* name);
    void traceEnd(std::chrono::steady_clock::duration duration);

private:
    bool _enabled = false;
    uint64_t _timestep = 0;
    std::mutex _mutex;
    uint64_t _callIndex = 0;
};

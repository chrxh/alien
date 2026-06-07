#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Singleton.h"

// Lightweight per-kernel profiler.
//
// It is only active when explicitly enabled (see setEnabled). The recording is meant to be driven from the
// stream kernel-call macros while debug mode is active: there each kernel launch is followed by a stream
// synchronization, so the measured wall-clock interval around a launch corresponds to that kernel's execution
// (plus a small, constant launch/sync overhead). The accumulated times therefore give a faithful ranking of
// where the simulation spends its GPU time.
class KernelProfiler
{
    MAKE_SINGLETON(KernelProfiler);

public:
    void setEnabled(bool value) { _enabled = value; }
    bool isEnabled() const { return _enabled; }

    void record(char const* name, std::chrono::steady_clock::time_point start);

    void reset();
    std::string getReport() const;

private:
    struct Entry
    {
        uint64_t count = 0;
        double totalNanoseconds = 0.0;
    };

    bool _enabled = false;
    mutable std::mutex _mutex;
    std::unordered_map<std::string, Entry> _entries;
};

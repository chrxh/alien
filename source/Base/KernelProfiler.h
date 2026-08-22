#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Singleton.h"

// Accumulates per-kernel wall-clock durations. Only active in debug mode; fed by the kernel-call macros, where each
// launch is followed by a stream sync so the recorded duration is the kernel's execution time. The report is written
// to the profile file in regular intervals and again when the process shuts down, so it is available even if the
// process is killed.
class KernelProfiler
{
    MAKE_SINGLETON(KernelProfiler);

public:
    ~KernelProfiler();

    // Starts a fresh measurement; may be called again to restart it
    void init(std::filesystem::path const& filename);
    void close();
    bool isEnabled() const { return _enabled; }

    void record(char const* name, std::chrono::steady_clock::duration duration);

    std::string getReport() const;

private:
    std::string createReport() const;
    void writeReport() const;

    struct Entry
    {
        uint64_t count = 0;
        double totalNanoseconds = 0.0;
    };

    bool _enabled = false;
    std::filesystem::path _filename;
    mutable std::mutex _mutex;
    std::unordered_map<std::string, Entry> _entries;
    std::chrono::steady_clock::time_point _lastWriteTimepoint;
};

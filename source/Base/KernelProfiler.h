#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Singleton.h"

enum class KernelCategory
{
    Simulation,
    Rendering,
    Other
};
auto constexpr NumKernelCategories = 3;

// Accumulates per-kernel wall-clock durations, separated by category. Only active in debug mode; fed by the
// kernel-call macros, where each launch is followed by a stream sync so the recorded duration is the kernel's
// execution time. The report is written to the profile file in regular intervals and again when the process shuts
// down, so it is available even if the process is killed.
class KernelProfiler
{
    MAKE_SINGLETON(KernelProfiler);

public:
    // Assigns every kernel call in its scope to a category and restores the previous one afterwards. Kernels outside
    // of any scope count as KernelCategory::Other. The category is kept per thread, since rendering runs on the GUI
    // thread while the simulation runs on the GPU worker thread.
    class CategoryScope
    {
    public:
        explicit CategoryScope(KernelCategory category);
        ~CategoryScope();

        CategoryScope(CategoryScope const&) = delete;
        CategoryScope& operator=(CategoryScope const&) = delete;

    private:
        KernelCategory _previousCategory;
    };

    ~KernelProfiler();

    // Starts a fresh measurement; may be called again to restart it
    void init(std::filesystem::path const& filename);
    void close();
    bool isEnabled() const { return _enabled; }

    void record(char const* name, std::chrono::steady_clock::duration duration);

    std::string getReport() const;

private:
    struct Entry
    {
        uint64_t count = 0;
        double totalNanoseconds = 0.0;
    };
    using Entries = std::unordered_map<std::string, Entry>;

    std::string createReport() const;
    void writeReport() const;
    static Entry sumUp(Entries const& entries);

    static thread_local KernelCategory _category;

    bool _enabled = false;
    std::filesystem::path _filename;
    mutable std::mutex _mutex;
    std::array<Entries, NumKernelCategories> _entriesByCategory;
    std::chrono::steady_clock::time_point _lastWriteTimepoint;
};

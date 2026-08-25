#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Singleton.h"

enum class KernelCategory
{
    Simulation,
    Rendering,
    Other,
    Count
};

// Accumulates per-kernel wall-clock durations by category. Only active in debug mode. The report is rewritten
// regularly, so it survives a killed process.
class KernelProfiler
{
    MAKE_SINGLETON(KernelProfiler);

public:
    // Assigns the kernel calls in its scope to a category. The category is kept per thread.
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

    void init(std::filesystem::path const& filename);
    void close();
    bool isEnabled() const { return _enabled; }

    void record(char const* name, std::chrono::steady_clock::duration duration, int numBlocks, int threadsPerBlock);

    // Key/value line printed above the rankings; an existing key is overwritten.
    void setReportEntry(std::string const& key, std::string const& value);

    std::string getReport() const;

private:
    struct Entry
    {
        uint64_t count = 0;
        double totalNanoseconds = 0.0;
        int numBlocks = 0;
        int threadsPerBlock = 0;
    };
    using Entries = std::unordered_map<std::string, Entry>;

    std::string createReport() const;
    void writeReport() const;
    static Entry sumUp(Entries const& entries);

    static thread_local KernelCategory _category;

    bool _enabled = false;
    std::filesystem::path _filename;
    mutable std::mutex _mutex;
    std::array<Entries, static_cast<int>(KernelCategory::Count)> _entriesByCategory;
    std::vector<std::pair<std::string, std::string>> _reportEntries;
    std::chrono::steady_clock::time_point _lastWriteTimepoint;
};

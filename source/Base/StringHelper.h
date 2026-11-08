#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <Base/MathTypes.h>

class StringHelper
{
public:
    static std::string format(uint64_t n, char separator = ',');
    static std::string format(float v, int decimalsAfterPoint);
    static std::string format(double v, int decimalsAfterPoint);
    static std::string format(std::chrono::seconds duration);
    static std::string format(std::chrono::milliseconds duration);
    static std::string format(std::chrono::system_clock::time_point const& timePoint);
    static std::string formatUtc(std::chrono::system_clock::time_point const& timePoint);  // e.g. "2026-10-07 18:19:05", like the server timestamps
    static std::string formatInHex(uint64_t value);
    static std::string formatHexColor(FloatColorRGB const& color);
    static std::string formatInThousands(double value);                               // e.g. 12000 -> "12K", 1000000 -> "1,000K"
    static std::string formatCompact(uint64_t value);                                 // e.g. 950 -> "950", 125000 -> "125K", 11380000 -> "11.4M"
    static std::string formatFileSize(uint64_t bytes);                                // e.g. 900 -> "900 B", 333447168 -> "318 MB", 6549825126 -> "6.1 GB"
    static std::string formatCount(uint64_t count, std::string const& singularNoun);  // e.g. "1 state", "1,200 states"
    static std::string encodeBase64(std::string_view data);
    static std::string join(std::vector<int> const& values);
    static std::string formatRanges(std::vector<int> const& sortedValues);
    static std::string formatEnumeration(std::vector<std::string> const& items);

    static void copy(char* target, int maxSize, std::string const& source);
    static bool compare(char const* target, int maxSize, char const* source);

    static bool containsCaseInsensitive(std::string const& str, std::string const& toMatch);
    static bool naturalLess(std::string const& left, std::string const& right);  // Ignores the case and compares numbers by value: "run 9" < "Run 10"
    static std::string toUpper(std::string const& str);
    static std::string truncate(std::string const& str, size_t maxLength, size_t maxLines);  // Appends "..." if shortened

    static std::filesystem::path toPath(std::string const& utf8);
    static std::string toUtf8(std::filesystem::path const& path);

    struct Decomposition
    {
        std::string beforeMatch;
        std::string match;
    };
    static Decomposition decomposeCaseInsensitiveMatch(std::string const& str, std::string const& toMatch);
};
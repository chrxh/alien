#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

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
    static std::string formatInHex(uint64_t value);
    static std::string formatHexColor(FloatColorRGB const& color);
    static std::string formatInThousands(double value);  // e.g. 12000 -> "12K", 1000000 -> "1,000K"
    static std::string encodeBase64(std::string_view data);

    static void copy(char* target, int maxSize, std::string const& source);
    static bool compare(char const* target, int maxSize, char const* source);

    static bool containsCaseInsensitive(std::string const& str, std::string const& toMatch);
    static std::string toUpper(std::string const& str);
    static std::string truncate(std::string const& str, size_t maxLength, size_t maxLines);  // Appends "..." if shortened

    struct Decomposition
    {
        std::string beforeMatch;
        std::string match;
    };
    static Decomposition decomposeCaseInsensitiveMatch(std::string const& str, std::string const& toMatch);
};
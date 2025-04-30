#include "StringHelper.h"

#include <algorithm>
#include <sstream>
#include <iomanip>

std::string StringHelper::format(uint64_t n, char separator)
{
    std::string result;

    std::string s = std::to_string(n);
    do {
        int len = std::min(3, static_cast<int>(s.length()));
        if (result.empty()) {
            result = s.substr(s.length() - len, len);
        } else {
            result = s.substr(s.length() - len, len) + separator + result;
        }
        s = s.substr(0, s.length() - len);
    } while (!s.empty());

    return result;
}

std::string StringHelper::format(float v, int fracPartDecimals)
{
    std::string result;
    if (v < 0) {
        result = "-";
        v = -v;
    }
    result += format(static_cast<uint64_t>(v));
    if (fracPartDecimals > 0) {
        result += ".";
    }
    while (fracPartDecimals-- > 0) {
        v *= 10;
        result += std::to_string(static_cast<uint64_t>(v) % 10);
    }
    return result;
}

namespace
{
    template<typename Time_t>
    std::string formatIntern(Time_t duration)
    {
        auto months = std::chrono::duration_cast<std::chrono::months>(duration);
        duration -= months;
        auto days = std::chrono::duration_cast<std::chrono::days>(duration);
        duration -= days;
        auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hours;
        auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= minutes;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= seconds;

        std::ostringstream oss;
        if (months.count() > 0) {
            oss << std::setw(2) << std::setfill('0') << months.count() << ":";
        }
        if (days.count() > 0 || months.count() > 0) {
            oss << std::setw(2) << std::setfill('0') << days.count() << ":";
        }
        if (hours.count() > 0 || days.count() > 0 || months.count() > 0) {
            oss << std::setw(2) << std::setfill('0') << hours.count() << ":";
        }
        oss << std::setw(2) << std::setfill('0') << minutes.count() << ":";
        oss << std::setw(2) << std::setfill('0') << seconds.count();
        if (std::is_same_v<Time_t, std::chrono::milliseconds>) {
            oss << ".";
            oss << std::setw(3) << std::setfill('0') << duration.count();
        }

        return oss.str();
    }
}

std::string StringHelper::format(std::chrono::seconds duration)
{
    return formatIntern(duration);
}

std::string StringHelper::format(std::chrono::milliseconds duration)
{
    return formatIntern(duration);
}

std::string StringHelper::format(std::chrono::system_clock::time_point const& timePoint)
{
    std::time_t time_t = std::chrono::system_clock::to_time_t(timePoint);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");

    return ss.str();
}

void StringHelper::copy(char* target, int maxSize, std::string const& source)
{
    auto sourceSize = source.size();
    if (sourceSize >= maxSize) {
        sourceSize = maxSize - 1;
    }
    source.copy(target, sourceSize);
    target[sourceSize] = 0;
}

bool StringHelper::compare(char const* target, int maxSize, char const* source)
{
    for (int i = 0; i < maxSize; ++i) {
        if (target[i] != source[i]) {
            return false;
        }
        if (source[i] == 0) {
            break;
        }
    }
    return true;
}

bool StringHelper::containsCaseInsensitive(std::string const& str, std::string const& toMatch)
{
    std::string strLower = str;
    std::string toMatchLower = toMatch;

    std::transform(str.begin(), str.end(), strLower.begin(), ::tolower);
    std::transform(toMatch.begin(), toMatch.end(), toMatchLower.begin(), ::tolower);

    return strLower.find(toMatchLower) != std::string::npos;
}

StringHelper::Decomposition StringHelper::decomposeCaseInsensitiveMatch(std::string const& str, std::string const& toMatch)
{
    std::string strLower = str;
    std::string toMatchLower = toMatch;

    std::transform(str.begin(), str.end(), strLower.begin(), ::tolower);
    std::transform(toMatch.begin(), toMatch.end(), toMatchLower.begin(), ::tolower);

    auto findResult = strLower.find(toMatchLower);
    if (findResult == std::string::npos) {
        return {.beforeMatch = str};
    }

    return {.beforeMatch = str.substr(0, findResult), .match = str.substr(findResult, toMatch.size())};
}

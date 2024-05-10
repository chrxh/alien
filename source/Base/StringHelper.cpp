#include "StringHelper.h"

#include <algorithm>
#include <sstream>
#include <iomanip>

std::string StringHelper::format(uint64_t n)
{
    std::string result;

    std::string s = std::to_string(n);
    do {
        int len = std::min(3, static_cast<int>(s.length()));
        if (result.empty()) {
            result = s.substr(s.length() - len, len);
        } else {
            result = s.substr(s.length() - len, len) + "," + result;
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
        result += std::to_string(static_cast<int>(v) % 10);
    }
    return result;
}

std::string StringHelper::format(std::chrono::milliseconds duration)
{
    if (duration.count() == 0) {
        return "0";
    }
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
    if (minutes.count() > 0 || hours.count() > 0 || days.count() > 0 || months.count() > 0) {
        oss << std::setw(2) << std::setfill('0') << minutes.count() << ":";
    }
    if (seconds.count() > 0 || minutes.count() > 0 || hours.count() > 0 || days.count() > 0 || months.count() > 0) {
        oss << std::setw(2) << std::setfill('0') << seconds.count() << ".";
    }
    oss << std::setw(3) << std::setfill('0') << duration.count();

    return oss.str();
}

void StringHelper::copy(char* target, int targetSize, std::string const& source)
{
    auto sourceSize = source.size();
    if (sourceSize < targetSize) {
        source.copy(target, sourceSize);
        target[sourceSize] = 0;
    } else {
        target[0] = 0;
    }
}

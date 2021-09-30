#include "StringFormatter.h"

#include <algorithm>

std::string StringFormatter::format(uint64_t n)
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

std::string StringFormatter::format(float v, int decimals)
{
    std::string result;
    if (v < 0) {
        result = "-";
        v = -v;
    }
    result += format(static_cast<uint64_t>(v)) + ".";
    while (decimals-- > 0) {
        v *= 10;
        result += std::to_string(static_cast<int>(v) % 10);
    }
    return result;
}

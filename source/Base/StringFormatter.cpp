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

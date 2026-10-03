#include "StringHelper.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ranges>
#include <sstream>
#include <format>

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
    return format(static_cast<double>(v), fracPartDecimals);
}

std::string StringHelper::format(double v, int fracPartDecimals)
{
    std::string result;
    if (v < 0) {
        result = "-";
        v = -v;
    }
    auto scale = static_cast<uint64_t>(std::llround(std::pow(10.0, fracPartDecimals)));
    auto scaled = static_cast<uint64_t>(std::llround(v * scale));
    result += format(scaled / scale);
    if (fracPartDecimals > 0) {
        result += ".";
        auto fracPart = std::to_string(scaled % scale);
        result += std::string(fracPartDecimals - static_cast<int>(fracPart.length()), '0') + fracPart;
    }
    return result;
}

namespace
{
    template <typename Time_t>
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

std::string StringHelper::formatInHex(uint64_t value)
{
    std::stringstream ss;
    ss << "0x" << std::hex << std::uppercase << value;
    return ss.str();
}

std::string StringHelper::formatHexColor(FloatColorRGB const& color)
{
    auto toByte = [](float value) { return std::clamp(static_cast<int>(std::lround(value * 255.0f)), 0, 255); };
    return std::format("#{:02x}{:02x}{:02x}", toByte(color.r), toByte(color.g), toByte(color.b));
}

std::string StringHelper::encodeBase64(std::string_view data)
{
    static auto constexpr Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string result;
    result.reserve((data.size() + 2) / 3 * 4);
    for (auto chunk : data | std::views::chunk(3)) {
        uint32_t block = 0;
        for (auto [byteIndex, byte] : std::views::enumerate(chunk)) {
            block |= static_cast<uint32_t>(static_cast<uint8_t>(byte)) << (16 - 8 * byteIndex);
        }
        for (auto charIndex : std::views::iota(0, 4)) {
            result.push_back(charIndex <= std::ssize(chunk) ? Alphabet[(block >> (18 - 6 * charIndex)) & 0x3f] : '=');
        }
    }
    return result;
}

std::string StringHelper::formatInThousands(double value)
{
    if (value == 0.0) {
        return "0";
    }
    auto result = format(value / 1000, 3);
    if (result.find('.') != std::string::npos) {
        while (result.back() == '0') {
            result.pop_back();
        }
        if (result.back() == '.') {
            result.pop_back();
        }
    }
    return result + "K";
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

std::string StringHelper::toUpper(std::string const& str)
{
    std::string result = str;
    std::transform(str.begin(), str.end(), result.begin(), ::toupper);
    return result;
}

std::string StringHelper::truncate(std::string const& str, size_t maxLength, size_t maxLines)
{
    auto length = std::min(str.size(), maxLength);
    auto newline = str.find('\n');
    for (auto line = size_t{1}; line < maxLines && newline < length; ++line) {
        newline = str.find('\n', newline + 1);
    }
    length = std::min(length, newline);
    if (length == str.size()) {
        return str;
    }
    while (length > 0 && (static_cast<unsigned char>(str.at(length)) & 0xc0) == 0x80) {  // Skip UTF-8 continuation bytes
        --length;
    }
    return str.substr(0, length) + "...";
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

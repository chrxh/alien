#include "VersionChecker.h"

#include <optional>
#include <vector>

#include <boost/range/adaptors.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/range/adaptor/sliced.hpp>
#include <boost/algorithm/string.hpp>

#include "Base/Resources.h"

bool VersionChecker::isVersionValid(std::string const& s)
{
    std::vector<std::string> versionParts;
    boost::split(versionParts, s, boost::is_any_of("."));
    if (versionParts.size() != 3 && versionParts.size() != 5) {
        return false;
    }
    try {
        for (auto const& versionPart : versionParts | boost::adaptors::sliced(0, 3)) {
            static_cast<void>(std::stoi(versionPart));
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace 
{
    enum class VersionType
    {
        Alpha,
        Beta,
        Release
    };
    struct VersionParts
    {
        int major;
        int minor;
        int patch;
        VersionType versionType; 
        std::optional<int> preRelease;
    };
    VersionParts getVersionParts(std::string const& s)
    {
        std::vector<std::string> versionParts;
        boost::split(versionParts, s, boost::is_any_of("."));
        VersionParts result{
            .major = std::stoi(versionParts.at(0)),
            .minor = std::stoi(versionParts.at(1)),
        };

        if (versionParts.size() == 3) {
            result.patch = std::stoi(versionParts.at(2));
            result.versionType = VersionType::Release;
        }
        if (versionParts.size() == 5) {
            result.patch = 0;
            if (versionParts.at(3) == "alpha") {
                result.versionType = VersionType::Alpha;
            } else if (versionParts.at(3) == "beta") {
                result.versionType = VersionType::Beta;
            } else {
                std::runtime_error("Unexpected version number.");
            }
            result.preRelease = std::stoi(versionParts.at(4));
        }
        return result;
    }
}
bool VersionChecker::isVersionOutdated(std::string const& s)
{
    auto otherParts = getVersionParts(s);
    auto ownParts = getVersionParts(Const::ProgramVersion);
    if (otherParts.major < ownParts.major) {
        return true;
    }
    if (otherParts.major == 4 && otherParts.versionType == VersionType::Alpha && *otherParts.preRelease < 2) {
        return true;
    }
    return false;
}

bool VersionChecker::isVersionNewer(std::string const& s)
{
    return false;
}

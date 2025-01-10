#include "VersionParserService.h"

#include <optional>
#include <vector>

#include <boost/range/adaptors.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/range/adaptor/sliced.hpp>
#include <boost/algorithm/string.hpp>

#include "Base/Resources.h"

bool VersionParserService::isVersionValid(std::string const& otherVersionString)
{
    try {
        std::vector<std::string> versionParts;
        boost::split(versionParts, otherVersionString, boost::is_any_of(".-"));
        if (versionParts.size() != 3 && versionParts.size() != 5) {
            return false;
        }
        for (auto const& versionPart : versionParts | boost::adaptors::sliced(0, 3)) {
            static_cast<void>(std::stoi(versionPart));
        }
        if (versionParts.size() == 5) {
            if (versionParts[3] != "alpha" && versionParts[3] != "beta") {
                return false;
            }
            static_cast<void>(std::stoi(versionParts[4]));
        }
    } catch (...) {
        return false;
    }
    return true;
}

VersionParserService::VersionParts VersionParserService::getVersionParts(std::string const& versionString)
{
    if (versionString.empty()) {
        return VersionParts();
    }

    std::vector<std::string> versionParts;
    boost::split(versionParts, versionString, boost::is_any_of(".-"));
    VersionParts result{
        .major = std::stoi(versionParts.at(0)),
        .minor = std::stoi(versionParts.at(1)),
    };

    if (versionParts.size() == 3) {
        result.patch = std::stoi(versionParts.at(2));
        result.versionType = VersionType_Release;
    }
    if (versionParts.size() == 5) {
        result.patch = 0;
        if (versionParts.at(3) == "alpha") {
            result.versionType = VersionType_Alpha;
        } else if (versionParts.at(3) == "beta") {
            result.versionType = VersionType_Beta;
        } else {
            std::runtime_error("Unexpected version number.");
        }
        result.preRelease = std::stoi(versionParts.at(4));
    }
    return result;
}

bool VersionParserService::isVersionOutdated(std::string const& otherVersionString)
{
    auto otherParts = getVersionParts(otherVersionString);
    auto ownParts = getVersionParts(Const::ProgramVersion);
    if (otherParts.major < ownParts.major) {
        return true;
    }
    if (otherParts.major == 4 && otherParts.versionType == VersionType_Alpha && *otherParts.preRelease < 2) {
        return true;
    }
    return false;
}

bool VersionParserService::isVersionNewer(std::string const& otherVersionString)
{
    auto otherParts = getVersionParts(otherVersionString);
    auto ownParts = getVersionParts(Const::ProgramVersion);
    if (otherParts.major > ownParts.major) {
        return true;
    }
    if (otherParts.major < ownParts.major) {
        return false;
    }
    if (otherParts.versionType > ownParts.versionType) {
        return true;
    }
    if (otherParts.versionType < ownParts.versionType) {
        return false;
    }
    if (otherParts.versionType == VersionType_Alpha || otherParts.versionType == VersionType_Beta) {
        if (otherParts.preRelease > ownParts.preRelease) {
            return true;
        }
        if (otherParts.preRelease < ownParts.preRelease) {
            return false;
        }
    }
    if (otherParts.versionType == VersionType_Release) {
        if (otherParts.minor > ownParts.minor) {
            return true;
        }
        if (otherParts.minor < ownParts.minor) {
            return false;
        }
        if (otherParts.patch > ownParts.patch) {
            return true;
        }
        if (otherParts.patch < ownParts.patch) {
            return false;
        }
    }
    return false;
}

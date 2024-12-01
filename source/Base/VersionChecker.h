#pragma once
#include <optional>
#include <string>

class VersionChecker
{
public:
    static bool isVersionValid(std::string const& otherVersionString);
    static bool isVersionOutdated(std::string const& otherVersionString);
    static bool isVersionNewer(std::string const& otherVersionString);

    using VersionType = int;
    enum VersionType_
    {
        VersionType_Alpha,
        VersionType_Beta,
        VersionType_Release
    };
    struct VersionParts
    {
        int major = 0;
        int minor = 0;
        int patch = 0;
        VersionType versionType = VersionType_Release;
        std::optional<int> preRelease;
    };
    static VersionParts getVersionParts(std::string const& versionString);
};


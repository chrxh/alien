#pragma once
#include <string>

class VersionChecker
{
public:
    static bool isVersionValid(std::string const& otherVersionString);
    static bool isVersionOutdated(std::string const& otherVersionString);
    static bool isVersionNewer(std::string const& otherVersionString);
};


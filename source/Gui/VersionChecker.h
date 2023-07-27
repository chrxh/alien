#pragma once
#include <string>

class VersionChecker
{
public:
    static bool isVersionValid(std::string const& s);
    static bool isVersionOutdated(std::string const& s);
    static bool isVersionNewer(std::string const& s);
};


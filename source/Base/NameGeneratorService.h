#pragma once

#include <deque>
#include <functional>
#include <string>
#include <unordered_set>

#include <Base/Singleton.h>

class NameGeneratorService
{
    MAKE_SINGLETON(NameGeneratorService);

public:
    std::string createSimulationName();
    std::string createGenomeName(std::unordered_set<std::string> const& usedNames = {});

private:
    std::string createUniqueName(std::function<std::string()> const& createName, std::unordered_set<std::string> const& usedNames);

    std::deque<std::string> _recentNames;
};

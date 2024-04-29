#pragma once

#include <string>
#include <unordered_set>

#include "Network/Definitions.h"

class LastSessionBrowserData
{
public:
    void load(std::unordered_set<NetworkResourceRawTO> const& rawTOs);
    void save();
    bool isNew(NetworkResourceRawTO const& rawTO) const;

    void registrate(NetworkResourceRawTO const& rawTO);

private:
    std::unordered_set<std::string> convertToIdentifiers(std::unordered_set<NetworkResourceRawTO> const& rawTOs) const;
    std::string convertToIdentifier(NetworkResourceRawTO const& rawTO) const;

    std::unordered_set<std::string> _identifiers;
    std::unordered_set<std::string> _newIdentifiers;
};

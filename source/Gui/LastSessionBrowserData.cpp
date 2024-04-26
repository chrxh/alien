#include "LastSessionBrowserData.h"

#include "Base/GlobalSettings.h"
#include "Network/NetworkResourceRawTO.h"

void LastSessionBrowserData::load(std::unordered_set<NetworkResourceRawTO> const& rawTOs)
{
    auto currentIdentifiers = convertToIdentifiers(rawTOs);
    auto lastIdentifiers = GlobalSettings::getInstance().getStringVector(
        "windows.browser.last session.simulation ids", std::vector(currentIdentifiers.begin(), currentIdentifiers.end()));
    _identifiers = std::unordered_set(lastIdentifiers.begin(), lastIdentifiers.end());
}

void LastSessionBrowserData::save(std::unordered_set<NetworkResourceRawTO> const& rawTOs)
{
    auto currentIdentifiers = convertToIdentifiers(rawTOs);
    GlobalSettings::getInstance().setStringVector(
        "windows.browser.last session.simulation ids", std::vector(currentIdentifiers.begin(), currentIdentifiers.end()));
}

bool LastSessionBrowserData::isNew(NetworkResourceRawTO const& rawTO) const
{
    return !_identifiers.count(convertToIdentifier(rawTO));
}

std::unordered_set<std::string> LastSessionBrowserData::convertToIdentifiers(std::unordered_set<NetworkResourceRawTO> const& rawTOs) const
{
    std::unordered_set<std::string> result;
    for (auto const& rawTO : rawTOs) {
        result.insert(convertToIdentifier(rawTO));
    }
    return result;
}

std::string LastSessionBrowserData::convertToIdentifier(NetworkResourceRawTO const& rawTO) const
{
    return rawTO->id + "/" + rawTO->timestamp;
}

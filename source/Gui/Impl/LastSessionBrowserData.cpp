#include "LastSessionBrowserData.h"

#include <chrono>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <Network/NetworkResourceRawTO.h>

namespace
{
    auto constexpr NewPeriod = std::chrono::days(3);
}

void LastSessionBrowserData::load()
{
    auto lastIdentifiers = GlobalSettings::get().getValue("windows.browser.last session.simulation ids", std::vector<std::string>());
    _identifiers = std::unordered_set(lastIdentifiers.begin(), lastIdentifiers.end());
    _identifiers.erase("");
    _firstSession = _identifiers.empty();
}

void LastSessionBrowserData::save(std::unordered_set<NetworkResourceRawTO> const& rawTOs)
{
    // Without a previous session, everything that already exists counts as seen
    if (_firstSession) {
        _newIdentifiers.merge(convertToIdentifiers(rawTOs));
    }
    _identifiers.insert(_newIdentifiers.begin(), _newIdentifiers.end());
    GlobalSettings::get().setValue("windows.browser.last session.simulation ids", std::vector(_identifiers.begin(), _identifiers.end()));
}

bool LastSessionBrowserData::isNew(NetworkResourceRawTO const& rawTO) const
{
    auto isUnseen = !_firstSession && !_identifiers.contains(convertToIdentifier(rawTO));
    return isUnseen || rawTO->timestamp >= StringHelper::formatUtc(std::chrono::system_clock::now() - NewPeriod);
}

void LastSessionBrowserData::registrate(NetworkResourceRawTO const& rawTO)
{
    _newIdentifiers.insert(convertToIdentifier(rawTO));
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

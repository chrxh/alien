#pragma once

#include <string>

#include <Network/NetworkResourceRawTO.h>

#include <Persister/Interface/LocalLibrary.h>

#include "Definitions.h"

class _MenuBarInfoWidget
{
public:
    static MenuBarInfoWidget create();

    void process();

private:
    _MenuBarInfoWidget() = default;

    void processCurrentSimulation();
    void processServerSimulation(NetworkResourceRawTO const& rawTO);
    void processLocalSimulation(LibraryFile const& file);
    void processServerTooltip(NetworkResourceRawTO const& rawTO, std::string const& displayName);
    void processLocalTooltip(LibraryFile const& file, std::string const& displayName);
    void processHeartButton(NetworkResourceRawTO const& rawTO, std::string const& displayName);

    bool _localSimulationShown = false;
};

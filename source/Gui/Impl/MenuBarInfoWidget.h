#pragma once

#include <string>

#include <Network/NetworkResourceRawTO.h>

#include "Definitions.h"

class _MenuBarInfoWidget
{
public:
    static MenuBarInfoWidget create();

    void process();

private:
    _MenuBarInfoWidget() = default;

    void processCurrentSimulation();
    void processNameButton(NetworkResourceRawTO const& rawTO, std::string const& displayName, float width);
    void processHeartButton(NetworkResourceRawTO const& rawTO, std::string const& displayName);
};

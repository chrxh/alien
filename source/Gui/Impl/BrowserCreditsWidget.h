#pragma once

#include <string>

#include <Network/NetworkResourceRawTO.h>

#include "Definitions.h"

class _BrowserCreditsWidget
{
public:
    static BrowserCreditsWidget create(BrowserData const& data);

    void process();

private:
    _BrowserCreditsWidget(BrowserData const& data);

    void processNameButton(NetworkResourceRawTO const& rawTO, std::string const& displayName, float width);
    void processHeartButton(NetworkResourceRawTO const& rawTO, std::string const& displayName);

    BrowserData _data;
};

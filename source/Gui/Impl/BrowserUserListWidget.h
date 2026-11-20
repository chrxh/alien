#pragma once

#include "Definitions.h"

class _BrowserUserListWidget
{
public:
    static BrowserUserListWidget create(BrowserData const& data);

    void process();

private:
    _BrowserUserListWidget(BrowserData const& data);

    void processSortSpecs(bool isGpuVisible);

    BrowserData _data;
};

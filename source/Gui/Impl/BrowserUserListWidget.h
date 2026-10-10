#pragma once

#include "Definitions.h"

class _BrowserUserListWidget
{
public:
    static BrowserUserListWidget create(BrowserData const& data);

    void process();

private:
    _BrowserUserListWidget(BrowserData const& data);

    BrowserData _data;
};

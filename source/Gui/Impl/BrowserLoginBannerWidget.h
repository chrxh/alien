#pragma once

#include "Definitions.h"

class _BrowserLoginBannerWidget
{
public:
    static BrowserLoginBannerWidget create();

    void process();

    bool isVisible() const;

private:
    _BrowserLoginBannerWidget() = default;

    bool _dismissed = false;
};

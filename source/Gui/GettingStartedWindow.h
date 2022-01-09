#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _GettingStartedWindow
{
public:
    _GettingStartedWindow();

    ~_GettingStartedWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    bool _on = false;
    bool _showAfterStartup = true;
};
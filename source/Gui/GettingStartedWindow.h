#pragma once

#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _GettingStartedWindow
{
public:
    _GettingStartedWindow(StyleRepository const& styleRepository);

    ~_GettingStartedWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    StyleRepository _styleRepository;

    bool _on = false;
    bool _showAfterStartup = true;
};
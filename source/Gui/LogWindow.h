#pragma once

#include "Definitions.h"

class _LogWindow
{
public:
    _LogWindow(StyleRepository const& styleRepository, GuiLogger const& logger);

    void process();


    bool isOn() const;
    void setOn(bool value);

private:
    bool _on = false;
    bool _verbose = false;

    StyleRepository _styleRepository;
    GuiLogger _logger;
};

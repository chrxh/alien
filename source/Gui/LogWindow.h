#pragma once

#include "Definitions.h"

class _LogWindow
{
public:
    _LogWindow(SimpleLogger const& logger);
    ~_LogWindow();

    void process();


    bool isOn() const;
    void setOn(bool value);

private:
    bool _on = false;
    bool _verbose = false;

    SimpleLogger _logger;
};

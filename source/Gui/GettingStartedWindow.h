#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _GettingStartedWindow : public _AlienWindow
{
public:
    _GettingStartedWindow();

    virtual ~_GettingStartedWindow();

private:
    void processIntern() override;

    void headline(std::string const& text);
    void itemText(std::string const& text);
    void openWeblink(std::string const& link);

    bool _showAfterStartup = true;
};


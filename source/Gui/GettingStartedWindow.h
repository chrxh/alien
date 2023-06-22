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

    void drawTitle();
    void drawHeadline(std::string const& text);
    void drawItemText(std::string const& text);
    void openWeblink(std::string const& link);

    bool _showAfterStartup = true;
};


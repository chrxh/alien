#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _GettingStartedWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(_GettingStartedWindow);

public:
    void init();
    void shutdown();

private:
    _GettingStartedWindow();

    void processIntern() override;

    void drawTitle();
    void drawHeading1(std::string const& text);
    void drawHeading2(std::string const& text);
    void drawItemText(std::string const& text);
    void drawParagraph(std::string const& text);

    bool _showAfterStartup = true;
};



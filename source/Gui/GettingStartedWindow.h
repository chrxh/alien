#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class GettingStartedWindow : public AlienWindow<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GettingStartedWindow);

private:
    GettingStartedWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;

    void drawTitle();
    void drawHeading1(std::string const& text);
    void drawHeading2(std::string const& text);
    void drawItemText(std::string const& text);
    void drawParagraph(std::string const& text);

    bool _showAfterStartup = true;
};



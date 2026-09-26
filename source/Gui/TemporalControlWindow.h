#pragma once

#include <Base/Singleton.h>

#include "AlienWindow.h"
#include "Definitions.h"

class TemporalControlWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(TemporalControlWindow);

private:
    TemporalControlWindow();

    void initIntern() override;
    void processIntern();

    void processTpsInfo();
    void processTotalTimestepsInfo();
    void processRealTimeInfo();
    void processTpsRestriction();

    void processToolbar();

    int _tpsRestriction = 100;
};

#pragma once

#include <Base/Singleton.h>

#include "CreatorController.h"
#include "Definitions.h"

class CreatorWidget
{
    MAKE_SINGLETON(CreatorWidget);

public:
    void process();

private:
    void processColorWidget();
    void processPencilWidget(CreatorParameters& parameters);
    void processMaterialWidgets(CreatorParameters& parameters);
    void processShapeWidgets(CreatorParameters& parameters);
    void processObjectDistanceWidget(CreatorParameters& parameters);
    void processStickyWidget(CreatorParameters& parameters);
    void processStaticWidget(CreatorParameters& parameters);
    void processPointButtons();
};

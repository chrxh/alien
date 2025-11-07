#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienDialog.h"
#include "Definitions.h"

class ResizeWorldDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ResizeWorldDialog);

public:
    void open() override;

private:
    ResizeWorldDialog();

    void initIntern() override;
    void processIntern() override;

    void onResizing();

    bool _scaleContent = false;
    int _width = 0;
    int _height = 0;
};

#pragma once

#include "AlienWindow.h"
#include "Definitions.h"

class _AutosaveWindow : public _AlienWindow
{
public:
    _AutosaveWindow(SimulationController const& simController);
    ~_AutosaveWindow();

private:
    void processIntern() override;

    void processHeader();
    void processTable();

    void validationAndCorrection();

    SimulationController _simController;

    using SaveMode = int;
    enum SaveMode_
    {
        SaveMode_Circular,
        SaveMode_Unlimited
    };
    SaveMode _origSaveMode = SaveMode_Circular;
    SaveMode _saveMode = SaveMode_Circular;
    int _origNumberOfFiles = 20;
    int _numberOfFiles = 20;
};
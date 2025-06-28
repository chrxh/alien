#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/CudaSettings.h"

#include "AlienDialog.h"
#include "Definitions.h"

class GpuSettingsDialog : public AlienDialog<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GpuSettingsDialog);

private:
    GpuSettingsDialog();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    void openIntern();

    void validateAndCorrect(CudaSettings& settings) const;

    SimulationFacade _simulationFacade;

    CudaSettings _gpuSettings;
};
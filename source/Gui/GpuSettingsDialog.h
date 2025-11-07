#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/Definitions.h>

#include "AlienDialog.h"
#include "Definitions.h"

class GpuSettingsDialog : public AlienDialog<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GpuSettingsDialog);

private:
    GpuSettingsDialog();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void openIntern();

    void validateAndCorrect(CudaSettings& settings) const;

    CudaSettings _gpuSettings;
};

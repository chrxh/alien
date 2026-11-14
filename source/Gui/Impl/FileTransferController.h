#pragma once

#include <Base/Singleton.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Persister/Interface/PersisterFacade.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class FileTransferController : public MainLoopEntity
{
    MAKE_SINGLETON(FileTransferController);

public:
    void onOpenSimulationDialog();
    void onSaveSimulationDialog();

    void onOpenGenomeDialog(std::function<void(GenomeDesc const&)> const& openFunc);
    void onSaveGenomeDialog(GenomeDesc const& genome, std::function<void()> const& afterSaveFunc = nullptr);

    void onOpenCameraFlightDialog(std::function<void()> const& afterOpenFunc);
    void onSaveCameraFlightDialog();

private:
    void init() override;
    void process() override;
    void shutdown() override;

    std::string _referencePath;
};

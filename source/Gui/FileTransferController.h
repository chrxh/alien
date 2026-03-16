#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/PersisterFacade.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class FileTransferController : public MainLoopEntity
{
    MAKE_SINGLETON(FileTransferController);

public:
    void onOpenSimulationDialog();
    void onOpenSimulation(std::filesystem::path const& filename);
    void onSaveSimulationDialog();

    void onOpenGenomeDialog(std::function<void(GenomeDesc const&)> const& openFunc);
    void onSaveGenomeDialog(GenomeDesc const& genome, std::function<void()> const& saveFunc = nullptr);

private:
    void init() override;
    void process() override;
    void shutdown() override;

    TaskProcessor _openSimulationProcessor;
    TaskProcessor _saveSimulationProcessor;

    std::string _referencePath;
};

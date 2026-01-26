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

    std::string getReferencePath() const { return _referencePath; }
    void setReferencePath(std::string const& path) { _referencePath = path; }

private:
    void init() override;
    void process() override;
    void shutdown() override;

    TaskProcessor _openSimulationProcessor;
    TaskProcessor _saveSimulationProcessor;

    std::string _referencePath;
};

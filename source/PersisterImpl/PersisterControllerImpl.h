#pragma once

#include <thread>

#include "PersisterInterface/PersisterController.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _PersisterControllerImpl : public _PersisterController
{
public:
    ~_PersisterControllerImpl() override;

    void init(SimulationController const& simController) override;
    void shutdown() override;

    bool isBusy() const override;
    PersisterRequestState getJobState(PersisterRequestId const& id) const override;
    std::vector<PersisterErrorInfo> fetchAllErrorInfos(SenderId const& senderId) override;
    PersisterErrorInfo fetchError(PersisterRequestId const& id) override;

    PersisterRequestId scheduleSaveSimulationToFile(SenderInfo const& senderInfo, std::string const& filename, float const& zoom, RealVector2D const& center) override;
    SavedSimulationData fetchSavedSimulationData(PersisterRequestId const& id) override;

    PersisterRequestId scheduleLoadSimulationFromFile(SenderInfo const& senderInfo, std::string const& filename) override;

private:
    static auto constexpr MaxWorkerThreads = 4;

    PersisterRequestId generateNewJobId();

    PersisterWorker _worker;
    std::thread* _thread[MaxWorkerThreads];
    int _latestJobId = 0;
};

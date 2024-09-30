#pragma once

#include <chrono>
#include <variant>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "LoadedSimulationResultData.h"
#include "LoadSimulationRequestData.h"
#include "PersisterErrorInfo.h"
#include "PersisterRequestId.h"
#include "PersisterRequestState.h"
#include "SavedSimulationResultData.h"
#include "SaveSimulationRequestData.h"
#include "SenderId.h"
#include "SenderInfo.h"

class _PersisterController
{
public:
    virtual ~_PersisterController() = default;

    virtual void init(SimulationController const& simController) = 0;
    virtual void shutdown() = 0;

    //generic logic
    virtual bool isBusy() const = 0;
    virtual PersisterRequestState getRequestState(PersisterRequestId const& id) const = 0;
    virtual std::vector<PersisterErrorInfo> fetchAllErrorInfos(SenderId const& senderId) = 0;
    virtual PersisterErrorInfo fetchError(PersisterRequestId const& id) = 0;

    //specific request
    virtual PersisterRequestId scheduleSaveSimulationToFile(SenderInfo const& senderInfo, SaveSimulationRequestData const& data) = 0;
    virtual SavedSimulationResultData fetchSavedSimulationData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleLoadSimulationFromFile(SenderInfo const& senderInfo, LoadSimulationRequestData const& data) = 0;
    virtual LoadedSimulationResultData fetchLoadSimulationData(PersisterRequestId const& id) = 0;
};

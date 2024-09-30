#pragma once

#include <chrono>
#include <variant>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "PersisterErrorInfo.h"
#include "PersisterRequestId.h"
#include "PersisterRequestState.h"
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
    virtual PersisterRequestState getJobState(PersisterRequestId const& id) const = 0;
    virtual std::vector<PersisterErrorInfo> fetchAllErrorInfos(SenderId const& senderId) = 0;
    virtual PersisterErrorInfo fetchError(PersisterRequestId const& id) = 0;

    //specific logic
    virtual PersisterRequestId scheduleSaveSimulationToFile(SenderInfo const& senderInfo, std::string const& filename, float const& zoom, RealVector2D const& center) = 0;
    struct SavedSimulationData
    {
        std::string name;
        uint64_t timestep = 0;
        std::chrono::system_clock::time_point timestamp;
    };
    virtual SavedSimulationData fetchSavedSimulationData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleLoadSimulationFromFile(SenderInfo const& senderInfo, std::string const& filename) = 0;
};

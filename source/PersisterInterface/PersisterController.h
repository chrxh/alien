#pragma once

#include <chrono>
#include <variant>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "DownloadNetworkResourceRequestData.h"
#include "DownloadNetworkResourceResultData.h"
#include "GetNetworkResourcesRequestData.h"
#include "GetNetworkResourcesResultData.h"
#include "LoginRequestData.h"
#include "LoginResultData.h"
#include "ReadSimulationResultData.h"
#include "ReadSimulationRequestData.h"
#include "PersisterErrorInfo.h"
#include "PersisterRequestId.h"
#include "PersisterRequestState.h"
#include "ReplaceNetworkResourceRequestData.h"
#include "ReplaceNetworkResourceResultData.h"
#include "SaveSimulationResultData.h"
#include "SaveSimulationRequestData.h"
#include "SenderId.h"
#include "SenderInfo.h"
#include "UploadNetworkResourceRequestData.h"
#include "UploadNetworkResourceResultData.h"

class _PersisterController
{
public:
    virtual ~_PersisterController() = default;

    virtual void init(SimulationController const& simController) = 0;
    virtual void shutdown() = 0;
    virtual void restart() = 0;

    //generic logic
    virtual bool isBusy() const = 0;
    virtual PersisterRequestState getRequestState(PersisterRequestId const& id) const = 0;
    virtual std::vector<PersisterErrorInfo> fetchAllErrorInfos(SenderId const& senderId) = 0;
    virtual PersisterErrorInfo fetchError(PersisterRequestId const& id) = 0;

    //specific request
    virtual PersisterRequestId scheduleSaveSimulationToFile(SenderInfo const& senderInfo, SaveSimulationRequestData const& data) = 0;
    virtual SaveSimulationResultData fetchSavedSimulationData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleReadSimulationFromFile(SenderInfo const& senderInfo, ReadSimulationRequestData const& data) = 0;
    virtual ReadSimulationResultData fetchReadSimulationData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleLogin(SenderInfo const& senderInfo, LoginRequestData const& data) = 0;
    virtual LoginResultData fetchLoginData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleGetNetworkResources(SenderInfo const& senderInfo, GetNetworkResourcesRequestData const& data) = 0;
    virtual GetNetworkResourcesResultData fetchGetNetworkResourcesData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleDownloadNetworkResource(SenderInfo const& senderInfo, DownloadNetworkResourceRequestData const& data) = 0;
    virtual DownloadNetworkResourceResultData fetchDownloadNetworkResourcesData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleUploadNetworkResource(SenderInfo const& senderInfo, UploadNetworkResourceRequestData const& data) = 0;
    virtual UploadNetworkResourceResultData fetchUploadNetworkResourcesData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleReplaceNetworkResource(SenderInfo const& senderInfo, ReplaceNetworkResourceRequestData const& data) = 0;
    virtual ReplaceNetworkResourceResultData fetchReplaceNetworkResourcesData(PersisterRequestId const& id) = 0;
};

#pragma once

#include <chrono>
#include <variant>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "DeleteNetworkResourceRequestData.h"
#include "DeleteNetworkResourceResultData.h"
#include "DownloadNetworkResourceRequestData.h"
#include "DownloadNetworkResourceResultData.h"
#include "EditNetworkResourceRequestData.h"
#include "EditNetworkResourceResultData.h"
#include "GetNetworkResourcesRequestData.h"
#include "GetNetworkResourcesResultData.h"
#include "GetUserNamesForEmojiRequestData.h"
#include "GetUserNamesForEmojiResultData.h"
#include "LoginRequestData.h"
#include "LoginResultData.h"
#include "MoveNetworkResourceRequestData.h"
#include "MoveNetworkResourceResultData.h"
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
#include "ToggleLikeNetworkResourceRequestData.h"
#include "ToggleLikeNetworkResourceResultData.h"
#include "UploadNetworkResourceRequestData.h"
#include "UploadNetworkResourceResultData.h"

class _PersisterController
{
public:
    virtual ~_PersisterController() = default;

    //lifecycle control
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

    virtual PersisterRequestId scheduleGetUserNamesForEmoji(SenderInfo const& senderInfo, GetUserNamesForEmojiRequestData const& data) = 0;
    virtual GetUserNamesForEmojiResultData fetchGetUserNamesForEmojiData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleDeleteNetworkResource(SenderInfo const& senderInfo, DeleteNetworkResourceRequestData const& data) = 0;
    virtual DeleteNetworkResourceResultData fetchDeleteNetworkResourcesData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleEditNetworkResource(SenderInfo const& senderInfo, EditNetworkResourceRequestData const& data) = 0;
    virtual EditNetworkResourceResultData fetchEditNetworkResourcesData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleMoveNetworkResource(SenderInfo const& senderInfo, MoveNetworkResourceRequestData const& data) = 0;
    virtual MoveNetworkResourceResultData fetchMoveNetworkResourcesData(PersisterRequestId const& id) = 0;

    virtual PersisterRequestId scheduleToggleLikeNetworkResource(SenderInfo const& senderInfo, ToggleLikeNetworkResourceRequestData const& data) = 0;
    virtual ToggleLikeNetworkResourceResultData fetchToggleLikeNetworkResourcesData(PersisterRequestId const& id) = 0;
};

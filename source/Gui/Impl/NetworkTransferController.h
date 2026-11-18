#pragma once

#include <functional>

#include <Base/Singleton.h>

#include <Engine/Interface/Definitions.h>
#include <Engine/Interface/SimulationFacade.h>

#include <Persister/Interface/Definitions.h>
#include <Persister/Interface/DeleteNetworkResourceRequestData.h>
#include <Persister/Interface/DownloadNetworkResourceRequestData.h>
#include <Persister/Interface/PersisterFacade.h>
#include <Persister/Interface/PersisterRequestId.h>
#include <Persister/Interface/ReplaceNetworkResourceRequestData.h>
#include <Persister/Interface/UploadNetworkResourceRequestData.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class NetworkTransferController : public MainLoopEntity
{
    MAKE_SINGLETON(NetworkTransferController);

public:
    void onDownload(DownloadNetworkResourceRequestData const& requestData);
    void onUpload(UploadNetworkResourceRequestData const& requestData, std::function<void()> const& afterUploadFunc = nullptr);
    void onReplace(ReplaceNetworkResourceRequestData const& requestData, std::function<void()> const& afterReplaceFunc = nullptr);
    void onDelete(DeleteNetworkResourceRequestData const& requestData);
    void onEdit(EditNetworkResourceRequestData const& requestData);
    void onMove(MoveNetworkResourceRequestData const& requestData);

    // Uploads, name changes and deletions that the server has not finished yet, after which the uploads of the user carry other names
    bool hasPendingNameChanges() const;

private:
    void init() override;
    void process() override;
    void shutdown() override {}

    TaskProcessor _downloadProcessor;
    TaskProcessor _uploadProcessor;
    TaskProcessor _replaceProcessor;
    TaskProcessor _deleteProcessor;
    TaskProcessor _editProcessor;
    TaskProcessor _moveProcessor;
};

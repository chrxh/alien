
add_library(PersisterInterface
    Definitions.h
    DeleteNetworkResourceRequestData.h
    DeleteNetworkResourceResultData.h
    DownloadCache.h
    DownloadNetworkResourceRequestData.h
    DownloadNetworkResourceResultData.h
    EditNetworkResourceRequestData.h
    EditNetworkResourceResultData.h
    GetNetworkResourcesRequestData.h
    GetNetworkResourcesResultData.h
    GetUserNamesForEmojiRequestData.h
    GetUserNamesForEmojiResultData.h
    LoginRequestData.h
    LoginResultData.h
    MoveNetworkResourceRequestData.h
    MoveNetworkResourceResultData.h
    PersisterErrorInfo.h
    PersisterFacade.h
    PersisterRequestId.h
    PersisterRequestState.h
    ReadSimulationRequestData.h
    ReadSimulationResultData.h
    ReplaceNetworkResourceRequestData.h
    ReplaceNetworkResourceResultData.h
    SaveSimulationRequestData.h
    SaveSimulationResultData.h
    SenderId.h
    SenderInfo.h
    TaskProcessor.cpp
    TaskProcessor.h
    ToggleLikeNetworkResourceRequestData.h
    ToggleLikeNetworkResourceResultData.h
    UploadNetworkResourceRequestData.h
    UploadNetworkResourceResultData.h)

target_link_libraries(PersisterInterface Base)

target_link_libraries(PersisterInterface Boost::boost)

if (MSVC)
    target_compile_options(PersisterInterface PRIVATE "/MP")
endif()

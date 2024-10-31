
add_library(PersisterInterface
    AuxiliaryData.h
    AuxiliaryDataParserService.cpp
    AuxiliaryDataParserService.h
    Definitions.h
    DeleteNetworkResourceRequestData.h
    DeleteNetworkResourceResultData.h
    DeserializedSimulation.h
    DownloadCache.h
    DownloadNetworkResourceRequestData.h
    DownloadNetworkResourceResultData.h
    EditNetworkResourceRequestData.h
    EditNetworkResourceResultData.h
    GetNetworkResourcesRequestData.h
    GetNetworkResourcesResultData.h
    GetSimulationRequestData.h
    GetSimulationResultData.h
    GetUserNamesForReactionRequestData.h
    GetUserNamesForReactionResultData.h
    LegacyAuxiliaryDataParserService.cpp
    LegacyAuxiliaryDataParserService.h
    LoginRequestData.h
    LoginResultData.h
    MoveNetworkResourceRequestData.h
    MoveNetworkResourceResultData.h
    ParameterParser.h
    PersisterErrorInfo.h
    PersisterFacade.h
    PersisterRequestId.h
    PersisterRequestState.h
    ReadSimulationRequestData.h
    ReadSimulationResultData.h
    ReplaceNetworkResourceRequestData.h
    ReplaceNetworkResourceResultData.h
    SavepointTable.h
    SavepointTableService.cpp
    SavepointTableService.h
    SaveSimulationRequestData.h
    SaveSimulationResultData.h
    SenderId.h
    SenderInfo.h
    SerializerService.cpp
    SerializerService.h
    SerializedSimulation.h
    TaskProcessor.cpp
    TaskProcessor.h
    ToggleReactionNetworkResourceRequestData.h
    ToggleReactionNetworkResourceResultData.h
    UploadNetworkResourceRequestData.h
    UploadNetworkResourceResultData.h)

target_link_libraries(PersisterInterface Base)

target_link_libraries(PersisterInterface Boost::boost)
target_link_libraries(PersisterInterface ZLIB::ZLIB)

find_path(ZSTR_INCLUDE_DIRS "zstr.hpp")
target_include_directories(PersisterInterface PRIVATE ${ZSTR_INCLUDE_DIRS})

if (MSVC)
    target_compile_options(PersisterInterface PRIVATE "/MP")
endif()

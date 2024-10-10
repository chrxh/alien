
add_library(PersisterInterface
    Definitions.h
    GetNetworkResourcesRequestData.h
    GetNetworkResourcesResultData.h
    LoginRequestData.h
    LoginResultData.h
    PersisterController.cpp
    PersisterController.h
    PersisterErrorInfo.h
    PersisterRequestId.h
    PersisterRequestState.h
    ReadSimulationRequestData.h
    ReadSimulationResultData.h
    SaveSimulationRequestData.h
    SavedSimulationResultData.h
    SenderId.h
    SenderInfo.h
    TaskProcessor.cpp
    TaskProcessor.h)

target_link_libraries(PersisterInterface Base)

target_link_libraries(PersisterInterface Boost::boost)

if (MSVC)
    target_compile_options(PersisterInterface PRIVATE "/MP")
endif()

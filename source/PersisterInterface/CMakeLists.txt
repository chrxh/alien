
add_library(PersisterInterface
    Definitions.h
    PersisterController.cpp
    PersisterController.h
    PersisterErrorInfo.h
    PersisterRequestId.h
    PersisterRequestState.h
    SenderId.h
    SenderInfo.h)

target_link_libraries(PersisterInterface Base)

target_link_libraries(PersisterInterface Boost::boost)

if (MSVC)
    target_compile_options(PersisterInterface PRIVATE "/MP")
endif()

if (MSVC)
    set(LIB_TYPE OBJECT)
else()
    set(LIB_TYPE STATIC)
endif()

add_library(PersisterImpl ${LIB_TYPE}
    Definitions.h
    PersisterFacadeImpl.cpp
    PersisterFacadeImpl.h
    PersisterRequest.h
    PersisterRequestError.cpp
    PersisterRequestError.h
    PersisterWorker.cpp
    PersisterWorker.h)

target_link_libraries(PersisterImpl PUBLIC Base)
target_link_libraries(PersisterImpl PUBLIC PersisterInterface)

target_link_libraries(PersisterImpl PUBLIC Boost::boost)

if (MSVC)
    target_compile_options(PersisterImpl PRIVATE "/MP")
endif()

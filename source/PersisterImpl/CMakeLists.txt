
add_library(PersisterImpl OBJECT
    Definitions.h
    PersisterFacadeImpl.cpp
    PersisterFacadeImpl.h
    PersisterRequest.h
    PersisterRequestError.cpp
    PersisterRequestError.h
    PersisterWorker.cpp
    PersisterWorker.h)

target_include_directories(PersisterImpl PUBLIC ${CMAKE_SOURCE_DIR}/source ${CMAKE_SOURCE_DIR}/external)
target_link_libraries(PersisterImpl Boost::boost)

if (MSVC)
    target_compile_options(PersisterImpl PRIVATE "/MP")
endif()


add_library(PersisterImpl
    Definitions.h
    PersisterFacadeImpl.cpp
    PersisterFacadeImpl.h
    PersisterRequest.h
    PersisterRequestError.cpp
    PersisterRequestError.h
    PersisterWorker.cpp
    PersisterWorker.h)

target_link_libraries(PersisterImpl Base)
target_link_libraries(PersisterImpl PersisterInterface)

target_link_libraries(PersisterImpl Boost::boost)

target_precompile_headers(PersisterImpl PRIVATE
    <memory>
    <optional>
    <string>
    <vector>
)

if (MSVC)
    target_compile_options(PersisterImpl PRIVATE "/MP")
endif()

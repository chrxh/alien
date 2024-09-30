
add_library(PersisterImpl
    Definitions.h
    PersisterJob.cpp
    PersisterJob.h
    PersisterWorker.cpp
    PersisterWorker.h)

target_link_libraries(PersisterImpl Base)
target_link_libraries(PersisterImpl PersisterInterface)

target_link_libraries(PersisterImpl Boost::boost)

if (MSVC)
    target_compile_options(PersisterImpl PRIVATE "/MP")
endif()

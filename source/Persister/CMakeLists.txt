
add_library(Persister
    Definitions.h
    PersisterJob.cpp
    PersisterJob.h
    PersisterWorker.cpp
    PersisterWorker.h)

target_link_libraries(Persister Boost::boost)

if (MSVC)
    target_compile_options(Persister PRIVATE "/MP")
endif()

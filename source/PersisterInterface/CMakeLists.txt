
add_library(PersisterInterface
    Definitions.h
    PersisterController.cpp
    PersisterController.H)

target_link_libraries(PersisterImpl Base)

target_link_libraries(PersisterInterface Boost::boost)

if (MSVC)
    target_compile_options(PersisterInterface PRIVATE "/MP")
endif()

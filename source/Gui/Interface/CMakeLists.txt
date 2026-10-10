add_library(GuiInterface
    Definitions.h
    GuiFacade.cpp
    GuiFacade.h)

if (MSVC)
    target_compile_options(GuiInterface PRIVATE "/MP")
endif()

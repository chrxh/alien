#.rst:
# CopyImportedTargetProperties
# --------------------------
#
# Copies the `INTERFACE*` and `IMPORTED*` properties from a target
# to another one.
# This function can be used to duplicate an `IMPORTED` or an `ALIAS` library
# with a different name since ``add_library(... ALIAS ...)`` does not work
# for those targets.
#
# ::
#
#   copy_imported_target_properties(<source-target> <destination-target>)
#
# The function copies all the `INTERFACE*` and `IMPORTED*` target
# properties from `<source-target>` to `<destination-target>`.
#
# The function uses the `IMPORTED_CONFIGURATIONS` property to determine
# which configuration-dependent properties should be copied
# (`IMPORTED_LOCATION_<CONFIG>`, etc...)
#
# Example:
#
# Internally the CMake project of ZLIB builds the ``zlib`` and
# ``zlibstatic`` targets which can be exported in the ``ZLIB::`` namespace
# with the ``install(EXPORT ...)`` command.
#
# The config-module will then create the import libraries ``ZLIB::zlib`` and
# ``ZLIB::zlibstatic``. To use ``ZLIB::zlibstatic`` under the standard
# ``ZLIB::ZLIB`` name we need to create the ``ZLIB::ZLIB`` imported library
# and copy the appropriate properties:
#
#   add_library(ZLIB::ZLIB STATIC IMPORTED)
#   copy_imported_target_properties(ZLIB::zlibstatic ZLIB::ZLIB)
#

function(copy_imported_target_properties src_target dest_target)

    set(config_dependent_props
        IMPORTED_IMPLIB
        IMPORTED_LINK_DEPENDENT_LIBRARIES
        IMPORTED_LINK_INTERFACE_LANGUAGES
        IMPORTED_LINK_INTERFACE_LIBRARIES
        IMPORTED_LINK_INTERFACE_MULTIPLICITY
        IMPORTED_LOCATION
        IMPORTED_NO_SONAME
        IMPORTED_SONAME
    )

    # copy configuration-independent properties
    foreach(prop
        ${config_dependent_props}
        IMPORTED_CONFIGURATIONS
        INTERFACE_AUTOUIC_OPTIONS
        INTERFACE_COMPILE_DEFINITIONS
        INTERFACE_COMPILE_FEATURES
        INTERFACE_COMPILE_OPTIONS
        INTERFACE_INCLUDE_DIRECTORIES
        INTERFACE_LINK_LIBRARIES
        INTERFACE_POSITION_INDEPENDENT_CODE
        INTERFACE_SOURCES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    )
        get_property(is_set TARGET ${src_target} PROPERTY ${prop} SET)
        if(is_set)
            get_target_property(v ${src_target} ${prop})
            set_target_properties(${dest_target} PROPERTIES ${prop} "${v}")
            # message(STATUS "set_target_properties(${dest_target} PROPERTIES ${prop} ${v})")
        endif()
    endforeach()

    # copy configuration-dependent properties
    get_target_property(imported_configs ${src_target}
        IMPORTED_CONFIGURATIONS)

    foreach(config ${imported_configs})
        foreach(prop_prefix ${config_dependent_props})
            set(prop ${prop_prefix}_${config})
            get_property(is_set TARGET ${src_target} PROPERTY ${prop} SET)
            if(is_set)
                get_target_property(v ${src_target} ${prop})
                set_target_properties(${dest_target}
                    PROPERTIES ${prop} "${v}")
                # message(STATUS "set_target_properties(${dest_target} PROPERTIES ${prop} ${v})")
            endif()
        endforeach()
    endforeach()
endfunction()

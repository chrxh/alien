
cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

# Some vcpkg ports (e.g. glad) still declare cmake_minimum_required(VERSION <3.5),
# which CMake 4.x rejects. Raise the minimum policy version for the child CMake
# processes that vcpkg spawns to build those ports so configuration succeeds.
if(NOT DEFINED ENV{CMAKE_POLICY_VERSION_MINIMUM})
    set(ENV{CMAKE_POLICY_VERSION_MINIMUM} 3.5)
endif()

# Using the vcpkg submodule
set(CMAKE_TOOLCHAIN_FILE external/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "")

# Default to Debug for single-config generators. Multi-config generators (Visual
# Studio, Ninja Multi-Config, Xcode) select the configuration at build time, so
# CMAKE_BUILD_TYPE must stay empty for them.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_GENERATOR MATCHES "Multi-Config|Visual Studio|Xcode")
    message("Build type not specified, using Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type not specified, using Debug" FORCE)
endif()

option(USE_HIP "Build the GPU engine with HIP for AMD GPUs instead of CUDA" OFF)

set(CMAKE_CXX_STANDARD 23)

# This project does not use C++20 modules. Disabling the module dependency scan
# avoids the per-target dyndep (CXX.dd) generation, which both speeds up the build
# and works around a Ninja assertion (build.cc:423 "edge && !edge->outputs_ready()")
# hit by the Ninja Multi-Config generator while processing those dyndep files.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(USE_HIP)
  # No CMAKE_HIP_ARCHITECTURES pin: enable_language(HIP) below auto-detects the
  # host GPU(s) and errors if none is found (pass -DCMAKE_HIP_ARCHITECTURES to
  # override). It is de-duplicated right after enable_language so the custom HIP
  # link command (which embeds --offload-arch=${CMAKE_HIP_ARCHITECTURES}) gets a
  # clean value on multi-GCD hosts.
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_SEPARABLE_COMPILATION ON)
  # -ffast-math is a GCC/clang gcc-driver flag; clang-cl uses /clang:-ffast-math.
  # Defer the flag until after project() where CMAKE_HIP_COMPILER_FRONTEND_VARIANT
  # is known (set below after enable_language(HIP)).
else()
  # Default CUDA target architectures
  # DIAGNOSTIC: 120 (Blackwell) added to check whether the hanging constructor kernel is an artifact of the PTX JIT
  # compilation that a compute_75-only build goes through on such a GPU. A Blackwell card now runs native SASS instead,
  # while every other card keeps working via the compute_75 PTX. Building 120 requires CUDA 12.8 or newer.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 120)
  endif()

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --use-local-env -use_fast_math")

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
  endif()
endif()

project(alien-project LANGUAGES C CXX)
if(USE_HIP)
  enable_language(HIP)
  list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES)
  # The custom Windows link commands below embed the offload targets literally, so
  # every architecture needs its own --offload-arch flag: a bare
  # ${CMAKE_HIP_ARCHITECTURES} would expand to one semicolon-joined argument.
  # All listed targets end up in the same fat binary, so a single executable runs
  # on all of them -- the generic targets (gfx11-generic, ...) each already cover a
  # whole GPU family with one code object.
  set(_HIP_OFFLOAD_ARCH_FLAGS "")
  foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
    string(APPEND _HIP_OFFLOAD_ARCH_FLAGS "--offload-arch=${_arch} ")
  endforeach()
  string(STRIP "${_HIP_OFFLOAD_ARCH_FLAGS}" _HIP_OFFLOAD_ARCH_FLAGS)
  if(WIN32)
    # The link commands below run external/hip/hip_link_win.py. Use the interpreter
    # CMake locates rather than a bare "python3", which a Windows installation
    # usually does not provide under that name.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
  endif()
else()
  enable_language(CUDA)
endif()

if(USE_HIP)
  # Detect clang-cl (MSVC-frontend) vs clang++ (GCC-frontend).
  # CMAKE_HIP_COMPILER_FRONTEND_VARIANT is set after enable_language(HIP).
  if(CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # clang-cl does not accept -ffast-math; use the pass-through form.
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} /clang:-ffast-math")
    # Force-include: clang-cl requires /FI<path> (no space, MSVC style).
    set(_HIP_FORCE_INCLUDE "/FI${CMAKE_CURRENT_SOURCE_DIR}/external/hip/cuda_to_hip.h")
    set(_CXX_FORCE_INCLUDE "/FI${CMAKE_CURRENT_SOURCE_DIR}/external/hip/cuda_to_hip.h")
    if(WIN32)
      # On Windows, clang-cl -fgpu-rdc invokes clang-offload-bundler with the
      # same path as both bundler input and output (the /Fo<path> the driver
      # wrote the host object to). Windows refuses to write over an active
      # memory-mapped file (ERROR_USER_MAPPED_FILE, 0x4C8). With -o <path>
      # instead of /Fo<path> the driver routes the host object to a distinct
      # temp file so input != output; the bundler can rewrite the output.
      # Override the HIP compile rule to use -o <OBJECT> on this platform.
      # See ROCm/TheRock#5615.
      string(REPLACE "/Fo<OBJECT>" "-o <OBJECT>"
        CMAKE_HIP_COMPILE_OBJECT "${CMAKE_HIP_COMPILE_OBJECT}")

      # The MSVC-mode HIP link rule (cmake -E vs_link_exe -- lld-link) does not
      # do the -fgpu-rdc device-link step. Override it to use the GCC-driver
      # clang.exe (not clang-cl) with a Windows MSVC target: the GCC driver
      # unbundles HIP fat objects, runs the amdgcn device link, and embeds the
      # resulting code object. clang-cl in link mode does not do this.
      # --offload-arch is required at link time to select the correct target.
      #
      # The GCC-driver clang.exe does not accept bare MSVC-style linker flags
      # (/machine:x64, /subsystem:console, etc.) -- it treats them as file
      # paths. external/hip/hip_link_win.py wraps them with -Xlinker automatically.
      get_filename_component(_HIP_BIN_DIR "${CMAKE_HIP_COMPILER}" DIRECTORY)
      set(_CLANG_GCC "${_HIP_BIN_DIR}/clang.exe")
      set(_HIP_LINK_WRAP "${CMAKE_CURRENT_SOURCE_DIR}/external/hip/hip_link_win.py")
      # ROCm 7.13 amd_hip_cooperative_groups.h defines this_cluster() as a
      # non-inline __device__ function -- each TU that includes the header
      # gets a definition, causing duplicate-symbol errors in the -fgpu-rdc
      # device link. Pass --allow-multiple-definition to the amdgcn linker
      # (-Xoffload-linker) so it keeps the first definition silently.
      set(CMAKE_HIP_LINK_EXECUTABLE
        "\"${Python3_EXECUTABLE}\" ${_HIP_LINK_WRAP} ${_CLANG_GCC} --target=x86_64-pc-windows-msvc -fms-compatibility ${_HIP_OFFLOAD_ARCH_FLAGS} <OBJECTS> -fgpu-rdc --hip-link -Xoffload-linker --allow-multiple-definition -o <TARGET> <LINK_FLAGS> <LINK_LIBRARIES>")
    endif()
  else()
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffast-math")
    set(_HIP_FORCE_INCLUDE "-include${CMAKE_CURRENT_SOURCE_DIR}/external/hip/cuda_to_hip.h")
    set(_CXX_FORCE_INCLUDE "-include${CMAKE_CURRENT_SOURCE_DIR}/external/hip/cuda_to_hip.h")
    if(WIN32)
      # CMake injects -fuse-ld=lld-link (Windows-Clang platform default) into
      # <LINK_FLAGS>; this flag conflicts with --hip-link (HIP device-link mode)
      # on the GCC-driver clang++. Also, bare .lib names and MSVC-style /flags
      # in <LINK_FLAGS> need -Xlinker wrapping for the GCC driver.
      # Override CMAKE_HIP_LINK_EXECUTABLE to route through hip_link_win.py
      # (which strips -fuse-ld=lld-link and wraps MSVC flags / .lib names) and
      # add --allow-multiple-definition for the ROCm 7.13 this_cluster() issue.
      set(_HIP_LINK_WRAP "${CMAKE_CURRENT_SOURCE_DIR}/external/hip/hip_link_win.py")
      set(CMAKE_HIP_LINK_EXECUTABLE
        "\"${Python3_EXECUTABLE}\" ${_HIP_LINK_WRAP} ${CMAKE_HIP_COMPILER} ${_HIP_OFFLOAD_ARCH_FLAGS} <OBJECTS> -fgpu-rdc --hip-link -Xoffload-linker --allow-multiple-definition -o <TARGET> <LINK_FLAGS> <LINK_LIBRARIES>")
    endif()
  endif()
endif()

include_directories(
    source
    external)

if(USE_HIP)
    find_package(hip REQUIRED)

    # ROCm ships no CUDA-named toolkit headers; the forwarding shims in
    # external/hip/hip_compat (cuda_runtime.h, cooperative_groups.h, ...) stand in for
    # them on the HIP build only. Prepend the shim dir (BEFORE) so its
    # cuda/helper_cuda.h shadows the project's vendored external/cuda/helper_cuda.h,
    # which does not parse under hipcc. The CUDA include resolution is unchanged.
    # The parent external/hip/ dir is also prepended so the shims resolve the force-included
    # header as <cuda_to_hip.h>. All HIP-only sources live under external/hip/, never source/.
    include_directories(BEFORE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/hip/hip_compat
        ${CMAKE_CURRENT_SOURCE_DIR}/external/hip)

    # The HIP runtime headers (pulled by the force-included compat header) must
    # be on the path for host C++ TUs too, which do not link hip::host.
    include_directories(${hip_INCLUDE_DIRS})

    # Force-include the single CUDA->HIP compat header on every HIP translation
    # unit and on host C++ TUs that include cuda/HIP headers (e.g. cuda_fp16.h
    # in EngineImpl). On Windows/clang-cl, the force-include of hip_runtime.h
    # onto plain CXX TUs (Base, EngineInterface, PersisterInterface) causes
    # header pollution: hip_runtime.h pulls <cmath>/<algorithm> early, which
    # instantiates std:: templates against forward-declared project types before
    # their full definitions are visible. The CXX files in those non-engine libs
    # do not reference CUDA/HIP types directly, so skipping the force-include
    # there is safe. The one CXX file that does (EngineImpl/DescConverterService)
    # is retagged LANGUAGE HIP and receives the force-include via the HIP path.
    add_compile_options($<$<COMPILE_LANGUAGE:HIP>:${_HIP_FORCE_INCLUDE}>)
    if(NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      # Linux / gcc-driver clang: force-include on CXX too (no header pollution).
      add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${_CXX_FORCE_INCLUDE}>)
    endif()
    add_compile_definitions(USE_HIP)
    # Windows SDK minwindef.h defines min/max macros that conflict with std::max
    # in HIP TUs. The CXX NOMINMAX below does not cover HIP language TUs.
    add_compile_definitions($<$<COMPILE_LANGUAGE:HIP>:NOMINMAX>)
endif()

if(MSVC AND NOT USE_HIP)
    # Pure MSVC (cl.exe) CUDA build: suppress macro-redefinition warnings,
    # treat CXX warnings as errors, enable conforming preprocessor.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-wd4005>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-WX>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
elseif(MSVC AND USE_HIP)
    # clang-cl HIP build on Windows: suppress the macro-redefinition warning.
    # Do NOT add -WX or /Zc:preprocessor -- clang-cl does not support
    # /Zc:preprocessor and rejects it as an unknown argument; without -WX the
    # subdir /MP targets and vcpkg-injected flags are mere warnings, not errors.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-wd4005>)
    # Suppress clang-cl "argument unused" warnings for flags added by subdirs
    # (e.g. /MP, /bigobj from target_compile_options in subdir CMakeLists).
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-command-line-argument>)
elseif(NOT USE_HIP)
    # Linux GCC/clang++ CUDA build: treat warnings as errors.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()
add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:NOMINMAX>)

if(NOT USE_HIP)
    # Treat all NVCC (CUDA) warnings as errors
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)
endif()

# Helper for the subdir CMakeLists: on the HIP build retag a target's .cu
# sources LANGUAGE HIP and enable relocatable device code. EngineKernels holds
# all device code, including __constant__ symbols (ConstantMemory.cu) used by
# device code in sibling translation units, so it must be compiled -fgpu-rdc and
# device-linked as one unit -- mirroring the CUDA path's
# CUDA_SEPARABLE_COMPILATION + CUDA_RESOLVE_DEVICE_SYMBOLS. A no-op on CUDA.
function(alien_mark_hip_target target)
    if(USE_HIP)
        get_target_property(_srcs ${target} SOURCES)
        if(_srcs)
            foreach(_src ${_srcs})
                if(_src MATCHES "\\.cu$")
                    set_source_files_properties(${_src} TARGET_DIRECTORY ${target} PROPERTIES LANGUAGE HIP)
                endif()
            endforeach()
        endif()
        set_target_properties(${target} PROPERTIES
            HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
            HIP_SEPARABLE_COMPILATION ON
            HIP_RESOLVE_DEVICE_SYMBOLS ON)
        # CMake 3.31 does not reliably inject -fgpu-rdc from the property for the
        # HIP language; add it to the .cu compiles explicitly.
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
    endif()
endfunction()

# Executables that link the relocatable-device-code libraries must perform the
# HIP device link themselves: -fgpu-rdc makes the HIP linker emit the fatbin and
# its registration symbols (__hip_fatbin/__hip_gpubin_handle) that the rdc
# objects reference. CMake 3.31 does not add this from HIP_RESOLVE_DEVICE_SYMBOLS
# on the linked libraries, so set it on each consumer. A no-op on CUDA.
function(alien_link_hip_rdc target)
    if(USE_HIP)
        target_link_options(${target} PRIVATE -fgpu-rdc --hip-link)
        if(WIN32 AND CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            # MSVC-frontend path: the clang.exe link wrapper (hip_link_win.py)
            # does not auto-add Windows system import libraries. Pass them as
            # bare .lib names; hip_link_win.py wraps them with -Xlinker.
            # GCC-frontend path: CMake's Windows-Clang platform rules already
            # inject -lkernel32 -luser32 ... via LINK_LIBRARIES, so no extras.
            target_link_options(${target} PRIVATE
                advapi32.lib shell32.lib
                gdi32.lib user32.lib
                ole32.lib oleaut32.lib
                uuid.lib winspool.lib
                comdlg32.lib)
        endif()
    endif()
endfunction()

# Emit host debug info (nvcc -g) only for debug-capable configurations. On MSVC,
# -g forces the host compiler to generate debug info that is serialized through
# mspdbsrv, which noticeably slows down Release compilation for no runtime benefit.
add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug,RelWithDebInfo>>:-g>)

add_executable(alien)
add_executable(cli)
add_executable(EngineTests)
add_executable(EngineInterfaceTests)
add_executable(NetworkTests)
add_executable(PersisterTests)

# find_package(CUDAToolkit REQUIRED) is too slow
find_package(Boost REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(cereal CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(GTest REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

if(USE_HIP)
    # The subdir CMakeLists link the bare "cudart" target. On the HIP build the
    # device runtime comes from the HIP language toolchain; provide a stub that
    # carries only the host HIP runtime so those link lines need no edits.
    if(NOT TARGET cudart)
        add_library(cudart INTERFACE)
        target_link_libraries(cudart INTERFACE hip::host)
    endif()
endif()

add_subdirectory(external/ImFileDialog)
add_subdirectory(source/Base)
add_subdirectory(source/Cli)
add_subdirectory(source/EngineKernels)
add_subdirectory(source/EngineImpl)
add_subdirectory(source/EngineInterface)
add_subdirectory(source/EngineInterfaceTests)
add_subdirectory(source/EngineTestData)
add_subdirectory(source/EngineTests)
add_subdirectory(source/Gui)
add_subdirectory(source/Network)
add_subdirectory(source/NetworkTests)
add_subdirectory(source/PersisterImpl)
add_subdirectory(source/PersisterInterface)
add_subdirectory(source/PersisterTests)
add_subdirectory(source/Shaders)

# Executables linking the HIP relocatable-device-code engine libraries must run
# the device link themselves (see alien_link_hip_rdc).
alien_link_hip_rdc(alien)
alien_link_hip_rdc(cli)
alien_link_hip_rdc(EngineTests)
alien_link_hip_rdc(PersisterTests)

# Copy resources next to the build output so that ALIEN finds them
add_custom_command(
    TARGET alien POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        ${CMAKE_CURRENT_BINARY_DIR}/resources
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        $<TARGET_FILE_DIR:alien>/resources)

# The build output directory doubles as the deployment payload
# (see .github/workflows/windows-nightly-deploy.yml), so every dependency that is
# not part of Windows itself has to end up next to the executables. The vcpkg
# libraries are handled by vcpkg's applocal deployment; the two runtimes below are
# not. The Universal CRT (ucrtbase.dll, api-ms-win-crt-*) ships with Windows 10 and
# newer and is deliberately not deployed.

# The CUDA runtime is linked dynamically, so the executables need cudart64_*.dll
# at run time. It is part of the CUDA toolkit and not of the NVIDIA driver, so a
# machine without the toolkit installed only has it if it is deployed with the
# binaries.
if(WIN32 AND NOT USE_HIP)
    get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
    file(GLOB CUDA_RUNTIME_DLL "${CUDA_BIN_DIR}/cudart64_*.dll")
    if(NOT CUDA_RUNTIME_DLL)
        message(FATAL_ERROR "Could not find cudart64_*.dll in ${CUDA_BIN_DIR}")
    endif()
    add_custom_command(
        TARGET alien POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CUDA_RUNTIME_DLL}
            $<TARGET_FILE_DIR:alien>)
endif()

# The HIP counterpart: the executables link the HIP runtime dynamically and need
# amdhip64_*.dll at run time. It belongs to the HIP SDK, so deploy it alongside
# the binaries for the same reason as the CUDA runtime above.
if(WIN32 AND USE_HIP)
    get_filename_component(HIP_BIN_DIR ${CMAKE_HIP_COMPILER} DIRECTORY)
    file(GLOB HIP_RUNTIME_DLL "${HIP_BIN_DIR}/amdhip64*.dll")
    if(NOT HIP_RUNTIME_DLL)
        message(FATAL_ERROR "Could not find amdhip64*.dll in ${HIP_BIN_DIR}")
    endif()
    add_custom_command(
        TARGET alien POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${HIP_RUNTIME_DLL}
            $<TARGET_FILE_DIR:alien>)
endif()

# The MSVC runtime (msvcp140.dll, vcruntime140*.dll) comes with the Visual C++
# redistributable rather than with Windows, so it can be missing on a machine
# without development tools. InstallRequiredSystemLibraries locates the matching
# redistributable of the compiler in use; _SKIP keeps it from adding install()
# rules, which this project does not use, so it only fills the variable below.
if(MSVC)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
    include(InstallRequiredSystemLibraries)
    if(NOT CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
        message(FATAL_ERROR "Could not locate the MSVC runtime redistributable")
    endif()
    add_custom_command(
        TARGET alien POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
            $<TARGET_FILE_DIR:alien>)
endif()

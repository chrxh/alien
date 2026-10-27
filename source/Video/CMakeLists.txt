add_library(Video
    CameraPath.cpp
    CameraPath.h
    Definitions.h
    RecordingSchedule.cpp
    RecordingSchedule.h
    SmoothCurve.cpp
    SmoothCurve.h
    VideoWriter.cpp
    VideoWriter.h)

target_link_libraries(Video Base)

# Only the GUIDs of Media Foundation are linked, its functions are looked up at run time, see VideoWriter.cpp
if (WIN32)
    target_link_libraries(Video mfuuid)
endif()

if (MSVC)
    target_compile_options(Video PRIVATE "/MP")
endif()

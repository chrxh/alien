target_sources(VideoTests
PUBLIC
    CameraPathTests.cpp
    RecordingScheduleTests.cpp
    SmoothCurveTests.cpp
    Testsuite.cpp
    VideoWriterTests.cpp)

target_link_libraries(VideoTests Base)
target_link_libraries(VideoTests Video)

target_link_libraries(VideoTests GTest::GTest GTest::Main)

target_precompile_headers(VideoTests PRIVATE
    <string>
    <vector>
    <gtest/gtest.h>
)

if (MSVC)
    target_compile_options(VideoTests PRIVATE "/MP")
endif()

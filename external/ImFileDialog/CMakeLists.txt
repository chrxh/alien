if (MSVC)
    set(LIB_TYPE OBJECT)
else()
    set(LIB_TYPE STATIC)
endif()

add_library(im_file_dialog ${LIB_TYPE}
  ImFileDialog.cpp)

target_include_directories(im_file_dialog PUBLIC .)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(im_file_dialog PUBLIC imgui::imgui)

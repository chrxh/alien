
add_library(im_file_dialog
  ImFileDialog.cpp)

target_include_directories(im_file_dialog PUBLIC .)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(im_file_dialog imgui::imgui)

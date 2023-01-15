add_executable (eugene 
  main.cpp
  vk_engine.cpp
  vk_initializers.cpp
  vk_mesh.cpp)

set_property(TARGET eugene PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:eugene>)

target_include_directories(eugene PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${SDL2_INCLUDE_DIRS}")

target_link_libraries(eugene vkbootstrap vma glm tinyobjloader) # imgui stb_image)
target_link_libraries(eugene Vulkan::Vulkan SDL2::SDL2)

set_target_properties(eugene PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

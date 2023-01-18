add_executable (eugene 
  main.cpp
  vk_engine.cpp
  vk_initializers.cpp
  vk_mesh.cpp
  vma_impl.cpp
  stb_impl.cpp)

set_property(TARGET eugene PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:eugene>)

target_include_directories(eugene PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${SDL2_INCLUDE_DIRS}")

target_link_libraries(eugene vkbootstrap vma glm tinyobjloader stb) # imgui
target_link_libraries(eugene Vulkan::Vulkan SDL2::SDL2)

set_target_properties(eugene PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

target_compile_options(eugene PUBLIC "-Wall" "-Wextra")

find_package(Vulkan REQUIRED)

add_library(vkbootstrap STATIC)
add_library(glm INTERFACE)
add_library(vma INTERFACE)
#add_library(stb_image INTERFACE)

#add_library(tinyobjloader STATIC)

target_sources(vkbootstrap PRIVATE
  vk-bootstrap/src/VkBootstrap.h
  vk-bootstrap/src/VkBootstrapDispatch.h
  vk-bootstrap/src/VkBootstrap.cpp
)

target_include_directories(vkbootstrap PUBLIC vk-bootstrap/src)
target_link_libraries(vkbootstrap PUBLIC Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)

target_include_directories(glm INTERFACE glm/)
target_include_directories(vma INTERFACE vma/include)
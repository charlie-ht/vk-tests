#include "vk_engine.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.hpp>
#include <vk_types.hpp>

#include "VkBootstrap.h"

#define VMA_VULKAN_VERSION 1001000 // Vulkan 1.1
#include "vk_mem_alloc.h"

#define VK_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        VkResult err = x;                                                                                              \
        if (err) {                                                                                                     \
            std::cout << "Detected Vulkan error: " << err << std::endl;                                                \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

void VulkanEngine::init()
{
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                               _windowExtent.height, window_flags);

    init_vulkan();
    init_swapchain();

    // everything went fine
    _isInitialized = true;
}

inline VKAPI_ATTR VkBool32 VKAPI_CALL eugene_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                            VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                            void*)
{
    auto ms = vkb::to_string_message_severity(messageSeverity);
    auto mt = vkb::to_string_message_type(messageType);
    printf("[%s: %s]\n%s\n", ms, mt, pCallbackData->pMessage);

    return VK_FALSE; // Applications must return false here
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;
    // make the Vulkan instance, with basic debug features
    auto                 inst_ret = builder.set_app_name("Eugene Rendering Experiment")
                        .request_validation_layers(true)
                        .require_api_version(1, 1, 0)
                        .set_debug_callback(eugene_debug_callback)
                        .build();
    vkb::Instance vkb_inst = inst_ret.value();
    // store the instance
    _instance = vkb_inst.instance;
    // store the debug messenger
    _debug_messenger = vkb_inst.debug_messenger;

    // get the surface of the window we opened with SDL
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);
    // use vkbootstrap to select a GPU.
    // We want a GPU that can write to the SDL surface and supports Vulkan 1.1
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(1, 1).set_surface(_surface).select().value();

    // create the final Vulkan device
    vkb::DeviceBuilder  deviceBuilder{physicalDevice};

    vkb::Device         vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a Vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;
}

void VulkanEngine::init_swapchain()
{
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

    vkb::Swapchain        vkbSwapchain = swapchainBuilder
                                      .use_default_format_selection()
                                      // use vsync present mode
                                      .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                                      .set_desired_extent(_windowExtent.width, _windowExtent.height)
                                      .build()
                                      .value();

    // store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();

    _swapchainImageFormat = vkbSwapchain.image_format;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) {
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);

        // destroy swapchain resources
        for (int i = 0; i < _swapchainImageViews.size(); i++) {

            vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
        }

        vkDestroyDevice(_device, nullptr);
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
        SDL_DestroyWindow(_window);
    }
    SDL_DestroyWindow(_window);
}

void VulkanEngine::draw()
{
    // nothing yet
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool      bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT) {
                bQuit = true;
            }
        }

        draw();
    }
}
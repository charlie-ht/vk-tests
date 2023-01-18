#include "vk_engine.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.hpp>
#include <vk_types.hpp>

#include "VkBootstrap.h"
#include "glm/gtx/transform.hpp"

#include "stb_image_write.h"
#include "vk_mem_alloc.h"

#include <iostream>
#include <string_view>
#include <unistd.h>

#include <unistd.h>

#define VK_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        VkResult err = x;                                                                                              \
        if (err) {                                                                                                     \
            fprintf(stderr, "Detected Vulkan error: %d\n", err);                                                       \
        }                                                                                                              \
    } while (0)

EngineOptions VulkanEngine::parse_options(int argc, char** argv)
{
    EngineOptions            options;

    std::vector<std::string> args;
    args.assign(argv + 1, argv + argc);

    struct OptionParsed {
        bool             isShort;
        bool             isLong;
        std::string      original;
        std::string_view name;
        std::string_view value;
    };

    auto parse_arg = [=](std::string rawOpt) {
        OptionParsed p;
        p.original = rawOpt;
        if (p.original.starts_with("--")) {
            p.isShort = false;
            p.isLong = true;
            if (auto eqPosition = p.original.find('='); eqPosition != std::string_view::npos) {
                p.name = p.original.substr(2, eqPosition - 2);
                p.value = p.original.substr(eqPosition + 1);
            }
            else {
                p.name = p.original.substr(2);
            }
        }
        else if (rawOpt.starts_with("-")) {
            p.isShort = true;
            p.isLong = false;
            p.name = p.original.substr(1);
        }
        return p;
    };

    auto caseInsensitiveCompare = [](std::string_view a, std::string_view b) {
        return std::equal(a.begin(), a.end(), b.begin(),
                          [](char a, char b) { return std::toupper(a) == std::toupper(b); });
    };
    auto parseBool = [=](const std::string_view val) {
        if (val.length() == 0) {
            return false;
        }
        if (caseInsensitiveCompare(val, "true") || caseInsensitiveCompare(val, "1") ||
            caseInsensitiveCompare(val, "on")) {
            return true;
        }
        else if (caseInsensitiveCompare(val, "false") || caseInsensitiveCompare(val, "0") ||
                 caseInsensitiveCompare(val, "off")) {
            return false;
        }

        abort();
        return false;
    };

    for (const auto& arg : args) {
        const auto& opt = parse_arg(arg);
        if (opt.name == "validation") {
            options.enableValidation = parseBool(opt.value);
        }
        else if (opt.name == "sync-validation") {
            options.enableSyncValidation = parseBool(opt.value);
        }
        else if (opt.name == "best-practices-validation") {
            options.enableBestPracticesValidation = parseBool(opt.value);
        }
    }

    return options;
}

void VulkanEngine::init(int argc, char** argv)
{
    _parsedOptions = parse_options(argc, argv);
    if (_parsedOptions.enableValidation) {
        std::cout << "Validation enabled" << std::endl;
    }
    if (_parsedOptions.enableSyncValidation) {
        std::cout << "Sync validation enabled" << std::endl;
    }
    if (_parsedOptions.enableBestPracticesValidation) {
        std::cout << "Best practises validation enabled" << std::endl;
    }

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                               _windowExtent.height, window_flags);

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_pipelines();
    load_meshes();
    init_scene();

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

    // Don't need to be told this, it's obvious when I'm running debug builds.
    if (strstr(pCallbackData->pMessage, "VK_EXT_debug_utils")) {
        return VK_FALSE;
    }
    // Don't need to be told this, it's obvious when I'm running debug builds.
    if (strstr(pCallbackData->pMessage,
               "Using debug builds of the validation layers *will* adversely affect performance")) {
        return VK_FALSE;
    }

    printf("[%s: %s]\n%s\n", ms, mt, pCallbackData->pMessage);

    if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        abort();
    }

    return VK_FALSE; // Applications must return false here
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;
    // make the Vulkan instance, with basic debug features
    builder.set_app_name("Vulkan Experiment");

    if (_parsedOptions.enableValidation) {
        builder.request_validation_layers(true).set_debug_callback(eugene_debug_callback);
    }
    if (_parsedOptions.enableSyncValidation) {
        builder.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
    }
    if (_parsedOptions.enableBestPracticesValidation) {
        builder.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
    }
    builder.require_api_version(1, 3, 0);

    auto system_info_ret = vkb::SystemInfo::get_system_info();
    if (!system_info_ret) {
        printf("init_vulkan: %s\n", system_info_ret.error().message().c_str());
        abort();
    }

    vkb::Instance vkb_inst = builder.build().value();
    // store the instance
    _instance = vkb_inst.instance;
    // store the debug messenger
    _debug_messenger = vkb_inst.debug_messenger;

    // get the surface of the window we opened with SDL
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);
    // use vkbootstrap to select a GPU.
    // We want a GPU that can write to the SDL surface and supports Vulkan 1.3
    vkb::PhysicalDeviceSelector selector{vkb_inst};
    selector.set_minimum_version(1, 3).set_surface(_surface);

    vkb::PhysicalDevice                      physicalDevice = selector.select().value();

    // create the final Vulkan device
    vkb::DeviceBuilder                       deviceBuilder{physicalDevice};

    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures = {};
    dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dynamicRenderingFeatures.pNext = nullptr;
    dynamicRenderingFeatures.dynamicRendering = VK_TRUE;
    deviceBuilder.add_pNext(&dynamicRenderingFeatures);

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a Vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // use vkbootstrap to get a Graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() { vmaDestroyAllocator(_allocator); });
}

void VulkanEngine::init_swapchain()
{
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

    vkb::Swapchain        vkbSwapchain =
        swapchainBuilder
            .use_default_format_selection()
            // use vsync present mode
            .set_desired_format(VkSurfaceFormatKHR{VK_FORMAT_R8G8B8A8_UINT, VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT})
            .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(_windowExtent.width, _windowExtent.height)
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            .build()
            .value();

    fprintf(stderr, "Swapchain format %d\n", vkbSwapchain.image_format);

    // store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();

    _swapchainImageFormat = vkbSwapchain.image_format;

    _mainDeletionQueue.push_function([=]() { vkDestroySwapchainKHR(_device, _swapchain, nullptr); });

    // create the multisample image (where the fragment shader will be invoked multiple times per pixel, and these
    // extra samples recorded in the this image)
    VkExtent3D        msImageExtent = {_windowExtent.width, _windowExtent.height, 1};

    // The multi-sample image needs xfer-src since it will be the src when copying to the resolved, single-sample,
    // color buffer. Perhaps after average this sub-pixel samples, for example.
    VkImageCreateInfo msImageInfo = vkinit::image_create_info(
        _swapchainImageFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, msImageExtent,
        _sampleCount);

    VmaAllocationCreateInfo msimgAllocinfo = {};
    msimgAllocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    msimgAllocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    vmaCreateImage(_allocator, &msImageInfo, &msimgAllocinfo, &_msImage._image, &_msImage._allocation, nullptr);

    // build an image-view for the multisample image to use for rendering
    VkImageViewCreateInfo msViewInfo =
        vkinit::imageview_create_info(_swapchainImageFormat, _msImage._image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &msViewInfo, nullptr, &_msImageView));

    // add to deletion queues
    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _msImageView, nullptr);
        vmaDestroyImage(_allocator, _msImage._image, _msImage._allocation);
    });
    // end creation of multisample image

    // Create an intermediate image to resolve into. This will be copied to the resolveCheckBuffer below,
    // and later copied to the swapchain image for presentation

    VkImageCreateInfo resolveImageInfo = vkinit::image_create_info(
        _swapchainImageFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, msImageExtent,
        VK_SAMPLE_COUNT_1_BIT);

    VmaAllocationCreateInfo resolveImgAllocinfo = {};
    resolveImgAllocinfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    resolveImgAllocinfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    resolveImgAllocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    // allocate and create the image
    vmaCreateImage(_allocator, &resolveImageInfo, &resolveImgAllocinfo, &_resolveImage._image,
                   &_resolveImage._allocation, nullptr);

    // build an image-view for the multisample image to use for rendering
    VkImageViewCreateInfo resolveViewInfo =
        vkinit::imageview_create_info(_swapchainImageFormat, _resolveImage._image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &resolveViewInfo, nullptr, &_resolveImageView));

    // add to deletion queues
    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _resolveImageView, nullptr);
        vmaDestroyImage(_allocator, _resolveImage._image, _resolveImage._allocation);
    });
    // End of intermediate image creation

    // create a buffer to hold the temporary state of the MS image
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = nullptr;
    bufferCreateInfo.flags = 0u;
    bufferCreateInfo.size = _windowExtent.width * _windowExtent.height * 4;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0u;
    bufferCreateInfo.pQueueFamilyIndices = nullptr;
    VmaAllocationCreateInfo bufferAllocInfo = {};
    bufferAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    bufferAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    vmaCreateBuffer(_allocator, &bufferCreateInfo, &bufferAllocInfo, &_resolveCheckBuffer._buffer,
                    &_resolveCheckBuffer._allocation, nullptr);
    // end buffer creation code

    // depth image size will match the window
    VkExtent3D depthImageExtent = {_windowExtent.width, _windowExtent.height, 1};
    // hardcoding the depth format to 32 bit float
    _depthFormat = VK_FORMAT_D32_SFLOAT;

    // the depth image will be an image with the format we selected and Depth Attachment usage flag
    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                                            depthImageExtent, _sampleCount);

    // for the depth image, we want to allocate it from GPU local memory
    VmaAllocationCreateInfo dimg_allocinfo = {};
    dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    dimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image
    vmaCreateImage(_allocator, &dimg_info, &dimg_allocinfo, &_depthImage._image, &_depthImage._allocation, nullptr);

    // build an image-view for the depth image to use for rendering
    VkImageViewCreateInfo dview_info =
        vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

    // add to deletion queues
    _mainDeletionQueue.push_function([=]() {
        vmaDestroyBuffer(_allocator, _resolveCheckBuffer._buffer, _resolveCheckBuffer._allocation);
        for (const auto& iview : _swapchainImageViews) {
            vkDestroyImageView(_device, iview, nullptr);
        }
        vkDestroyImageView(_device, _depthImageView, nullptr);
        vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
    });
}

void VulkanEngine::init_commands()
{
    // create a command pool for commands submitted to the graphics queue.
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

    // allocate the default command buffer that we will use for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_commandPool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));

    _mainDeletionQueue.push_function([=]() { vkDestroyCommandPool(_device, _commandPool, nullptr); });
}

void VulkanEngine::init_sync_structures()
{
    // create synchronization structures

    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = nullptr;

    // we want to create the fence with the Create Signaled flag, so we can wait on it before using it on a GPU
    // command (for the first frame)
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));

    // for the semaphores we don't need any flags
    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = nullptr;
    semaphoreCreateInfo.flags = 0;

    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_presentSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphore));

    // enqueue the destruction of semaphores
    _mainDeletionQueue.push_function([=]() {
        vkDestroyFence(_device, _renderFence, nullptr);
        vkDestroySemaphore(_device, _presentSemaphore, nullptr);
        vkDestroySemaphore(_device, _renderSemaphore, nullptr);
    });
}

void VulkanEngine::init_pipelines()
{
    // compile colored triangle modules
    VkShaderModule triangleFragShader;
    if (!load_shader_module("../../shaders/colored_triangle.frag.spv", &triangleFragShader)) {
        std::cerr << "Error when building the triangle fragment shader module" << std::endl;
        abort();
    }
    else {
        std::cout << "Triangle fragment shader successfully loaded" << std::endl;
    }

    VkShaderModule triangleVertexShader;
    if (!load_shader_module("../../shaders/colored_triangle.vert.spv", &triangleVertexShader)) {
        std::cerr << "Error when building the triangle vertex shader module" << std::endl;
        abort();
    }
    else {
        std::cout << "Triangle vertex shader successfully loaded" << std::endl;
    }

    // compile red triangle modules
    VkShaderModule redTriangleFragShader;
    if (!load_shader_module("../../shaders/triangle.frag.spv", &redTriangleFragShader)) {
        std::cerr << "Error when building the triangle fragment shader module" << std::endl;
        abort();
    }
    else {
        std::cout << "Red Triangle fragment shader successfully loaded" << std::endl;
    }

    VkShaderModule redTriangleVertShader;
    if (!load_shader_module("../../shaders/triangle.vert.spv", &redTriangleVertShader)) {
        std::cerr << "Error when building the triangle vertex shader module" << std::endl;
        abort();
    }
    else {
        std::cout << "Red Triangle vertex shader successfully loaded" << std::endl;
    }

    VkShaderModule meshVertShader;
    if (!load_shader_module("../../shaders/tri_mesh.vert.spv", &meshVertShader)) {
        std::cerr << "Error when building the mesh triangle vertex shader module" << std::endl;
        abort();
    }
    else {
        std::cout << "Mesh Triangle vertex shader successfully loaded" << std::endl;
    }
    VkShaderModule meshFragShader;
    if (!load_shader_module("../../shaders/tri_mesh.frag.spv", &meshFragShader)) {
        std::cerr << "Error when building the mesh triangle fragment shader module" << std::endl;
        abort();
    }
    else {
        std::cout << "Mesh Triangle fragment shader successfully loaded" << std::endl;
    }

    // build the pipeline layout that controls the inputs/outputs of the shader
    // we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));

    // build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader
    // modules per stage
    PipelineBuilder pipelineBuilder;

    pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);

    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, triangleVertexShader));

    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

    // vertex input controls how to read vertices from vertex buffers. We aren't using it yet
    pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();

    // input assembly is the configuration for drawing triangle lists, strips, or individual points.
    // we are just going to draw triangle list
    pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // build viewport and scissor from the swapchain extents
    pipelineBuilder._viewport.x = 0.0f;
    pipelineBuilder._viewport.y = 0.0f;
    pipelineBuilder._viewport.width = (float)_windowExtent.width;
    pipelineBuilder._viewport.height = (float)_windowExtent.height;
    pipelineBuilder._viewport.minDepth = 0.0f;
    pipelineBuilder._viewport.maxDepth = 1.0f;

    pipelineBuilder._scissor.offset = {0, 0};
    pipelineBuilder._scissor.extent = _windowExtent;

    // configure the rasterizer to draw filled triangles
    pipelineBuilder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);

    // we don't use multisampling, so just run the default one
    pipelineBuilder._multisampling = vkinit::multisampling_state_create_info(_sampleCount);

    // a single blend attachment with no blending and writing to RGBA
    pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();

    // use the triangle layout we created
    pipelineBuilder._pipelineLayout = _trianglePipelineLayout;

    // finally build the pipeline
    _trianglePipeline = pipelineBuilder.build_pipeline(_device, _swapchainImageFormat, _depthFormat);

    // clear the shader stages for the builder
    pipelineBuilder._shaderStages.clear();

    // add the other shaders
    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, redTriangleVertShader));

    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, redTriangleFragShader));

    // build the red triangle pipeline
    _redTrianglePipeline = pipelineBuilder.build_pipeline(_device, _swapchainImageFormat, _depthFormat);

    // build the mesh pipeline
    VertexInputDescription vertexDescription = Vertex::get_vertex_description();

    // connect the pipeline builder vertex input info to the one we get from Vertex
    pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
    pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

    pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
    pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

    // clear the shader stages for the builder
    pipelineBuilder._shaderStages.clear();

    // compile mesh vertex shader
    // we start from just the default empty pipeline layout info
    VkPipelineLayoutCreateInfo mesh_pipeline_layout_info = vkinit::pipeline_layout_create_info();
    // setup push constants
    VkPushConstantRange        push_constant;
    // this push constant range starts at the beginning
    push_constant.offset = 0;
    // this push constant range takes up the size of a MeshPushConstants struct
    push_constant.size = sizeof(MeshPushConstants);
    // this push constant range is accessible in the vertex and fragment stages
    push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
    mesh_pipeline_layout_info.pushConstantRangeCount = 1;
    VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info, nullptr, &_meshPipelineLayout));

    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));
    pipelineBuilder._shaderStages.push_back(
        vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, meshFragShader));

    pipelineBuilder._pipelineLayout = _meshPipelineLayout;

    _meshPipeline = pipelineBuilder.build_pipeline(_device, _swapchainImageFormat, _depthFormat);

    create_material(_meshPipeline, _meshPipelineLayout, "defaultmesh");

    // deleting all of the vulkan shaders
    vkDestroyShaderModule(_device, meshVertShader, nullptr);
    vkDestroyShaderModule(_device, meshFragShader, nullptr);
    vkDestroyShaderModule(_device, redTriangleVertShader, nullptr);
    vkDestroyShaderModule(_device, redTriangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);

    // adding the pipelines to the deletion queue
    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipeline(_device, _redTrianglePipeline, nullptr);
        vkDestroyPipeline(_device, _trianglePipeline, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);

        vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
    });
}

void VulkanEngine::load_meshes()
{
    // make the array 3 vertices long
    _triangleMesh._vertices.resize(3);

    // vertex positions
    _triangleMesh._vertices[0].position = {1.f, 1.f, 0.0f};
    _triangleMesh._vertices[1].position = {-3.f, 1.f, 0.0f};
    _triangleMesh._vertices[2].position = {0.f, -3.f, 0.0f};

    // vertex colors, all green
    _triangleMesh._vertices[0].color = {0.f, 1.f, 0.0f}; // pure green
    _triangleMesh._vertices[1].color = {0.f, 1.f, 0.0f}; // pure green
    _triangleMesh._vertices[2].color = {0.f, 1.f, 0.0f}; // pure green

    // load the monkey
    _monkeyMesh.load_from_obj("../../../assets/monkey.obj");

    // we don't care about the vertex normals
    upload_mesh(_triangleMesh);
    upload_mesh(_monkeyMesh);

    // note that we are copying them. Eventually we will delete the hardcoded _monkey and _triangle meshes, so it's
    // no problem now.
    _meshes["monkey"] = _monkeyMesh;
    _meshes["triangle"] = _triangleMesh;
}

void VulkanEngine::init_scene()
{
    RenderObject monkey;
    monkey.mesh = get_mesh("monkey");
    monkey.material = get_material("defaultmesh");
    monkey.transformMatrix = glm::translate(glm::mat4{1.0}, glm::vec3(0.0f, 6.0f, 0.0f));

    _renderables.push_back(monkey);

    for (int x = -20; x <= 20; x++) {
        for (int y = -20; y <= 20; y++) {

            RenderObject tri;
            tri.mesh = get_mesh("triangle");
            tri.material = get_material("defaultmesh");
            glm::mat4 translation = glm::translate(glm::mat4{1.0}, glm::vec3(x, 0, y));
            glm::mat4 scale = glm::scale(glm::mat4{1.0}, glm::vec3(0.2, 0.2, 0.2));
            tri.transformMatrix = translation * scale;

            _renderables.push_back(tri);
        }
    }
}

void VulkanEngine::upload_mesh(Mesh& mesh)
{
    // allocate vertex buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    // this is the total size, in bytes, of the buffer we are allocating
    bufferInfo.size = mesh._vertices.size() * sizeof(Vertex);
    // this buffer is going to be used as a Vertex Buffer
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    // let the VMA library know that this data should be writeable by CPU, but also readable by GPU
    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &mesh._vertexBuffer._buffer,
                             &mesh._vertexBuffer._allocation, nullptr));

    // add the destruction of triangle mesh buffer to the deletion queue
    _mainDeletionQueue.push_function(
        [=]() { vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation); });

    void* data;
    vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data);

    memcpy(data, mesh._vertices.data(), mesh._vertices.size() * sizeof(Vertex));

    vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation);
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) {
        // make sure the gpu has stopped doing its things
        vkDeviceWaitIdle(_device);

        _mainDeletionQueue.flush();

        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyDevice(_device, nullptr);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }
    SDL_DestroyWindow(_window);
}

VkImageMemoryBarrier make_image_barrier(VkImage image, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask,
                                        VkImageLayout srcLayout, VkImageLayout dstLayout)
{
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = srcLayout;
    barrier.newLayout = dstLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    return barrier;
}

void copyColorImageToBuffer(VkCommandBuffer cmd, VkImage image, uint32_t width, uint32_t height, VkBuffer buffer)
{
    VkImageMemoryBarrier imageBarrier =
        make_image_barrier(image, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 0u,
                         nullptr, 0u, nullptr, 1u, &imageBarrier);

    VkImageSubresourceLayers subresource = {};
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel = 0u;
    subresource.baseArrayLayer = 0u;
    subresource.layerCount = 1u;

    VkBufferImageCopy region = {};
    region.bufferOffset = 0ull;
    region.bufferRowLength = 0u;
    region.bufferImageHeight = 0u;
    region.imageSubresource = subresource;
    region.imageOffset = VkOffset3D{0u, 0u, 0u};
    region.imageExtent = VkExtent3D{width, height, 1u};

    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);

    imageBarrier = make_image_barrier(image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0u, 0u,
                         nullptr, 0u, nullptr, 1u, &imageBarrier);
}

void copyColorImageToSwapchainImage(VkCommandBuffer cmd, VkImage resolvedImage, uint32_t width, uint32_t height,
                                    VkImage swapchainImage)
{
    VkImageSubresourceLayers subresource;
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel = 0u;
    subresource.baseArrayLayer = 0u;
    subresource.layerCount = 1u;

    VkImageCopy imageCopy = {};
    imageCopy.srcSubresource = subresource;
    imageCopy.srcOffset = VkOffset3D{0u, 0u, 0u};
    imageCopy.dstSubresource = subresource;
    imageCopy.dstOffset = VkOffset3D{0u, 0u, 0u};
    imageCopy.extent = VkExtent3D{width, height, 1u};

    auto swapchainImageMemoryBarrier =
        make_image_barrier(swapchainImage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &swapchainImageMemoryBarrier);

    auto resolveImageMemoryBarrier =
        make_image_barrier(resolvedImage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &resolveImageMemoryBarrier);

    vkCmdCopyImage(cmd, resolvedImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);

    resolveImageMemoryBarrier =
        make_image_barrier(resolvedImage, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &resolveImageMemoryBarrier);

    swapchainImageMemoryBarrier =
        make_image_barrier(swapchainImage, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &swapchainImageMemoryBarrier);
}

void VulkanEngine::draw()
{
    // wait until the GPU has finished rendering the last frame. Timeout of 1 second
    VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, 1000000000));

    VK_CHECK(vkResetFences(_device, 1, &_renderFence));
    // request image from the swapchain, one second timeout
    uint32_t swapchainImageIndex;
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, _presentSemaphore, nullptr, &swapchainImageIndex));
    // now that we are sure that the commands finished executing, we can safely reset the command buffer to begin
    // recording again.
    VK_CHECK(vkResetCommandPool(_device, _commandPool, 0));
    // VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));
    //  naming it cmd for shorter writing
    VkCommandBuffer          cmd = _mainCommandBuffer;

    // begin the command buffer recording. We will use this command buffer exactly once, so we want to let Vulkan
    // know that
    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.pNext = nullptr;

    cmdBeginInfo.pInheritanceInfo = nullptr;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    // make a clear-color from frame number. This will flash with a 120*pi frame period.
    VkClearValue clearValue;
    float        flash = abs(sin(_frameNumber / 24.f));
    clearValue.color = {{0.0f, 0.0f, flash, 1.0f}};

    VkClearValue depthClear;
    depthClear.depthStencil.depth = 1.0f;

    // start the main renderpass.
    VkRenderingAttachmentInfo colorAttachmentInfo = {};

    if (_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        colorAttachmentInfo =
            vkinit::create_resolve_attachment(_msImageView, VK_RESOLVE_MODE_AVERAGE_BIT, _resolveImageView, clearValue);
    }
    else {
        colorAttachmentInfo = vkinit::create_color_attachment(_msImageView, clearValue);
    }
    VkRenderingAttachmentInfo depthAttachmentInfo = vkinit::create_depth_attachment(_depthImageView, depthClear);

    VkRenderingAttachmentInfo colorAttachments[1] = {colorAttachmentInfo};

    VkRenderingInfo           renderingInfo = {};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.flags = 0;
    renderingInfo.renderArea = {{0, 0}, _windowExtent};
    renderingInfo.layerCount = 1;
    renderingInfo.viewMask = 0;
    renderingInfo.colorAttachmentCount = SDL_arraysize(colorAttachments);
    renderingInfo.pColorAttachments = colorAttachments;
    renderingInfo.pDepthAttachment = &depthAttachmentInfo;
    renderingInfo.pStencilAttachment = nullptr;

    // Pre-barriers, get resources in the right state for drawing.
    VkImageMemoryBarrier swapchainImageMemoryBarrier =
        make_image_barrier(_swapchainImages[swapchainImageIndex], 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &swapchainImageMemoryBarrier);

    VkImageMemoryBarrier resolveImageMemoryBarrier =
        make_image_barrier(_resolveImage._image, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &resolveImageMemoryBarrier);

    VkImageMemoryBarrier msImageMemoryBarrier =
        make_image_barrier(_msImage._image, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &msImageMemoryBarrier);
    // End of pre barriers

    vkCmdBeginRendering(cmd, &renderingInfo);

    if (_selectedShader == 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _trianglePipeline);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
    else if (_selectedShader == 1) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _redTrianglePipeline);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
    else if (_selectedShader == 2) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);

        // make a model view matrix for rendering the object
        // camera position
        glm::vec3 camPos = {0.f, 0.f, -2.f};

        glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
        // camera projection
        glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
        projection[1][1] *= -1;
        // model rotation
        glm::mat4         model = glm::rotate(glm::mat4{1.0f}, glm::radians(_frameNumber * 0.4f), glm::vec3(0, 1, 0));

        // calculate final mesh matrix
        glm::mat4         mesh_matrix = projection * view * model;

        MeshPushConstants constants;
        constants.render_matrix = mesh_matrix;
        int mouse_x = 0;
        int mouse_y = 0;
        SDL_GetMouseState(&mouse_x, &mouse_y);
        constants.resolution_and_mouse = glm::vec4(_windowExtent.width, _windowExtent.height, mouse_x, mouse_y);
        constants.ticks = glm::vec4(SDL_GetTicks() / 400.0f, 0.0f, 0.0f, 0.0f);

        // upload the matrix to the GPU via push constants
        vkCmdPushConstants(cmd, _meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(MeshPushConstants), &constants);

        // bind the mesh vertex buffer with offset 0
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &_monkeyMesh._vertexBuffer._buffer, &offset);
        // we can now draw the mesh
        vkCmdDraw(cmd, _monkeyMesh._vertices.size(), 1, 0, 0);
    }
    else {
        draw_objects(cmd, _renderables.data(), _renderables.size());
    }

    // finalize the render pass
    vkCmdEndRendering(cmd);

    if (_sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        copyColorImageToBuffer(cmd, _resolveImage._image, _windowExtent.width, _windowExtent.height,
                               _resolveCheckBuffer._buffer);

        copyColorImageToSwapchainImage(cmd, _resolveImage._image, _windowExtent.width, _windowExtent.height,
                                       _swapchainImages[swapchainImageIndex]);
    }
    else {
        copyColorImageToSwapchainImage(cmd, _msImage._image, _windowExtent.width, _windowExtent.height,
                                       _swapchainImages[swapchainImageIndex]);
    }

    swapchainImageMemoryBarrier = make_image_barrier(
        _swapchainImages[swapchainImageIndex], VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &swapchainImageMemoryBarrier);

    //  finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));

    // prepare the submission to the queue.
    // we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    // we will signal the _renderSemaphore, to signal that rendering has finished

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = nullptr;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    submit.pWaitDstStageMask = &waitStage;

    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &_presentSemaphore;

    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &_renderSemaphore;

    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _renderFence));

    // this will put the image we just rendered into the visible window.
    // we want to wait on the _renderSemaphore for that,
    // as it's necessary that drawing commands have finished before the image is displayed to the user
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;

    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;

    presentInfo.pWaitSemaphores = &_renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;

    presentInfo.pImageIndices = &swapchainImageIndex;

    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

    // increase the number of frames drawn
    _frameNumber++;
}

void VulkanEngine::draw_objects(VkCommandBuffer cmd, RenderObject* first, int count)
{
    // make a model view matrix for rendering the object
    // camera view
    glm::vec3 camPos = {0.f, -6.f, -10.f};

    glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
    // camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
    projection[1][1] *= -1;

    Mesh*     lastMesh = nullptr;
    Material* lastMaterial = nullptr;
    for (int i = 0; i < count; i++) {
        RenderObject& object = first[i];

        // only bind the pipeline if it doesn't match with the already bound one
        if (object.material != lastMaterial) {

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
            lastMaterial = object.material;
        }

        glm::mat4 model = object.transformMatrix;

        model *= glm::rotate(glm::mat4{1.0f}, glm::radians(_frameNumber * 0.4f), glm::vec3(0, 1, 0));

        // final render matrix, that we are calculating on the cpu
        glm::mat4         mesh_matrix = projection * view * model;

        MeshPushConstants constants;
        constants.render_matrix = mesh_matrix;

        // upload the mesh to the GPU via push constants
        vkCmdPushConstants(cmd, object.material->pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(MeshPushConstants),
                           &constants);

        // only bind the mesh if it's a different one from last bind
        if (object.mesh != lastMesh) {
            // bind the mesh vertex buffer with offset 0
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);
            lastMesh = object.mesh;
        }
        // we can now draw
        vkCmdDraw(cmd, object.mesh->_vertices.size(), 1, 0, 0);
    }
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
            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_SPACE) {
                    _selectedShader += 1;
                    if (_selectedShader > 3) {
                        _selectedShader = 0;
                    }
                }
            }
        }

        draw();
    }
}

bool VulkanEngine::load_shader_module(const char* filePath, VkShaderModule* outShaderModule)
{
    FILE* file = fopen(filePath, "rb");
    if (!file) {
        return false;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    assert(length >= 0);
    fseek(file, 0, SEEK_SET);

    char* buffer = new char[length];
    assert(buffer);

    size_t rc = fread(buffer, 1, length, file);
    assert(rc == size_t(length));
    fclose(file);

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.codeSize = length;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer);

    VkShaderModule shaderModule = 0;
    if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return false;
    }
    *outShaderModule = shaderModule;
    delete[] buffer;

    return true;
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkFormat swapchainFormat, VkFormat depthFormat)
{
    // make viewport state from our stored viewport and scissor.
    // at the moment we won't support multiple viewports or scissors
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pNext = nullptr;

    viewportState.viewportCount = 1;
    viewportState.pViewports = &_viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &_scissor;

    // setup dummy color blending. We aren't using transparent objects yet
    // the blending is just "no blend", but we do write to the color attachment
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.pNext = nullptr;

    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &_colorBlendAttachment;

    // build the actual pipeline
    // we now use all of the info structs we have been writing into into this one to create the pipeline
    VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo = {};
    pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingCreateInfo.viewMask = 0;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat;
    pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat;
    pipelineRenderingCreateInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &pipelineRenderingCreateInfo;

    pipelineInfo.stageCount = _shaderStages.size();
    pipelineInfo.pStages = _shaderStages.data();
    pipelineInfo.pVertexInputState = &_vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &_inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &_rasterizer;
    pipelineInfo.pMultisampleState = &_multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &_depthStencil;
    pipelineInfo.layout = _pipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    // it's easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK
    // case
    VkPipeline newPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
        fprintf(stderr, "failed to create pipeline\n");
        return VK_NULL_HANDLE; // failed to create graphics pipeline
    }
    else {
        return newPipeline;
    }
}

Material* VulkanEngine::create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name)
{
    Material mat;
    mat.pipeline = pipeline;
    mat.pipelineLayout = layout;
    _materials[name] = mat;
    return &_materials[name];
}

Material* VulkanEngine::get_material(const std::string& name)
{
    // search for the object, and return nullptr if not found
    if (auto it = _materials.find(name); it != _materials.end()) {
        return &(*it).second;
    }
    return nullptr;
}

Mesh* VulkanEngine::get_mesh(const std::string& name)
{
    if (auto it = _meshes.find(name); it != _meshes.end()) {
        return &(*it).second;
    }
    return nullptr;
}

EngineOptions parse_options(int argc, char** argv);

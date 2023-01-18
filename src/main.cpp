#include "vk_engine.hpp"

#include <cstdlib>
#include <string>

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    VulkanEngine engine;

    engine.init(argc, argv);

    engine.run();

    engine.cleanup();
    return 0;
}

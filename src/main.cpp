#include "vk_engine.hpp"

#include <cstdlib>
#include <string>

int Eu_Global_LogLevel = 0;

int main(int argc, char* argv[])
{
    if (auto eugeneDebugEnvVar = getenv("EUGENE_DEBUG"); eugeneDebugEnvVar != nullptr) {
        std::string value = std::string(eugeneDebugEnvVar);
        if (value == "1") {
            Eu_Global_LogLevel = 100;
        }
    }

    VulkanEngine engine;

    engine.init();

    engine.run();

    engine.cleanup();
    return 0;
}
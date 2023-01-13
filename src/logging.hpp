#pragma once

#include <cstdio>

extern int Eu_Global_LogLevel;

#define Eu_DEBUG(fmt, ...)                                                                                \
    do                                                                                                    \
    {                                                                                                     \
        if (Eu_Global_LogLevel > 10)                                                                      \
            fprintf(stderr, "EUGENE DEBUG: " FILE_BASENAME ":%d >>| " fmt "\n", __LINE__, ##__VA_ARGS__); \
    } while (0);

#define Eu_LOG(fmt, ...)                                                                        \
    do                                                                                          \
    {                                                                                           \
        fprintf(stderr, "EUGENE: " FILE_BASENAME ":%d >>| " fmt "\n", __LINE__, ##__VA_ARGS__); \
    } while (0);

#define Eu_ERROR(fmt, ...)                                                       \
    do                                                                           \
    {                                                                            \
        fprintf(stderr, "EUGENE: !!ERROR!! " FILE_BASENAME ":%d >>|", __LINE__); \
        fprintf(stderr, fmt "\n\t", ##__VA_ARGS__);                              \
        perror("posix error");                                                   \
    } while (0);

#define Eu_DIE(errcode, fmt, ...)                                          \
    do                                                                     \
    {                                                                      \
        fprintf(stderr, "DEATH: file " FILE_BASENAME ":%d >>|", __LINE__); \
        fprintf(stderr, fmt "\n\t", ##__VA_ARGS__);                        \
        perror("posix error");                                             \
        exit(errcode);                                                     \
    } while (0);
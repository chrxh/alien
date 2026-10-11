#pragma once

#include <memory>
#include <mutex>

#define MAKE_SINGLETON(ClassName) \
public: \
    static ClassName& get() \
    { \
        static std::unique_ptr<ClassName> instance; \
        static std::mutex mutex; \
        if (!instance) { \
            std::lock_guard<std::mutex> lock(mutex); \
            instance = std::unique_ptr<ClassName>(new ClassName); \
        } \
        return *instance.get(); \
    } \
\
private: \
    ClassName() = default; \
    ClassName(ClassName const&) = delete; \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName const&) = delete; \
    ClassName& operator=(ClassName&&) = delete

#define MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ClassName) \
public: \
    static ClassName& get() \
    { \
        static std::unique_ptr<ClassName> instance; \
        static std::mutex mutex; \
        if (!instance) { \
            std::lock_guard<std::mutex> lock(mutex); \
            instance = std::unique_ptr<ClassName>(new ClassName); \
        } \
        return *instance.get(); \
    } \
\
private: \
    ClassName(ClassName const&) = delete; \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName const&) = delete; \
    ClassName& operator=(ClassName&&) = delete
